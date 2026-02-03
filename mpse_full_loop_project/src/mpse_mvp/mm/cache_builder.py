
from __future__ import annotations
import os, json
import numpy as np
from tqdm import tqdm

from mpse_mvp.segment.io import load_wav
from mpse_mvp.mm.encoders import WhisperAudioEncoder, CLIPVideoEncoder, sample_video_frames

def _mu_dict_to_vec(mu: dict, idx_names: list[str]) -> np.ndarray:
    return np.array([float(mu.get(k, 0.0)) for k in idx_names], dtype=np.float32)

def build_mm_cache(session_id: str, mp4_path: str, wav_path: str,
                   turns_upgraded_path: str, sft_jsonl: str,
                   out_dir: str,
                   whisper_dir: str, clip_dir: str,
                   idx_names: list[str],
                   n_frames: int = 8,
                   device: str = "cpu"):
    """
    Creates per-turn .npz with pooled audio/video embeddings + alpha + mu,
    and an index jsonl with messages and sample_weight.
    """
    os.makedirs(out_dir, exist_ok=True)
    wav, sr = load_wav(wav_path)

    # load upgraded turns
    up = {r["turn_id"]: r for r in (json.loads(l) for l in open(turns_upgraded_path, encoding="utf-8"))}
    # load sft samples
    sft = [json.loads(l) for l in open(sft_jsonl, encoding="utf-8")]

    aenc = WhisperAudioEncoder(whisper_dir, device=device)
    venc = CLIPVideoEncoder(clip_dir, device=device)

    index_path = os.path.join(out_dir, "mm_index.jsonl")
    with open(index_path, "w", encoding="utf-8") as f:
        for s in tqdm(sft, desc="MM cache"):
            tid = int(s["meta"]["turn_id"])
            r = up.get(tid)
            if r is None:
                continue
            t0, t1 = float(r["t0"]), float(r["t1"])

            # audio slice
            s0 = int(t0 * sr); s1 = int(t1 * sr)
            wav_seg = wav[s0:s1].astype(np.float32)

            audio_pooled, _ = aenc.encode(wav_seg, sr=sr, return_sequence=False)
            audio_feat = audio_pooled.detach().cpu().numpy().reshape(-1).astype(np.float32)

            # video frames
            frames = sample_video_frames(mp4_path, t0, t1, n_frames=n_frames)
            video_pooled, _ = venc.encode(frames, return_sequence=False)
            video_feat = video_pooled.detach().cpu().numpy().reshape(-1).astype(np.float32)

            # alpha as 2-dim (audio, video)
            alpha_dict = r.get("alpha", {})
            a = float(alpha_dict.get("audio", alpha_dict.get("a", alpha_dict.get("A", 0.5))))
            v = float(alpha_dict.get("audio", alpha_dict.get("v", alpha_dict.get("V", 0.5))))
            alpha = np.array([a, v], dtype=np.float32)

            mu = _mu_dict_to_vec(r.get("mu", {}), idx_names)

            npz_path = os.path.join(out_dir, f"turn_{tid:04d}.npz")
            np.savez_compressed(npz_path, audio_feat=audio_feat, video_feat=video_feat, alpha=alpha, mu=mu)

            rec = {
                "npz_path": npz_path,
                "messages": s["messages"],
                "sample_weight": float(s.get("sample_weight", 1.0)),
                "meta": s.get("meta", {}),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return index_path
