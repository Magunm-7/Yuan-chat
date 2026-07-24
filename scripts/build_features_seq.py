"""
Per-turn SEQUENCE features for AnnoMI (for cross-attention fusion, route 2).

Same as build_features.py but keeps modality SEQUENCES (not pooled) so a fusion
module can cross-attend them:

  audio_seq : Whisper-small encoder states, valid frames -> adaptive-pooled to 64  (64, 768)
  video_seq : CLIP per-frame CLS, fixed 8 frames (zero-padded if fewer)            (8, 768)
  text_emb  : sentence-transformer of utterance_text                               (384,)
  + pooled audio_emb/video_emb (compat) + q_* + chg_weak

Design decisions (2026-07-23, autonomous while user asleep):
  - Whisper returns a fixed 1500-frame sequence (30s padded); most is silence
    padding, so we keep only the first ceil((t1-t0)*50) valid frames then
    adaptive-avg-pool the time axis to a FIXED 64 -> no variable-length masking.
  - Video fixed at 8 frames (n_frames), zero-pad if fewer.
  Fixed lengths keep the cross-attention module simple (no padding masks) for v1.

  python scripts/build_features_seq.py --limit 2   # smoke first
"""
from __future__ import annotations
import os, json, argparse, math
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default="data/annomi/turns_chg.jsonl")
    ap.add_argument("--video_dir", default="data/annomi/video")
    ap.add_argument("--wav_dir", default="data/annomi/wav")
    ap.add_argument("--out_dir", default="data/annomi/feats_seq")
    ap.add_argument("--text_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--n_frames", type=int, default=8)
    ap.add_argument("--audio_len", type=int, default=64, help="fixed pooled audio sequence length")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from mpse_mvp.segment.io import load_wav
    from mpse_mvp.mm.encoders import WhisperAudioEncoder, CLIPVideoEncoder, sample_video_frames
    from mpse_mvp.features.audio_features import audio_quality_and_prosody
    from mpse_mvp.features.text_features import text_quality
    from sentence_transformers import SentenceTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    rows = [json.loads(l) for l in open(args.turns, encoding="utf-8")]
    by = {}
    for r in rows:
        by.setdefault(r["session_id"], []).append(r)
    sids = [s for s in by
            if os.path.exists(os.path.join(args.wav_dir, f"{s}.wav"))
            and os.path.exists(os.path.join(args.video_dir, f"{s}.mp4"))]
    sids.sort(key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
    if args.limit:
        sids = sids[:args.limit]
    print(f"sessions with media: {len(sids)} (of {len(by)})")

    aenc = WhisperAudioEncoder("openai/whisper-small", device=dev)
    venc = CLIPVideoEncoder("openai/clip-vit-base-patch32", device=dev)
    tenc = SentenceTransformer(args.text_model, device=dev)

    def pool_audio(hs, dur):  # hs (1,1500,768) -> (audio_len,768) over valid frames
        T = hs.shape[1]
        n_valid = max(1, min(T, int(math.ceil(dur * 50))))
        seq = hs[0, :n_valid]                                   # (n_valid,768)
        seq = F.adaptive_avg_pool1d(seq.transpose(0, 1).unsqueeze(0), args.audio_len)
        return seq.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32)  # (audio_len,768)

    def pack_video(vhs):  # (N,50,768) -> (n_frames,768) CLS per frame, zero-pad
        if vhs is None:
            return np.zeros((args.n_frames, 768), np.float32)
        cls = vhs[:, 0, :]                                      # (N,768)
        N, C = cls.shape
        if N >= args.n_frames:
            cls = cls[:args.n_frames]
        else:
            cls = torch.cat([cls, torch.zeros(args.n_frames - N, C, device=cls.device)], 0)
        return cls.cpu().numpy().astype(np.float32)

    index_path = os.path.join(args.out_dir, "index.jsonl")
    n_turns = 0
    with open(index_path, "w", encoding="utf-8") as idx:
        for si, sid in enumerate(sids, 1):
            wav, sr = load_wav(os.path.join(args.wav_dir, f"{sid}.wav"))
            mp4 = os.path.join(args.video_dir, f"{sid}.mp4")
            turns = sorted(by[sid], key=lambda r: r["turn_id"])
            print(f"[{si}/{len(sids)}] session {sid}: {len(turns)} turns", flush=True)
            for r in turns:
                t0, t1 = float(r["t0"]), float(r["t1"])
                s0, s1 = int(t0 * sr), int(t1 * sr)
                seg = wav[s0:s1].astype(np.float32)
                if len(seg) < sr // 10:
                    seg = np.zeros(sr // 10, dtype=np.float32)
                q_audio, _ = audio_quality_and_prosody(seg, sr)
                a_pool, a_hs = aenc.encode(seg, sr=sr, return_sequence=True)
                audio_seq = pool_audio(a_hs, max(0.1, t1 - t0))
                audio_emb = a_pool.detach().cpu().numpy().reshape(-1).astype(np.float32)

                frames = sample_video_frames(mp4, t0, t1, n_frames=args.n_frames)
                q_video = float(len(frames)) / float(args.n_frames)
                v_pool, v_hs = venc.encode(frames, return_sequence=True)
                video_seq = pack_video(v_hs)
                video_emb = v_pool.detach().cpu().numpy().reshape(-1).astype(np.float32)

                text_emb = tenc.encode([r["text"]], show_progress_bar=False)[0].astype(np.float32)
                q_text = float(text_quality(r["text"]))

                npz = os.path.join(args.out_dir, f"{sid}_{r['turn_id']:03d}.npz")
                np.savez_compressed(
                    npz, audio_seq=audio_seq, video_seq=video_seq, text_emb=text_emb,
                    audio_emb=audio_emb, video_emb=video_emb,
                    q_text=q_text, q_audio=float(q_audio), q_video=q_video,
                    chg_weak=float(r.get("chg_weak", 0.0)),
                )
                idx.write(json.dumps({
                    "npz": npz, "session_id": sid, "turn_id": r["turn_id"],
                    "talk_type": r["talk_type"], "mi_quality": r["mi_quality"],
                }, ensure_ascii=False) + "\n")
                n_turns += 1

    print(f"\ndone: {n_turns} turns -> {args.out_dir}")
    print(f"dims: audio_seq=(64,768) video_seq=(8,768) text_emb={text_emb.shape[0]}")


if __name__ == "__main__":
    main()
