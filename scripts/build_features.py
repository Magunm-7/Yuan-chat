"""
Per-turn multimodal feature cache for AnnoMI.

For each client turn (bounded by [t0,t1] from AnnoMI timestamps) extract three
pooled modality embeddings + quality scalars, and carry the chg weak label
(training target) and gold talk_type / mi_quality (eval only):

  text_emb  : sentence-transformer of utterance_text        (semantic content)
  audio_emb : Whisper-small encoder, mean-pooled over slice  (prosody/voice)
  video_emb : CLIP ViT-B/32, CLS mean over sampled frames    (facial/visual)
  q_text/q_audio/q_video : per-modality quality (for weighting + alpha)

Writes one .npz per turn under out_dir + an index jsonl. Sessions without a
downloaded video/wav are skipped (the 3 vimeo + 2 dead ones).

Run on the server (GPU):
  source /etc/network_turbo         # first run downloads the text encoder
  export HF_HOME=/root/autodl-tmp/hf
  python scripts/build_features.py --limit 2   # smoke on 2 sessions first
"""
from __future__ import annotations
import os, json, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default="data/annomi/turns_chg.jsonl")
    ap.add_argument("--video_dir", default="data/annomi/video")
    ap.add_argument("--wav_dir", default="data/annomi/wav")
    ap.add_argument("--out_dir", default="data/annomi/feats")
    ap.add_argument("--text_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--n_frames", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N sessions (0=all)")
    args = ap.parse_args()

    import torch
    from mpse_mvp.segment.io import load_wav
    from mpse_mvp.mm.encoders import WhisperAudioEncoder, CLIPVideoEncoder, sample_video_frames
    from mpse_mvp.features.audio_features import audio_quality_and_prosody
    from mpse_mvp.features.text_features import text_quality
    from sentence_transformers import SentenceTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # group turns by session
    rows = [json.loads(l) for l in open(args.turns, encoding="utf-8")]
    by = {}
    for r in rows:
        by.setdefault(r["session_id"], []).append(r)

    # sessions that actually have media
    sids = [s for s in by
            if os.path.exists(os.path.join(args.wav_dir, f"{s}.wav"))
            and os.path.exists(os.path.join(args.video_dir, f"{s}.mp4"))]
    sids.sort(key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
    if args.limit:
        sids = sids[:args.limit]
    print(f"sessions with media: {len(sids)} (of {len(by)})")

    # load encoders once
    whisper_dir = "openai/whisper-small"
    clip_dir = "openai/clip-vit-base-patch32"
    aenc = WhisperAudioEncoder(whisper_dir, device=dev)
    venc = CLIPVideoEncoder(clip_dir, device=dev)
    tenc = SentenceTransformer(args.text_model, device=dev)

    index_path = os.path.join(args.out_dir, "index.jsonl")
    n_turns = 0
    with open(index_path, "w", encoding="utf-8") as idx:
        for si, sid in enumerate(sids, 1):
            wav, sr = load_wav(os.path.join(args.wav_dir, f"{sid}.wav"))
            mp4 = os.path.join(args.video_dir, f"{sid}.mp4")
            turns = sorted(by[sid], key=lambda r: r["turn_id"])
            print(f"[{si}/{len(sids)}] session {sid}: {len(turns)} turns")
            for r in turns:
                t0, t1 = float(r["t0"]), float(r["t1"])

                # audio
                s0, s1 = int(t0 * sr), int(t1 * sr)
                seg = wav[s0:s1].astype(np.float32)
                if len(seg) < sr // 10:  # <0.1s guard
                    seg = np.zeros(sr // 10, dtype=np.float32)
                q_audio, _ = audio_quality_and_prosody(seg, sr)
                a_pool, _ = aenc.encode(seg, sr=sr, return_sequence=False)
                audio_emb = a_pool.detach().cpu().numpy().reshape(-1).astype(np.float32)

                # video
                frames = sample_video_frames(mp4, t0, t1, n_frames=args.n_frames)
                q_video = float(len(frames)) / float(args.n_frames)
                v_pool, _ = venc.encode(frames, return_sequence=False)
                video_emb = v_pool.detach().cpu().numpy().reshape(-1).astype(np.float32)

                # text
                text_emb = tenc.encode([r["text"]], show_progress_bar=False)[0].astype(np.float32)
                q_text = float(text_quality(r["text"]))

                npz = os.path.join(args.out_dir, f"{sid}_{r['turn_id']:03d}.npz")
                np.savez_compressed(
                    npz, text_emb=text_emb, audio_emb=audio_emb, video_emb=video_emb,
                    q_text=q_text, q_audio=float(q_audio), q_video=q_video,
                    chg_weak=float(r.get("chg_weak", 0.0)),
                )
                idx.write(json.dumps({
                    "npz": npz, "session_id": sid, "turn_id": r["turn_id"],
                    "talk_type": r["talk_type"], "mi_quality": r["mi_quality"],
                }, ensure_ascii=False) + "\n")
                n_turns += 1

    print(f"\ndone: {n_turns} turns cached -> {args.out_dir}")
    print(f"index: {index_path}")
    print(f"dims: text={text_emb.shape[0]} audio={audio_emb.shape[0]} video={video_emb.shape[0]}")


if __name__ == "__main__":
    main()
