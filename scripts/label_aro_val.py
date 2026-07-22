"""
Multimodal weak labels aro (arousal, audio) + val (valence, face) per turn.

These are the dimensions where audio/video SHOULD carry signal that text does not
(unlike chg, which is text-semantic). No turn-level gold exists for them, so we
report their dynamic range + session-level association with mi_quality; the real
test (option C) is whether adding them helps quality discrimination downstream.

  aro : audio prosody (RMS energy + spectral centroid + pitch spread) -> [0,1]
  val : facial valence proxy from MediaPipe face mesh (smile geometry) -> [0,1]
        q_face = fraction of sampled frames with a detectable face (reliability)

Run on the server:
  python scripts/label_aro_val.py --limit 2   # smoke first
"""
from __future__ import annotations
import os, json, argparse
import numpy as np


def arousal(wav, sr):
    import librosa
    wav = wav.astype(np.float32)
    if len(wav) < sr // 10:
        return 0.0
    wav = wav / (np.max(np.abs(wav)) + 1e-8)
    rms = float(np.sqrt(np.mean(wav ** 2)))
    cen = float(np.mean(librosa.feature.spectral_centroid(y=wav, sr=sr)))
    try:
        f0 = librosa.yin(wav, fmin=80, fmax=400, sr=sr)
        pit = float(np.nanstd(f0)) if np.isfinite(f0).any() else 0.0
    except Exception:
        pit = 0.0
    # crude normalization into [0,1]; combined activation
    a = 0.4 * np.clip(rms / 0.15, 0, 1) + 0.4 * np.clip((cen - 1500) / 2500, 0, 1) \
        + 0.2 * np.clip(pit / 60.0, 0, 1)
    return float(np.clip(a, 0, 1))


def valence_from_frames(frames, face_mesh):
    """Smile proxy: mouth-corner elevation relative to mouth height. [0,1], 0.5=neutral."""
    import mediapipe as mp
    vals = []
    for fr in frames:
        res = face_mesh.process(fr)
        if not res.multi_face_landmarks:
            continue
        lm = res.multi_face_landmarks[0].landmark
        # mouth corners 61/291, top lip 13, bottom lip 14 (MediaPipe face mesh indices)
        lx, rx = lm[61], lm[291]
        top, bot = lm[13], lm[14]
        mouth_h = abs(bot.y - top.y) + 1e-6
        center_y = (top.y + bot.y) / 2
        corner_y = (lx.y + rx.y) / 2
        # corners ABOVE mouth center (smaller y) -> smile -> higher valence
        smile = (center_y - corner_y) / mouth_h
        vals.append(0.5 + np.clip(smile, -0.5, 0.5))
    if not vals:
        return 0.5, 0.0
    return float(np.clip(np.mean(vals), 0, 1)), float(len(vals) / len(frames))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default="data/annomi/turns_chg.jsonl")
    ap.add_argument("--video_dir", default="data/annomi/video")
    ap.add_argument("--wav_dir", default="data/annomi/wav")
    ap.add_argument("--out", default="data/annomi/turns_labeled.jsonl")
    ap.add_argument("--n_frames", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import mediapipe as mp
    from mpse_mvp.segment.io import load_wav
    from mpse_mvp.mm.encoders import sample_video_frames

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

    face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                           min_detection_confidence=0.5)
    out_rows = []
    for si, sid in enumerate(sids, 1):
        wav, sr = load_wav(os.path.join(args.wav_dir, f"{sid}.wav"))
        mp4 = os.path.join(args.video_dir, f"{sid}.mp4")
        turns = sorted(by[sid], key=lambda r: r["turn_id"])
        print(f"[{si}/{len(sids)}] session {sid}: {len(turns)} turns")
        for r in turns:
            t0, t1 = float(r["t0"]), float(r["t1"])
            seg = wav[int(t0 * sr):int(t1 * sr)]
            r["aro_weak"] = arousal(seg, sr)
            frames = sample_video_frames(mp4, t0, t1, n_frames=args.n_frames)
            r["val_weak"], r["q_face"] = valence_from_frames(frames, face)
            out_rows.append(r)
    face.close()

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # dynamic-range report + session-level association with mi_quality
    aro = np.array([r["aro_weak"] for r in out_rows])
    val = np.array([r["val_weak"] for r in out_rows])
    qf = np.array([r["q_face"] for r in out_rows])
    print(f"\n=== dynamic range ({len(out_rows)} turns) ===")
    print(f"  aro: mean {aro.mean():.3f} std {aro.std():.3f} range [{aro.min():.2f},{aro.max():.2f}]")
    print(f"  val: mean {val.mean():.3f} std {val.std():.3f} range [{val.min():.2f},{val.max():.2f}]")
    print(f"  q_face: mean {qf.mean():.3f} (frac frames with a face)")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
