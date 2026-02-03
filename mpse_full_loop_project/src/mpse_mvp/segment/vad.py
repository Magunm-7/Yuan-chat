import numpy as np

def frame_energy(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    if len(x) < frame_len:
        return np.array([float(np.mean(x**2))], dtype=np.float32)
    n = 1 + (len(x) - frame_len) // hop
    e = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = i * hop
        fr = x[s:s+frame_len]
        e[i] = float(np.mean(fr*fr))
    return e

def energy_vad_segments(wav: np.ndarray, sr: int, frame_ms: int = 30, thr: float = 0.02,
                        min_speech_ms: int = 250, min_silence_ms: int = 400):
    """Return list of (t0,t1) in seconds."""
    wav = wav.astype(np.float32)
    wav = wav / (np.max(np.abs(wav)) + 1e-8)

    frame_len = int(sr * frame_ms / 1000)
    hop = frame_len  # non-overlap for simplicity
    e = frame_energy(wav, frame_len, hop)
    mask = e > thr

    segs = []
    in_seg = False
    start = 0
    for i, m in enumerate(mask):
        if m and not in_seg:
            in_seg = True
            start = i
        if (not m) and in_seg:
            end = i
            in_seg = False
            segs.append((start, end))
    if in_seg:
        segs.append((start, len(mask)))

    # merge short silences
    merged = []
    min_speech_frames = max(1, int(min_speech_ms / frame_ms))
    min_silence_frames = max(1, int(min_silence_ms / frame_ms))

    for (s, e_) in segs:
        if e_ - s < min_speech_frames:
            continue
        if not merged:
            merged.append([s, e_])
            continue
        prev = merged[-1]
        if s - prev[1] <= min_silence_frames:
            prev[1] = e_
        else:
            merged.append([s, e_])

    out = []
    for s, e_ in merged:
        t0 = s * frame_ms / 1000.0
        t1 = e_ * frame_ms / 1000.0
        out.append((t0, t1))
    return out
