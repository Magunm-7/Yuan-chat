from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

# librosa is in your env (environment.yml pip deps)
import librosa


@dataclass
class SpeakerTurn:
    t0: float
    t1: float
    spk: int  # 0 or 1


def _seg_audio(wav: np.ndarray, sr: int, t0: float, t1: float) -> np.ndarray:
    s0 = max(0, int(t0 * sr))
    s1 = min(len(wav), int(t1 * sr))
    y = wav[s0:s1].astype(np.float32)
    if y.size == 0:
        return y
    # normalize (robust)
    m = np.max(np.abs(y)) + 1e-8
    return y / m


def _mfcc_embed(y: np.ndarray, sr: int, n_mfcc: int = 20) -> np.ndarray:
    """
    Very lightweight speaker-ish embedding:
    mean/std of MFCC (and delta MFCC) over time.
    Works decently for 2-speaker clustering in clean dialogue.
    """
    if y.size < int(0.2 * sr):  # too short
        return np.zeros((n_mfcc * 4,), dtype=np.float32)

    # use short hop for stability
    hop = int(0.01 * sr)
    win = int(0.025 * sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop, n_fft=win*2)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    feat = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        d1.mean(axis=1), d1.std(axis=1),
    ], axis=0).astype(np.float32)

    # normalize
    feat = feat - feat.mean()
    feat = feat / (feat.std() + 1e-6)
    return feat


def _kmeans2(X: np.ndarray, iters: int = 30, seed: int = 0) -> np.ndarray:
    """
    Minimal kmeans for K=2 (no sklearn dependency).
    Returns labels in {0,1}.
    """
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    if n == 1:
        return np.zeros((1,), dtype=np.int64)

    # init: pick 2 far-ish points
    i0 = rng.randint(0, n)
    c0 = X[i0]
    d = ((X - c0) ** 2).sum(axis=1)
    i1 = int(np.argmax(d))
    c1 = X[i1]

    for _ in range(iters):
        d0 = ((X - c0) ** 2).sum(axis=1)
        d1 = ((X - c1) ** 2).sum(axis=1)
        lab = (d1 < d0).astype(np.int64)

        if lab.sum() == 0 or lab.sum() == n:
            # degenerate; reinit second center
            i1 = rng.randint(0, n)
            c1 = X[i1]
            continue

        c0_new = X[lab == 0].mean(axis=0)
        c1_new = X[lab == 1].mean(axis=0)

        if np.allclose(c0, c0_new, atol=1e-4) and np.allclose(c1, c1_new, atol=1e-4):
            break
        c0, c1 = c0_new, c1_new

    return lab


def diarize_segments_k2(
    wav: np.ndarray,
    sr: int,
    segs: List[Tuple[float, float]],
    min_seg_sec: float = 0.35,
    gap_merge_sec: float = 0.8,
    seed: int = 0,
) -> List[SpeakerTurn]:
    """
    Input: VAD segments (t0,t1) that may be over-segmented.
    Output: speaker turns after clustering into 2 speakers and merging adjacent same-speaker segments.

    - min_seg_sec: ignore ultra-short segs for embedding (still kept, but embedding=0)
    - gap_merge_sec: merge adjacent turns if same speaker and gap <= this threshold
    """
    if not segs:
        return []

    # build embeddings per seg
    embs = []
    for (t0, t1) in segs:
        if (t1 - t0) < min_seg_sec:
            embs.append(np.zeros((80,), dtype=np.float32))  # 20*4
            continue
        y = _seg_audio(wav, sr, t0, t1)
        embs.append(_mfcc_embed(y, sr, n_mfcc=20))
    X = np.stack(embs, axis=0)

    labels = _kmeans2(X, iters=30, seed=seed)

    # merge consecutive segs with same label
    turns: List[SpeakerTurn] = []
    for (t0, t1), spk in zip(segs, labels):
        if not turns:
            turns.append(SpeakerTurn(float(t0), float(t1), int(spk)))
            continue
        prev = turns[-1]
        if prev.spk == int(spk) and (t0 - prev.t1) <= gap_merge_sec:
            prev.t1 = float(t1)
        else:
            turns.append(SpeakerTurn(float(t0), float(t1), int(spk)))

    return turns


def map_speakers_to_roles(
    turns: List[SpeakerTurn],
    start_role: str = "assistant",
) -> List[Tuple[float, float, str, int]]:
    """
    Map spk {0,1} to roles {assistant,user}.
    Assumption: dialogue starts with start_role (default assistant).
    We use the first turn's speaker as start_role speaker, the other speaker as opposite role.

    Returns list of (t0,t1,role,spk)
    """
    if not turns:
        return []
    first_spk = turns[0].spk
    if start_role == "assistant":
        spk2role = {first_spk: "assistant", 1 - first_spk: "user"}
    else:
        spk2role = {first_spk: "user", 1 - first_spk: "assistant"}

    out = []
    for t in turns:
        out.append((t.t0, t.t1, spk2role[t.spk], t.spk))
    return out
