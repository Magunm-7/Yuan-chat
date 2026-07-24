"""
Discretize the evaluator's mu (chg/aro/val) into a natural-language client-state
tag that goes into the generation prompt (route: evaluator fuses audio+video ->
mu -> state tag -> prompt). Text is 'hard' so the model must read it -> the state
is controllable by construction (soft-token injection was ignored; see notes 0.4).

Thresholds are per-dim tertiles fit on the training turns (data-driven, balanced).
"""
from __future__ import annotations
import numpy as np

DIMS = ["chg", "aro", "val"]
LABELS = {
    "chg": ("change-readiness", ["low", "moderate", "high"]),
    "aro": ("arousal", ["low", "moderate", "high"]),
    "val": ("valence", ["negative", "neutral", "positive"]),
}


def fit_thresholds(mus):
    """mus: iterable of {chg,aro,val} dicts. Returns per-dim (t33, t66) tertiles."""
    thr = {}
    for d in DIMS:
        v = np.array([m[d] for m in mus], dtype=float)
        thr[d] = (float(np.percentile(v, 33.3)), float(np.percentile(v, 66.6)))
    return thr


def _level(v, t, labels):
    return labels[0] if v < t[0] else (labels[1] if v < t[1] else labels[2])


def state_tag(mu, thr):
    """mu: {chg,aro,val} floats. Returns e.g. '[Observed client state — change-readiness: high, arousal: low, valence: neutral]'."""
    parts = [f"{LABELS[d][0]}: {_level(mu[d], thr[d], LABELS[d][1])}" for d in DIMS]
    return "[Observed client state — " + ", ".join(parts) + "]"
