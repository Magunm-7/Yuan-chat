"""
Pure-numpy metrics for MPSE evaluation (no torch / scipy dependency, so it runs
anywhere). Used by eval/h1.py and eval/h3.py.

talk_type ordinal: change=-1, neutral=0, sustain=+1 (aligned with mu; lower = more
change-oriented). See docs/eval-design.md 3.
"""
from __future__ import annotations
import numpy as np

TALK_SCORE = {"change": -1, "neutral": 0, "sustain": +1}


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks, handling ties (like scipy.stats.rankdata)."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    """Spearman rank correlation. Returns nan if degenerate."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def auc(scores, labels) -> float:
    """
    AUROC via the Mann-Whitney U statistic. `labels` in {0,1}; higher `scores`
    should indicate label==1. Returns nan if only one class present.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _rankdata(scores)
    u = r[pos].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def bootstrap_ci(values, stat_fn, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of stat_fn over a 1-D sample (resample rows)."""
    rng = np.random.RandomState(seed)
    values = np.asarray(values)
    n = len(values)
    stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        s = stat_fn(values[idx])
        if not np.isnan(s):  # drop degenerate resamples (e.g. one class only)
            stats.append(s)
    if not stats:
        return float(stat_fn(values)), float("nan"), float("nan")
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return float(stat_fn(values)), lo, hi


def talk_ordinal(talk_types) -> np.ndarray:
    return np.array([TALK_SCORE[t] for t in talk_types], dtype=float)


# ------------------------------------------------------------------ self-test
def make_synthetic_predictions(seed=0, n_sessions=40, signal=0.8):
    """
    Fabricate per-turn predictions with a KNOWN structure so eval scripts can be
    verified: mu_chg is (talk ordinal * signal + noise), where noise scale grows
    with sigma. So mu should track talk_type, and low-sigma turns should track it
    BETTER (that's what H3's non-circular calibration must recover).
    """
    rng = np.random.RandomState(seed)
    rows = []
    for s in range(n_sessions):
        quality = "high" if s % 5 != 0 else "low"  # 80/20-ish
        n_turns = rng.randint(10, 40)
        for t in range(n_turns):
            # more change talk late in high-quality sessions (a rising arc)
            p_change = 0.15 + (0.4 * t / n_turns if quality == "high" else 0.1)
            u = rng.rand()
            tt = "change" if u < p_change else ("sustain" if u > 0.85 else "neutral")
            ord_ = TALK_SCORE[tt]
            sigma = float(rng.uniform(0.05, 0.5))
            noise = rng.randn() * sigma * 2.0
            mu = 0.3 + signal * ord_ * 0.15 + noise  # lower mu <-> change
            rows.append({
                "session_id": f"S{s}", "turn_id": t + 1, "mi_quality": quality,
                "talk_type": tt, "mu": {"chg": float(mu)}, "sigma": {"chg": sigma},
            })
    return rows


if __name__ == "__main__":
    rows = make_synthetic_predictions()
    mu = np.array([r["mu"]["chg"] for r in rows])
    ordv = talk_ordinal([r["talk_type"] for r in rows])
    print("self-test on synthetic data:")
    print(f"  spearman(mu, talk_ord) = {spearman(mu, ordv):+.3f}  (expect > 0)")
    mask = np.array([r["talk_type"] in ("change", "sustain") for r in rows])
    lab = np.array([1 if r["talk_type"] == "change" else 0 for r in rows])[mask]
    print(f"  auc(-mu -> change)     = {auc(-mu[mask], lab):.3f}  (expect > 0.6)")
    shuf = mu.copy(); np.random.RandomState(1).shuffle(shuf)
    print(f"  spearman(shuffled)     = {spearman(shuf, ordv):+.3f}  (expect ~0)")
