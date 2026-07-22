"""
Pure-numpy metrics for MPSE evaluation (no torch / scipy dependency, so it runs
anywhere). Used by eval/h1.py and eval/h3.py.

Convention (unified 2026-07-22): HIGHER mu = MORE change-oriented, matching the
chg weak label. talk ordinal: change=+1, neutral=0, sustain=-1.
"""
from __future__ import annotations
import numpy as np

TALK_SCORE = {"change": +1, "neutral": 0, "sustain": -1}


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


def prf_best(scores, labels):
    """
    Precision / recall / F1 at the threshold that maximizes F1 (higher score =>
    label 1). Returns (precision, recall, f1, threshold). In-sample operating
    point (report as such). nan if only one class.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    P = int(labels.sum())
    if P == 0 or P == len(labels):
        return float("nan"), float("nan"), float("nan"), float("nan")
    order = np.argsort(-scores)
    y = labels[order]
    s = scores[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / (tp + fp)
    rec = tp / P
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    i = int(np.argmax(f1))
    return float(prec[i]), float(rec[i]), float(f1[i]), float(s[i])


def balanced_best(scores, labels):
    """
    Operating point that maximizes BALANCED accuracy (mean of TPR and TNR) — the
    right choice for imbalanced classes where max-F1 collapses to all-positive.
    Returns (balanced_acc, precision, recall, f1, threshold).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    P = int(labels.sum()); N = len(labels) - P
    if P == 0 or N == 0:
        return (float("nan"),) * 5
    order = np.argsort(-scores)
    y = labels[order]; s = scores[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = tp / P; tnr = (N - fp) / N
    bacc = (tpr + tnr) / 2
    i = int(np.argmax(bacc))
    prec = tp[i] / (tp[i] + fp[i] + 1e-12)
    rec = tpr[i]
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(bacc[i]), float(prec), float(rec), float(f1), float(s[i])


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
            ord_ = TALK_SCORE[tt]  # change=+1
            sigma = float(rng.uniform(0.05, 0.5))
            noise = rng.randn() * sigma * 2.0
            mu = 0.5 + signal * ord_ * 0.15 + noise  # HIGHER mu <-> change
            rows.append({
                "session_id": f"S{s}", "turn_id": t + 1, "mi_quality": quality,
                "talk_type": tt, "mu": {"chg": float(mu)}, "sigma": {"chg": sigma},
            })
    return rows


if __name__ == "__main__":
    rows = make_synthetic_predictions()
    mu = np.array([r["mu"]["chg"] for r in rows])
    ordv = talk_ordinal([r["talk_type"] for r in rows])
    print("self-test on synthetic data (HIGH mu = change):")
    print(f"  spearman(mu, talk_ord) = {spearman(mu, ordv):+.3f}  (expect > 0)")
    mask = np.array([r["talk_type"] in ("change", "sustain") for r in rows])
    lab = np.array([1 if r["talk_type"] == "change" else 0 for r in rows])[mask]
    print(f"  auc(mu -> change)      = {auc(mu[mask], lab):.3f}  (expect > 0.6)")
    p, r, f1, thr = prf_best(mu[mask], lab)
    print(f"  P/R/F1(change) @bestF1 = {p:.2f}/{r:.2f}/{f1:.2f} (thr={thr:.2f})")
    shuf = mu.copy(); np.random.RandomState(1).shuffle(shuf)
    print(f"  spearman(shuffled)     = {spearman(shuf, ordv):+.3f}  (expect ~0)")
