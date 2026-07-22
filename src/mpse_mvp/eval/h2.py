"""
H2 — trajectory shape discriminates MI quality.

premise (notes.md 6.8 F3): net improvement does NOT separate high/low (both rise to
~80% change-orientation); the SHAPE does — high sessions climb monotonically, low
sessions dip mid-session. So we read shape features off each session's mu_chg
trajectory and test whether they discriminate `mi_quality`.

Per-session shape features on the mu_chg sequence (ordered by turn):
  monotonicity : Spearman(position, mu_chg)      (steady drift vs noise)
  net_drift    : mean(last third) - mean(first third)
  mid_dip      : mean(mid third) - (first+last)/2   (F3: low sessions dip in the middle)
  roughness    : variance of 2nd differences        (erratic vs smooth)

Each feature -> AUC(feature -> mi_quality == 'low'), with bootstrap CI over sessions.
Only sessions with >= min_turns are used (need enough turns to see an arc).
"""
from __future__ import annotations
import os
import json
import argparse
import numpy as np

from mpse_mvp.eval.metrics import auc, spearman, bootstrap_ci, make_synthetic_predictions
from mpse_mvp.eval.h1 import load_predictions


def session_trajectories(rows, min_turns=8):
    by = {}
    for r in rows:
        by.setdefault(r["session_id"], []).append(r)
    out = {}
    for sid, rs in by.items():
        if len(rs) < min_turns:
            continue
        rs = sorted(rs, key=lambda r: r["turn_id"])
        mu = np.array([r["mu"]["chg"] for r in rs], dtype=float)
        out[sid] = (mu, rs[0]["mi_quality"])
    return out


def shape_features(mu: np.ndarray) -> dict:
    n = len(mu)
    a, b = n // 3, 2 * n // 3
    early, mid, late = mu[:a].mean(), mu[a:b].mean(), mu[b:].mean()
    d2 = np.diff(mu, n=2) if n >= 3 else np.array([0.0])
    return {
        "monotonicity": spearman(np.arange(n), mu),
        "net_drift": float(late - early),
        "mid_dip": float(mid - (early + late) / 2.0),
        "roughness": float(np.var(d2)),
    }


FEATURES = ["monotonicity", "net_drift", "mid_dip", "roughness"]


def evaluate(rows, min_turns=8):
    traj = session_trajectories(rows, min_turns)
    feats = {sid: shape_features(mu) for sid, (mu, _) in traj.items()}
    label = np.array([1 if traj[sid][1] == "low" else 0 for sid in traj])
    n_low = int(label.sum())

    print(f"=== H2: trajectory shape vs MI quality ===")
    print(f"  sessions used (>= {min_turns} turns): {len(traj)}  (low {n_low} / high {len(traj) - n_low})")
    print(f"  {'feature':13s} {'AUC(->low)':>11s}   95% CI")
    results = {}
    sids = list(traj)
    for fname in FEATURES:
        vals = np.array([feats[s][fname] for s in sids], dtype=float)
        good = ~np.isnan(vals)
        v, lab = vals[good], label[good]
        idx = np.arange(len(v))
        point, lo, hi = bootstrap_ci(idx, lambda ii: auc(v[ii], lab[ii]))
        results[fname] = {"auc": point, "ci": [lo, hi]}
        arrow = "" if np.isnan(point) else ("  (low>high)" if point >= 0.5 else "  (low<high)")
        print(f"  {fname:13s} {point:11.3f}   [{lo:.3f}, {hi:.3f}]{arrow}")
    return results


def main():
    ap = argparse.ArgumentParser(description="H2 trajectory-shape eval")
    ap.add_argument("--pred", default=None)
    ap.add_argument("--min_turns", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.pred:
        rows = load_predictions(args.pred)
    else:
        print("[self-test] no --pred given, using synthetic predictions\n")
        rows = make_synthetic_predictions()

    res = evaluate(rows, args.min_turns)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2)
        print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
