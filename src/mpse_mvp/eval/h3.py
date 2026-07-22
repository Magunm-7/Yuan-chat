"""
H3 — the uncertainty story (two roles).

(a) NON-CIRCULAR calibration (the key figure): bin turns by sigma_chg into equal-
    frequency bins; within EACH bin compute H1's AUC against the EXTERNAL label
    (change vs sustain). If sigma carries "trustworthiness", AUC should DECREASE as
    sigma rises. Reported as per-bin AUC + Spearman(bin_index, AUC) (expect < 0).
    This avoids the circularity of binning by sigma and scoring against y_soft
    (the training target) — see notes.md 6.5 A1.

(b) sigma as a QUALITY signal: low-quality (erratic) sessions should have higher
    session-mean sigma. Reported as AUC(session-mean sigma -> mi_quality == low).

Consumes the same predictions jsonl as h1.py.
"""
from __future__ import annotations
import os
import json
import argparse
import numpy as np

from mpse_mvp.eval.metrics import auc, spearman, bootstrap_ci, make_synthetic_predictions
from mpse_mvp.eval.h1 import load_predictions, pooled_auc_change_sustain


def sigma_bins_auc(rows, n_bins=5):
    """Equal-frequency sigma bins; per-bin AUC(-mu -> change vs sustain)."""
    sig = np.array([r["sigma"]["chg"] for r in rows])
    order = np.argsort(sig)
    bins = np.array_split(order, n_bins)
    out = []
    for b, idx in enumerate(bins):
        sub = [rows[i] for i in idx]
        a = pooled_auc_change_sustain(sub)
        smean = float(np.mean([rows[i]["sigma"]["chg"] for i in idx]))
        out.append({"bin": b, "sigma_mean": smean, "auc": a, "n": len(idx)})
    valid = [(o["bin"], o["auc"]) for o in out if not np.isnan(o["auc"])]
    mono = spearman([b for b, _ in valid], [a for _, a in valid]) if len(valid) >= 3 else float("nan")
    return out, mono


def sigma_as_quality(rows):
    """AUC(session-mean sigma -> mi_quality == 'low')."""
    by = {}
    for r in rows:
        by.setdefault(r["session_id"], []).append(r)
    smean, label = [], []
    for sid, rs in by.items():
        smean.append(float(np.mean([r["sigma"]["chg"] for r in rs])))
        label.append(1 if rs[0]["mi_quality"] == "low" else 0)
    smean = np.array(smean); label = np.array(label)
    point, lo, hi = bootstrap_ci(
        np.arange(len(smean)),
        lambda idx: auc(smean[idx], label[idx]),
    )
    return point, lo, hi, int(label.sum()), len(label)


def evaluate(rows, n_bins=5):
    print("=== H3(a): non-circular sigma calibration ===")
    bins, mono = sigma_bins_auc(rows, n_bins)
    for o in bins:
        au = "  nan" if np.isnan(o["auc"]) else f"{o['auc']:.3f}"
        print(f"  bin {o['bin']} (sigma~{o['sigma_mean']:.3f}, n={o['n']:4d}): auc={au}")
    print(f"  monotonicity Spearman(bin, auc) = {mono:+.3f}  (expect < 0: high sigma -> low auc)")

    print("=== H3(b): sigma as a quality signal ===")
    au, lo, hi, n_low, n_tot = sigma_as_quality(rows)
    print(f"  auc(session-mean sigma -> low quality) = {au:.3f}  [95% CI {lo:.3f}, {hi:.3f}]"
          f"   ({n_low} low / {n_tot} sessions)")

    return {"calibration_bins": bins, "calibration_monotonicity": mono,
            "sigma_quality_auc": au, "sigma_quality_ci": [lo, hi]}


def main():
    ap = argparse.ArgumentParser(description="H3 uncertainty eval")
    ap.add_argument("--pred", default=None)
    ap.add_argument("--n_bins", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.pred:
        rows = load_predictions(args.pred)
    else:
        print("[self-test] no --pred given, using synthetic predictions\n")
        rows = make_synthetic_predictions()

    res = evaluate(rows, args.n_bins)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2)
        print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
