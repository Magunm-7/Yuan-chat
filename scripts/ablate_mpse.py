"""
Evaluator ablation: knock out each design choice and measure the drop, over
several seeds. Shows each MPSE design decision earns its place (that the
evaluator's parts are load-bearing, not decoration).

Configs (all predict chg/aro/val):
  full     text+audio+video, learned sigmoid alpha gate, heteroscedastic NLL
  -audio   drop the audio modality (input)
  -video   drop the video modality (input)
  -alpha   uniform (equal-weight) fusion instead of the learned gate
  -sigma   quality-weighted MSE instead of heteroscedastic NLL (removes sigma feedback)

Two metrics per config, mean +/- std over seeds:
  chg AUC (H1)  : mu_chg vs gold talk_type (change vs sustain)  [text-dominated]
  OptC AUC      : chg+aro+val trajectory -> mi_quality           [the multimodal payoff]

  python scripts/ablate_mpse.py --seeds 0,1,2
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_mpse import load_sessions, run_cv
from test_multimodal_value import session_matrix, ridge_cv_auc
from mpse_mvp.eval.metrics import auc

DIMS = ("chg", "aro", "val")
CONFIGS = [
    ("full",   ("text", "audio", "video"), True,  "nll"),
    ("-audio", ("text", "video"),          True,  "nll"),
    ("-video", ("text", "audio"),          True,  "nll"),
    ("-alpha", ("text", "audio", "video"), False, "nll"),
    ("-sigma", ("text", "audio", "video"), True,  "mse"),
]


def chg_auc(preds):
    mu = np.array([p["mu"]["chg"] for p in preds])
    tt = [p["talk_type"] for p in preds]
    m = np.array([t in ("change", "sustain") for t in tt])
    lab = np.array([1 if t == "change" else 0 for t in tt])[m]
    return auc(mu[m], lab)


def optc_auc(preds, splits):
    X, y, sids = session_matrix(preds, list(DIMS), min_turns=8)
    a, _, _ = ridge_cv_auc(X, y, sids, splits)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/annomi/feats/index.jsonl")
    ap.add_argument("--labels", default="data/annomi/turns_labeled.jsonl")
    ap.add_argument("--splits", default="data/annomi/splits.json")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    splits = json.load(open(args.splits, encoding="utf-8"))

    print(f"=== MPSE ablation ({len(seeds)} seeds: {seeds}, epochs={args.epochs}) ===")
    print(f"{'config':8s} {'chg AUC (H1)':>16s} {'OptC AUC (quality)':>22s}")
    rows = []
    for name, mods, use_alpha, loss in CONFIGS:
        sessions, feat_dims = load_sessions(args.index, args.labels, mods, DIMS)
        chg, optc = [], []
        for sd in seeds:
            preds = run_cv(sessions, feat_dims, splits, mods, DIMS,
                           epochs=args.epochs, use_alpha=use_alpha, loss=loss, seed=sd)
            chg.append(chg_auc(preds))
            optc.append(optc_auc(preds, splits))
        chg, optc = np.array(chg), np.array(optc)
        rows.append((name, chg.mean(), chg.std(), optc.mean(), optc.std()))
        print(f"{name:8s} {chg.mean():.3f} +/- {chg.std():.3f}    "
              f"{optc.mean():.3f} +/- {optc.std():.3f}", flush=True)

    full = rows[0]
    print("\n--- drop vs full ---")
    print(f"{'config':8s} {'ΔchgAUC':>10s} {'ΔOptcAUC':>12s}")
    for name, cm, _, om, _ in rows[1:]:
        print(f"{name:8s} {cm - full[1]:>+10.3f} {om - full[3]:>+12.3f}")


if __name__ == "__main__":
    main()
