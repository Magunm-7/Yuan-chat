"""
Evaluator-LEVEL ablation: does the evaluator itself add information, on quality
discrimination? Three tiers, simplest form:

  no-eval    raw weak-label trajectory (chg_weak/aro_weak/val_weak) -> quality
             (NO evaluator: feed the three raw weak scores straight in)
  eval-text  text-only MPSE mu trajectory -> quality
             (evaluator KEPT, but audio/video removed)
  full       three-modality MPSE mu trajectory -> quality

Same shape-features + ridge probe for all three, so the ONLY difference is what
produces each turn's state trajectory:
  full > no-eval  => the evaluator adds information (temporal denoise + fusion)
  full > eval-text => that added info comes (partly) from audio/video

  python scripts/ablate_evaluator.py --seeds 0,1,2
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_mpse import load_sessions, run_cv
from test_multimodal_value import session_matrix, ridge_cv_auc

DIMS = ("chg", "aro", "val")


def optc(preds, splits):
    X, y, sids = session_matrix(preds, list(DIMS), min_turns=8)
    a, _, _ = ridge_cv_auc(X, y, sids, splits)
    return a


def weak_label_preds(labels_path):
    """no-evaluator baseline: use the raw weak labels themselves as the trajectory."""
    preds = []
    for l in open(labels_path, encoding="utf-8"):
        r = json.loads(l)
        preds.append({
            "session_id": r["session_id"], "turn_id": r["turn_id"],
            "mi_quality": r["mi_quality"], "talk_type": r.get("talk_type", "neutral"),
            "mu": {"chg": float(r["chg_weak"]), "aro": float(r["aro_weak"]), "val": float(r["val_weak"])},
        })
    return preds


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

    print(f"=== evaluator-level ablation ({len(seeds)} seeds, epochs={args.epochs}) ===", flush=True)

    ne = optc(weak_label_preds(args.labels), splits)
    print(f"{'no-eval':10s} OptC AUC = {ne:.3f}          (raw weak-label trajectory, NO evaluator)", flush=True)

    for name, mods in [("eval-text", ("text",)), ("full", ("text", "audio", "video"))]:
        sessions, feat_dims = load_sessions(args.index, args.labels, mods, DIMS)
        vals = []
        for sd in seeds:
            preds = run_cv(sessions, feat_dims, splits, mods, DIMS, epochs=args.epochs, seed=sd)
            vals.append(optc(preds, splits))
        vals = np.array(vals)
        print(f"{name:10s} OptC AUC = {vals.mean():.3f} +/- {vals.std():.3f}", flush=True)

    print("ABLATION2 DONE", flush=True)


if __name__ == "__main__":
    main()
