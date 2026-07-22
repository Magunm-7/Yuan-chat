"""
Option C: does the MULTIMODAL state trajectory (chg+aro+val) discriminate MI
quality better than chg alone? This is where multimodal proves its worth, using
the gold `mi_quality` label (chg has no multimodal advantage; aro/val have no
turn-gold — but their trajectories may help predict session quality).

Per session, each state dim's mu-trajectory -> shape features; a ridge probe maps
them to mi_quality under session CV. Compare feature set {chg} vs {chg,aro,val}.
"""
from __future__ import annotations
import json, argparse
import numpy as np
from mpse_mvp.eval.metrics import auc, spearman, bootstrap_ci


def shape_features(mu: np.ndarray) -> list[float]:
    n = len(mu)
    a, b = max(1, n // 3), max(2, 2 * n // 3)
    early, mid, late = mu[:a].mean(), mu[a:b].mean(), mu[b:].mean()
    d2 = np.diff(mu, n=2) if n >= 3 else np.array([0.0])
    mono = spearman(np.arange(n), mu)
    return [float(mu.mean()), float(mu.std()),
            0.0 if np.isnan(mono) else float(mono),
            float(late - early), float(mid - (early + late) / 2), float(np.var(d2))]


def session_matrix(preds, dims, min_turns):
    by = {}
    for p in preds:
        by.setdefault(p["session_id"], []).append(p)
    rows, y, sids = [], [], []
    for sid, ps in by.items():
        if len(ps) < min_turns:
            continue
        ps.sort(key=lambda p: p["turn_id"])
        feat = []
        for d in dims:
            feat += shape_features(np.array([p["mu"][d] for p in ps]))
        rows.append(feat)
        y.append(1 if ps[0]["mi_quality"] == "low" else 0)
        sids.append(sid)
    return np.array(rows), np.array(y), sids


def ridge_cv_auc(X, y, sids, splits, lam=1.0):
    """Session-CV out-of-fold ridge probe -> AUC(pred, low)."""
    id2i = {s: i for i, s in enumerate(sids)}
    oof = np.full(len(y), np.nan)
    for cf in splits["cv_folds"]:
        tr = [id2i[s] for s in cf["train_sessions"] if s in id2i]
        te = [id2i[s] for s in cf["test_sessions"] if s in id2i]
        if not te or not tr:
            continue
        Xtr, Xte = X[tr], X[te]
        mu_, sd_ = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr, Xte = (Xtr - mu_) / sd_, (Xte - mu_) / sd_
        w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ y[tr])
        oof[te] = Xte @ w
    ok = ~np.isnan(oof)
    return auc(oof[ok], y[ok]), oof, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="data/annomi/pred_mm.jsonl")
    ap.add_argument("--splits", default="data/annomi/splits.json")
    ap.add_argument("--min_turns", type=int, default=8)
    args = ap.parse_args()

    preds = [json.loads(l) for l in open(args.pred, encoding="utf-8")]
    splits = json.load(open(args.splits, encoding="utf-8"))

    print("=== Option C: multimodal state trajectory -> MI quality ===")
    results = {}
    for name, dims in [("chg-only", ["chg"]), ("chg+aro+val", ["chg", "aro", "val"])]:
        X, y, sids = session_matrix(preds, dims, args.min_turns)
        a, oof, ok = ridge_cv_auc(X, y, sids, splits)
        # bootstrap CI over sessions
        yy, pp = y[ok], oof[ok]
        point, lo, hi = bootstrap_ci(np.arange(len(yy)), lambda ii: auc(pp[ii], yy[ii]))
        results[name] = (a, lo, hi)
        print(f"  {name:12s}: AUC={a:.3f}  [95% CI {lo:.3f}, {hi:.3f}]  "
              f"({int(y.sum())} low / {len(y)} sessions, {X.shape[1]} feats)")

    d = results["chg+aro+val"][0] - results["chg-only"][0]
    print(f"\n  multimodal - chg-only = {d:+.3f}  "
          f"({'multimodal helps' if d > 0.02 else 'no clear gain'})")


if __name__ == "__main__":
    main()
