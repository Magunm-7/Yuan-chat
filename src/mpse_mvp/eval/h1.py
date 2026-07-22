"""
H1 — estimator validity: does MPSE's mu_chg recover the expert talk_type?

Main   : per-session Spearman(mu_chg, talk_ordinal), aggregated over sessions.
Aux    : AUC(-mu_chg -> change vs sustain), neutral dropped.
Null   : permute mu WITHIN session, recompute pooled AUC, 1000x -> p-value.
Judgment (docs/eval-design.md 3): rho significantly > 0 AND AUC > 0.60.

Consumes a predictions jsonl (one row per client turn):
  {session_id, turn_id, mi_quality, talk_type, mu:{chg:..}, sigma:{chg:..}}
Optionally uses splits.json to report per-CV-fold mean ± std.
"""
from __future__ import annotations
import os
import json
import argparse
import numpy as np

from mpse_mvp.eval.metrics import spearman, auc, talk_ordinal, make_synthetic_predictions


def load_predictions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _by_session(rows):
    d = {}
    for r in rows:
        d.setdefault(r["session_id"], []).append(r)
    return d


def session_spearmans(rows, min_turns=5):
    """Per-session Spearman(mu_chg, talk_ord). Returns list of (sid, rho, n)."""
    out = []
    for sid, rs in _by_session(rows).items():
        if len(rs) < min_turns:
            continue
        mu = np.array([r["mu"]["chg"] for r in rs])
        ordv = talk_ordinal([r["talk_type"] for r in rs])
        rho = spearman(mu, ordv)
        if not np.isnan(rho):
            out.append((sid, rho, len(rs)))
    return out


def _change_sustain(rows):
    mu, lab = [], []
    for r in rows:
        if r["talk_type"] == "change":
            lab.append(1); mu.append(r["mu"]["chg"])
        elif r["talk_type"] == "sustain":
            lab.append(0); mu.append(r["mu"]["chg"])
    return np.asarray(mu), np.asarray(lab)


def pooled_auc_change_sustain(rows):
    """AUC(mu_chg -> change vs sustain), pooled. HIGHER mu => change."""
    mu, lab = _change_sustain(rows)
    return auc(mu, lab) if len(mu) else float("nan")


def permutation_p(rows, n_perm=1000, seed=0):
    """Null: shuffle mu within each session, recompute pooled AUC. One-sided p (AUC>obs)."""
    obs = pooled_auc_change_sustain(rows)
    rng = np.random.RandomState(seed)
    by = _by_session(rows)
    ge = 1  # +1 smoothing
    for _ in range(n_perm):
        shuffled = []
        for rs in by.values():
            mus = [r["mu"]["chg"] for r in rs]
            rng.shuffle(mus)
            for r, m in zip(rs, mus):
                shuffled.append({**r, "mu": {"chg": m}})
        if pooled_auc_change_sustain(shuffled) >= obs:
            ge += 1
    return obs, ge / (n_perm + 1)


def _summ(vals):
    vals = [v for v in vals if not np.isnan(v)]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))


def evaluate(rows, splits=None):
    def fold_stats(subset):
        ss = session_spearmans(subset)
        rho_mean = float(np.mean([r for _, r, _ in ss])) if ss else float("nan")
        return rho_mean, pooled_auc_change_sustain(subset), len(ss)

    print("=== H1: estimator validity (mu_chg vs talk_type) ===")
    if splits:
        rhos, aucs = [], []
        for cf in splits["cv_folds"]:
            test = set(cf["test_sessions"])
            sub = [r for r in rows if r["session_id"] in test]
            rho, au, nsess = fold_stats(sub)
            rhos.append(rho); aucs.append(au)
            print(f"  fold {cf['fold']}: rho={rho:+.3f}  auc={au:.3f}  ({nsess} sessions)")
        rm, rs = _summ(rhos); am, as_ = _summ(aucs)
        print(f"  --- CV mean: rho={rm:+.3f}±{rs:.3f}   auc={am:.3f}±{as_:.3f}")
    else:
        rho, au, nsess = fold_stats(rows)
        print(f"  whole set: rho={rho:+.3f}  auc={au:.3f}  ({nsess} sessions)")

    obs_auc, p = permutation_p(rows)
    print(f"  permutation (within-session shuffle): auc={obs_auc:.3f}  p={p:.4f}")

    # classification view: change vs sustain at the BALANCED operating point
    # (max-F1 collapses to all-change because change is the majority here)
    from mpse_mvp.eval.metrics import balanced_best
    mu, lab = _change_sustain(rows)
    bacc, prec, rec, f1, thr = balanced_best(mu, lab)
    n_chg, n_sus = int(lab.sum()), int((lab == 0).sum())
    print(f"  change-vs-sustain ({n_chg} change / {n_sus} sustain): "
          f"balanced-acc={bacc:.3f}  P={prec:.3f} R={rec:.3f} F1={f1:.3f} (thr={thr:.3f})")

    verdict = "PASS" if (obs_auc > 0.60 and p < 0.05) else "FAIL"
    print(f"  judgment (auc>0.60 & p<0.05): {verdict}")
    return {"auc": obs_auc, "p": p, "balanced_acc": bacc,
            "precision": prec, "recall": rec, "f1": f1, "verdict": verdict}


def main():
    ap = argparse.ArgumentParser(description="H1 estimator-validity eval")
    ap.add_argument("--pred", default=None, help="predictions jsonl; omit for synthetic self-test")
    ap.add_argument("--splits", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.pred:
        rows = load_predictions(args.pred)
    else:
        print("[self-test] no --pred given, using synthetic predictions\n")
        rows = make_synthetic_predictions()
    splits = json.load(open(args.splits, encoding="utf-8")) if args.splits else None

    res = evaluate(rows, splits)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2)
        print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
