"""
Session-level, quality-stratified data splits for AnnoMI.

Rule (docs/eval-design.md 2.3): split by `session_id`, NEVER by turn; stratify by
`mi_quality` (110 high / 23 low is imbalanced). Produces:
  - cv_folds : 5-fold CV for H1/H2/H3, each fold stratified
  - holdout  : a single train/val/test carved FROM the folds, for the generator demo
               (test = fold 0, val = fold 1, train = folds 2..4), so both uses share
               one coherent partition and no session leaks across a fold boundary.

Consumes manifest.json from mpse_mvp.data.annomi; writes splits.json.
"""
from __future__ import annotations
import os
import json
import random
import argparse
from collections import defaultdict, Counter


def _natural_key(sid: str):
    return (0, int(sid)) if sid.isdigit() else (1, sid)


def stratified_kfold(sessions: list[dict], k: int, seed: int,
                     strat_key: str = "mi_quality") -> list[list[str]]:
    """Round-robin sessions within each stratum into k folds (balanced strata)."""
    groups: dict = defaultdict(list)
    for s in sessions:
        groups[s[strat_key]].append(s["session_id"])
    rng = random.Random(seed)
    folds: list[list[str]] = [[] for _ in range(k)]
    for g in sorted(groups):
        items = sorted(groups[g], key=_natural_key)
        rng.shuffle(items)
        for i, sid in enumerate(items):
            folds[i % k].append(sid)
    return folds


def build_splits(manifest_path: str, out_path: str,
                 k: int = 5, seed: int = 42, min_turns: int = 1) -> dict:
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # only sessions that actually carry data
    sessions = [m for m in manifest if m["n_client_turns_kept"] >= min_turns]
    by_id = {m["session_id"]: m for m in sessions}

    folds = stratified_kfold(sessions, k=k, seed=seed)

    cv_folds = []
    all_ids = {m["session_id"] for m in sessions}
    for f in range(k):
        test = sorted(folds[f], key=_natural_key)
        train = sorted(all_ids - set(test), key=_natural_key)
        cv_folds.append({"fold": f, "test_sessions": test, "train_sessions": train})

    holdout = {
        "test": sorted(folds[0], key=_natural_key),
        "val": sorted(folds[1], key=_natural_key),
        "train": sorted([s for fold in folds[2:] for s in fold], key=_natural_key),
    }

    splits = {
        "seed": seed, "n_folds": k, "stratify_by": "mi_quality",
        "min_turns": min_turns,
        "n_sessions": len(sessions),
        "cv_folds": cv_folds,
        "holdout": holdout,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)

    _report(splits, by_id)
    _validate(splits, all_ids)
    print(f"wrote: {out_path}")
    return splits


def _fold_stats(ids: list[str], by_id: dict) -> str:
    q = Counter(by_id[s]["mi_quality"] for s in ids)
    turns = sum(by_id[s]["n_client_turns_kept"] for s in ids)
    h2 = sum(1 for s in ids if by_id[s]["usable_for_h2"])
    return f"{len(ids):3d} sess (high {q['high']:3d}/low {q['low']:2d})  {turns:4d} turns  {h2:3d} usable-H2"


def _report(splits: dict, by_id: dict):
    print("=== splits (session-level, stratified by mi_quality) ===")
    print("CV folds (test partition of each):")
    for cf in splits["cv_folds"]:
        print(f"  fold {cf['fold']}: {_fold_stats(cf['test_sessions'], by_id)}")
    print("Generator holdout:")
    for name in ("train", "val", "test"):
        print(f"  {name:5s}: {_fold_stats(splits['holdout'][name], by_id)}")


def _validate(splits: dict, all_ids: set):
    # CV test folds must partition all sessions exactly once
    seen = Counter()
    for cf in splits["cv_folds"]:
        seen.update(cf["test_sessions"])
    dupes = [s for s, c in seen.items() if c > 1]
    assert not dupes, f"session in >1 CV test fold: {dupes}"
    assert set(seen) == all_ids, "CV folds do not cover all sessions"
    # each CV fold: train and test disjoint
    for cf in splits["cv_folds"]:
        assert not (set(cf["test_sessions"]) & set(cf["train_sessions"])), \
            f"fold {cf['fold']} train/test overlap"
    # holdout: three-way disjoint, covers all
    h = splits["holdout"]
    assert not (set(h["train"]) & set(h["val"])), "holdout train/val overlap"
    assert not (set(h["train"]) & set(h["test"])), "holdout train/test overlap"
    assert not (set(h["val"]) & set(h["test"])), "holdout val/test overlap"
    assert set(h["train"]) | set(h["val"]) | set(h["test"]) == all_ids, "holdout misses sessions"
    print("validation: OK (folds disjoint, cover all, no leakage)")


def main():
    ap = argparse.ArgumentParser(description="Build session-level stratified splits")
    ap.add_argument("--manifest", default="data/annomi/manifest.json")
    ap.add_argument("--out", default="data/annomi/splits.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build_splits(args.manifest, args.out, k=args.k, seed=args.seed)


if __name__ == "__main__":
    main()
