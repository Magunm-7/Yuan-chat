"""
Settle alignment-vs-size: for several text representations, measure the SUPERVISED
linear-probe ceiling on chg (change vs sustain gold), session-CV. If bart hidden
states (task-aligned, 400M) beat mpnet (bigger general) and MiniLM, the bottleneck
is alignment, not size.
"""
from __future__ import annotations
import json, argparse
import numpy as np
from mpse_mvp.eval.metrics import auc

CANDIDATES = [
    ("MiniLM (22M, general)",      "sentence-transformers/all-MiniLM-L6-v2", "st"),
    ("mpnet  (110M, general)",     "sentence-transformers/all-mpnet-base-v2", "st"),
    ("bart-mnli enc (400M, aligned)", "facebook/bart-large-mnli", "bart"),
]


def encode_st(name, texts, dev):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device=dev).encode(
        texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)


def encode_bart(name, texts, dev, bs=32):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(name)
    enc = AutoModel.from_pretrained(name).to(dev).eval().get_encoder()
    out = []
    for i in range(0, len(texts), bs):
        t = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True,
                max_length=128).to(dev)
        with torch.no_grad():
            h = enc(**t).last_hidden_state
        m = t["attention_mask"].unsqueeze(-1)
        out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def ridge_cv(X, sids, gold, splits, lam=10.0, target=None):
    id2 = {}
    for i, s in enumerate(sids):
        id2.setdefault(s, []).append(i)
    # target to FIT: gold-derived (supervised upper bound) or an explicit weak label
    y = np.asarray(target) if target is not None else \
        np.array([{"change": 1.0, "neutral": 0.5, "sustain": 0.0}[g] for g in gold])
    oof = np.full(len(gold), np.nan)
    for cf in splits["cv_folds"]:
        tr = [i for s in cf["train_sessions"] if s in id2 for i in id2[s]]
        te = [i for s in cf["test_sessions"] if s in id2 for i in id2[s]]
        Xtr = X[tr]; mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr = (Xtr - mu) / sd; Xte = (X[te] - mu) / sd
        w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ y[tr])
        oof[te] = Xte @ w
    m = np.array([g in ("change", "sustain") for g in gold])
    lab = np.array([1 if g == "change" else 0 for g in gold])[m]
    return auc(oof[m], lab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default="data/annomi/turns_chg.jsonl")
    ap.add_argument("--splits", default="data/annomi/splits.json")
    ap.add_argument("--target", default="both", choices=["gold", "chg_weak", "both"])
    args = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rows = [json.loads(l) for l in open(args.turns, encoding="utf-8")]
    texts = [r["text"] for r in rows]
    sids = [r["session_id"] for r in rows]
    gold = [r["talk_type"] for r in rows]
    chg_weak = np.array([float(r.get("chg_weak", 0.0)) for r in rows])
    splits = json.load(open(args.splits, encoding="utf-8"))

    print(f"{len(texts)} turns | linear-probe AUC vs gold (change vs sustain)")
    print("  train-on-gold = supervised ceiling; train-on-chg_weak = what MPSE actually faces\n")
    print(f"{'encoder':32s} {'dim':>5s} {'gold':>7s} {'chg_weak':>9s}")
    print("-" * 58)
    for label, name, kind in CANDIDATES:
        X = encode_st(name, texts, dev) if kind == "st" else encode_bart(name, texts, dev)
        a_gold = ridge_cv(X, sids, gold, splits) if args.target in ("gold", "both") else float("nan")
        a_weak = ridge_cv(X, sids, gold, splits, target=chg_weak) if args.target in ("chg_weak", "both") else float("nan")
        print(f"{label:32s} {X.shape[1]:>5d} {a_gold:>7.3f} {a_weak:>9.3f}")
    print("\n(chg_weak vs gold directly = 0.667; ladder rung0 MiniLM-on-chg_weak = 0.577)")


if __name__ == "__main__":
    main()
