"""
chg weak label via zero-shot NLI, + immediate validation against gold talk_type.

The lexicon rater was too weak (AUC 0.58). This scores each client utterance for
change-orientation with a zero-shot entailment model, writes the weak label, and
reports AUC / Spearman vs the held-out gold `talk_type`. If AUC > ~0.65 the weak
label is a usable MPSE training target; gold itself is NEVER used for training.

Run on the server (needs HF via mirror/proxy + GPU):
  source /etc/network_turbo   # or export HF_ENDPOINT=https://hf-mirror.com
  python scripts/label_chg_zeroshot.py
"""
from __future__ import annotations
import os, json, argparse
import numpy as np

from mpse_mvp.eval.metrics import auc, spearman

CHANGE_HYP = "The speaker wants to change their behavior."
SUSTAIN_HYP = "The speaker wants to keep their behavior the same."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default="data/annomi/turns_client.jsonl")
    ap.add_argument("--out", default="data/annomi/turns_chg.jsonl")
    ap.add_argument("--model", default="facebook/bart-large-mnli")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    import torch
    from transformers import pipeline
    dev = 0 if torch.cuda.is_available() else -1

    rows = [json.loads(l) for l in open(args.turns, encoding="utf-8")]
    texts = [r["text"] for r in rows]

    clf = pipeline("zero-shot-classification", model=args.model, device=dev,
                   batch_size=args.batch_size)
    # single-label scoring: P(change hypothesis) with the two hypotheses as candidates
    labels = ["wanting to change", "wanting to stay the same"]
    print(f"scoring {len(texts)} utterances with {args.model} on device {dev} ...")
    out = clf(texts, candidate_labels=labels, multi_label=False)

    chg = []
    for o in out:
        d = dict(zip(o["labels"], o["scores"]))
        chg.append(float(d.get("wanting to change", 0.0)))
    chg = np.array(chg)

    for r, c in zip(rows, chg):
        r["chg_weak"] = float(c)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- validate against gold (eval only) ----
    tt = [r["talk_type"] for r in rows]
    ordv = np.array([{"change": -1, "neutral": 0, "sustain": 1}[t] for t in tt])
    mask = np.array([t in ("change", "sustain") for t in tt])
    lab = np.array([1 if t == "change" else 0 for t in tt])[mask]

    print("\n=== chg_weak vs gold talk_type ===")
    print(f"  range [{chg.min():.3f}, {chg.max():.3f}]  mean {chg.mean():.3f}")
    print(f"  spearman(chg_weak, talk_ordinal) = {spearman(chg, ordv):+.3f}  (expect NEGATIVE)")
    print(f"  AUC(chg_weak -> change vs sustain) = {auc(chg[mask], lab):.3f}  (lexicon was 0.58)")
    for q in ("change", "neutral", "sustain"):
        m = np.array([t == q for t in tt])
        print(f"    {q:8s} mean chg_weak = {chg[m].mean():.3f}")
    print(f"\nwrote: {args.out}")


if __name__ == "__main__":
    main()
