"""
Paired bootstrap on the multimodal-vs-text-only holdout perplexity gap.

Both dumps are per-turn {nll,ntok} in the SAME dataset order (eval_mm_ppl --dump),
so turn i is paired across configs. We resample TURNS with replacement and recompute
corpus ppl for each config on the resample -> CI on (ppl_mm - ppl_text). Also reports
the per-turn win rate and the one-sided bootstrap fraction where multimodal is better.

  python scripts/paired_ppl_boot.py --text dump_text.jsonl --mm dump_mm.jsonl
"""
from __future__ import annotations
import argparse, json
import numpy as np


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    nll = np.array([r["nll"] for r in rows], dtype=np.float64)
    ntok = np.array([r["ntok"] for r in rows], dtype=np.float64)
    return nll, ntok


def corpus_ppl(nll, ntok, idx):
    return np.exp(nll[idx].sum() / max(1.0, ntok[idx].sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--mm", required=True)
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    nll_t, tok_t = load(args.text)
    nll_m, tok_m = load(args.mm)
    assert len(nll_t) == len(nll_m) and np.allclose(tok_t, tok_m), \
        "dumps misaligned (different turns/token counts)"
    n = len(nll_t)
    all_idx = np.arange(n)

    ppl_t = corpus_ppl(nll_t, tok_t, all_idx)
    ppl_m = corpus_ppl(nll_m, tok_m, all_idx)

    # per-turn per-token nll (fair per-turn comparison independent of length)
    ptok_t = nll_t / np.maximum(1.0, tok_t)
    ptok_m = nll_m / np.maximum(1.0, tok_m)
    win = float((ptok_m < ptok_t).mean())            # turns where mm is better
    mean_diff = float((ptok_m - ptok_t).mean())      # mean per-token nll diff (neg = mm better)

    rng = np.random.RandomState(args.seed)
    diffs = np.empty(args.n_boot)
    for b in range(args.n_boot):
        idx = rng.randint(0, n, size=n)
        diffs[b] = corpus_ppl(nll_m, tok_m, idx) - corpus_ppl(nll_t, tok_t, idx)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    frac_mm_better = float((diffs < 0).mean())       # one-sided: mm ppl < text ppl

    print("=== paired bootstrap: multimodal vs text-only (holdout ppl) ===")
    print(f"  n turns = {n}")
    print(f"  ppl  text-only = {ppl_t:.3f}   multimodal = {ppl_m:.3f}   (Δ = {ppl_m-ppl_t:+.3f})")
    print(f"  Δppl 95% CI = [{lo:+.3f}, {hi:+.3f}]   (neg = multimodal better)")
    print(f"  bootstrap P(multimodal better) = {frac_mm_better:.3f}")
    print(f"  per-turn: multimodal better on {win*100:.1f}% of turns; "
          f"mean per-token NLL diff = {mean_diff:+.4f} nats")


if __name__ == "__main__":
    main()
