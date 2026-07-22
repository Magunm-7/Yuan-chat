"""
Test-driven diagnosis: build a ladder from a plain linear probe up to the full
MPSE, one component at a time (text-only, MiniLM features), and see WHERE the
AUC-vs-gold drops. Localizes the design flaw instead of guessing.

All rungs: train on chg_weak (the weak label MPSE actually uses), session-CV,
evaluate mu vs gold (change vs sustain). The weak label itself is ~0.667 vs gold;
a good model should approach that, not fall to 0.59.
"""
from __future__ import annotations
import json
import numpy as np


def load(index_path):
    idx = [json.loads(l) for l in open(index_path, encoding="utf-8")]
    by = {}
    for r in idx:
        d = np.load(r["npz"])
        by.setdefault(r["session_id"], []).append(
            (r["turn_id"], d["text_emb"].astype(np.float32), float(d["chg_weak"]), r["talk_type"]))
    for s in by:
        by[s].sort(key=lambda x: x[0])
    return by


def auc(score, lab):
    from mpse_mvp.eval.metrics import auc as _a
    return _a(score, lab)


def eval_vs_gold(pred, gold):
    m = np.array([g in ("change", "sustain") for g in gold])
    lab = np.array([1 if g == "change" else 0 for g in gold])[m]
    return auc(np.asarray(pred)[m], lab)


def ridge_rung(by, splits, lam=10.0):
    """Rung 0: closed-form linear regression X->chg_weak."""
    pred, gold = [], []
    id2 = {s: by[s] for s in by}
    for cf in splits["cv_folds"]:
        tr = [s for s in cf["train_sessions"] if s in id2]
        te = [s for s in cf["test_sessions"] if s in id2]
        Xtr = np.vstack([np.array([t[1] for t in id2[s]]) for s in tr])
        ytr = np.concatenate([[t[2] for t in id2[s]] for s in tr])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr = (Xtr - mu) / sd
        w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
        for s in te:
            X = (np.array([t[1] for t in id2[s]]) - mu) / sd
            pred += list(X @ w); gold += [t[3] for t in id2[s]]
    return eval_vs_gold(pred, gold)


def torch_rung(by, splits, use_mlp, use_sigmoid, loss, use_gru, epochs=80, lr=1e-3):
    import torch, torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dim = next(iter(by.values()))[0][1].shape[0]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            h = 256
            self.proj = nn.Sequential(nn.Linear(dim, h), nn.ReLU()) if use_mlp else nn.Linear(dim, h if use_gru else 1)
            self.gru = nn.GRU(h, h, batch_first=True) if use_gru else None
            self.mu = nn.Linear(h, 1) if (use_mlp or use_gru) else None
            self.lv = nn.Linear(h, 1) if (loss == "nll" and (use_mlp or use_gru)) else None

        def forward(self, x):  # x (1,T,dim)
            z = self.proj(x)
            if self.gru is not None:
                z = self.gru(z)[0]
            raw = self.mu(z) if self.mu is not None else z
            mu = torch.sigmoid(raw) if use_sigmoid else raw
            lv = self.lv(z) if self.lv is not None else None
            return mu, lv

    pred, gold = [], []
    for cf in splits["cv_folds"]:
        tr = [s for s in cf["train_sessions"] if s in by]
        te = [s for s in cf["test_sessions"] if s in by]
        net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=lr)
        net.train()
        for _ in range(epochs):
            for s in np.random.permutation(tr):
                X = torch.tensor(np.array([t[1] for t in by[s]])[None]).to(dev)
                y = torch.tensor(np.array([[t[2]] for t in by[s]])[None], dtype=torch.float32).to(dev)
                mu, lv = net(X)
                if loss == "nll" and lv is not None:
                    lv = torch.clamp(lv, -6, 2)
                    l = (0.5 * ((y - mu) ** 2 / torch.exp(lv) + lv)).mean()
                else:
                    l = ((y - mu) ** 2).mean()
                opt.zero_grad(); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            for s in te:
                X = torch.tensor(np.array([t[1] for t in by[s]])[None]).to(dev)
                mu, _ = net(X)
                pred += list(mu.squeeze().cpu().numpy().reshape(-1)); gold += [t[3] for t in by[s]]
    return eval_vs_gold(pred, gold)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/annomi/feats/index.jsonl")
    ap.add_argument("--splits", default="data/annomi/splits.json")
    args = ap.parse_args()
    by = load(args.index)
    splits = json.load(open(args.splits, encoding="utf-8"))
    np.random.seed(0)
    print("text-only (MiniLM) ladder, train on chg_weak, AUC vs gold  (weak label=0.667)\n")
    print(f"  {'rung':44s} {'AUC':>6s}")
    print("  " + "-" * 52)
    print(f"  {'0. ridge  linear, closed-form':44s} {ridge_rung(by, splits):>6.3f}")
    print(f"  {'1. linear + MSE (SGD)':44s} {torch_rung(by, splits, 0,0,'mse',0):>6.3f}")
    print(f"  {'2. MLP    + MSE, linear mu':44s} {torch_rung(by, splits, 1,0,'mse',0):>6.3f}")
    print(f"  {'3. MLP    + MSE, sigmoid mu':44s} {torch_rung(by, splits, 1,1,'mse',0):>6.3f}")
    print(f"  {'4. MLP    + hetero-NLL, sigmoid mu':44s} {torch_rung(by, splits, 1,1,'nll',0):>6.3f}")
    print(f"  {'5. MLP+GRU+ hetero-NLL (full MPSE)':44s} {torch_rung(by, splits, 1,1,'nll',1):>6.3f}")


if __name__ == "__main__":
    main()
