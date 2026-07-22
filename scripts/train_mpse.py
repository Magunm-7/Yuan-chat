"""
Train the multimodal temporal MPSE (multi-dimensional state) with session CV,
emit leakage-free out-of-fold predictions of mu/sigma per state dimension.

Inputs joined by (session_id, turn_id):
  - embeddings   from feats/index.jsonl (.npz: text_emb/audio_emb/video_emb + q_*)
  - weak targets from turns_labeled.jsonl (chg_weak, aro_weak, val_weak)

chg is text-semantic (gold-validated); aro/val are the audio/video dims whose
downstream value is tested in test_multimodal_value.py (option C).

  python scripts/train_mpse.py --dims chg,aro,val --modalities text,audio,video \
         --out data/annomi/pred_mm.jsonl
"""
from __future__ import annotations
import os, json, argparse
import numpy as np


def load_sessions(index_path, labels_path, modalities, dims):
    labels = {}
    for l in open(labels_path, encoding="utf-8"):
        r = json.loads(l)
        labels[(r["session_id"], r["turn_id"])] = r
    idx = [json.loads(l) for l in open(index_path, encoding="utf-8")]
    by = {}
    for r in idx:
        key = (r["session_id"], r["turn_id"])
        if key in labels:
            by.setdefault(r["session_id"], []).append((r, labels[key]))
    sessions, feat_dims = {}, None
    for sid, items in by.items():
        items.sort(key=lambda x: x[0]["turn_id"])
        arrs = {m: [] for m in modalities}
        Y, q, meta = [], [], []
        for r, lab in items:
            d = np.load(r["npz"])
            for m in modalities:
                arrs[m].append(d[f"{m}_emb"])
            Y.append([float(lab[f"{dim}_weak"]) for dim in dims])
            q.append(float((d["q_text"] + d["q_audio"] + d["q_video"]) / 3.0))
            meta.append((r["turn_id"], r["talk_type"], r["mi_quality"]))
        feats = {m: np.stack(arrs[m]).astype(np.float32) for m in modalities}
        if feat_dims is None:
            feat_dims = {m: feats[m].shape[1] for m in modalities}
        sessions[sid] = (feats, np.array(Y, np.float32), np.array(q, np.float32), meta)
    return sessions, feat_dims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/annomi/feats/index.jsonl")
    ap.add_argument("--labels", default="data/annomi/turns_labeled.jsonl")
    ap.add_argument("--splits", default="data/annomi/splits.json")
    ap.add_argument("--out", default="data/annomi/pred_mm.jsonl")
    ap.add_argument("--dims", default="chg,aro,val")
    ap.add_argument("--modalities", default="text,audio,video")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_gru", action="store_true")
    ap.add_argument("--loss", default="nll", choices=["nll", "mse"])
    args = ap.parse_args()

    import torch
    from mpse_mvp.mpse.model_mm import MPSE_MM, hetero_nll

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mods = tuple(args.modalities.split(","))
    dims = tuple(args.dims.split(","))

    sessions, feat_dims = load_sessions(args.index, args.labels, mods, dims)
    splits = json.load(open(args.splits, encoding="utf-8"))
    print(f"dims={dims} modalities={mods} feat_dims={feat_dims} sessions={len(sessions)}")

    def to_dev(feats):
        return {m: torch.from_numpy(feats[m]).unsqueeze(0).to(dev) for m in mods}

    preds = []
    for cf in splits["cv_folds"]:
        test = [s for s in cf["test_sessions"] if s in sessions]
        train = [s for s in cf["train_sessions"] if s in sessions]
        model = MPSE_MM(feat_dims, mods, hidden=args.hidden, num_idx=len(dims),
                        use_gru=not args.no_gru).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        model.train()
        for ep in range(args.epochs):
            for i in np.random.permutation(len(train)):
                feats, Y, q, _ = sessions[train[i]]
                ft = to_dev(feats)
                yt = torch.from_numpy(Y).unsqueeze(0).to(dev)          # (1,T,D)
                qt = torch.from_numpy(q).view(1, -1).to(dev)
                mu, sigma, alpha, logvar = model(ft)
                if args.loss == "nll":
                    loss = hetero_nll(mu, logvar, yt, weight=qt)
                else:
                    loss = (((mu - yt) ** 2).mean(-1) * qt).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            for sid in test:
                feats, Y, q, meta = sessions[sid]
                mu, sigma, _, _ = model(to_dev(feats))
                mu = mu.squeeze(0).cpu().numpy(); sg = sigma.squeeze(0).cpu().numpy()
                for j, (tid, tt, mq) in enumerate(meta):
                    preds.append({
                        "session_id": sid, "turn_id": tid, "mi_quality": mq, "talk_type": tt,
                        "mu": {d: float(mu[j, k]) for k, d in enumerate(dims)},
                        "sigma": {d: float(sg[j, k]) for k, d in enumerate(dims)},
                    })
        print(f"  fold {cf['fold']}: train {len(train)}, predict {len(test)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(preds)} predictions -> {args.out}")

    if "chg" in dims:
        from mpse_mvp.eval.metrics import auc, spearman
        mu = np.array([p["mu"]["chg"] for p in preds]); tt = [p["talk_type"] for p in preds]
        ordv = np.array([{"change": 1, "neutral": 0, "sustain": -1}[t] for t in tt])
        m = np.array([t in ("change", "sustain") for t in tt])
        lab = np.array([1 if t == "change" else 0 for t in tt])[m]
        print(f"HEADLINE chg: AUC(mu->change)={auc(mu[m], lab):.3f}  "
              f"spearman(mu,gold)={spearman(mu, ordv):+.3f}  (weak 0.667)")


if __name__ == "__main__":
    main()
