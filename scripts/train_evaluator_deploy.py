"""
训练一个**可部署**的评估器(MPSE_MM)并保存权重。

train_mpse.py 是 session-CV,每折训完只留 out-of-fold 预测、模型即弃,
因此项目此前没有任何可用于实时推理的评估器权重。本脚本用全部 session
训练一份并落盘,同时保存 state_tag 所需的分位阈值,供 demo 服务加载。

  python scripts/train_evaluator_deploy.py --out outputs/evaluator/mpse_deploy.pt
"""
from __future__ import annotations
import os, json, argparse
import numpy as np
import torch

from train_mpse import load_sessions          # 复用同一套数据装载,保证口径一致
from mpse_mvp.mpse.model_mm import MPSE_MM, hetero_nll
from mpse_mvp.mm.state_tag import fit_thresholds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/annomi/feats/index.jsonl")
    ap.add_argument("--labels", default="data/annomi/turns_labeled.jsonl")
    ap.add_argument("--out", default="outputs/evaluator/mpse_deploy.pt")
    ap.add_argument("--dims", default="chg,aro,val")
    ap.add_argument("--modalities", default="text,audio,video")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")   # 0.89M 参数, CPU 足够, 不抢 GPU
    args = ap.parse_args()

    mods, dims = tuple(args.modalities.split(",")), tuple(args.dims.split(","))
    sessions, feat_dims = load_sessions(args.index, args.labels, mods, dims)
    print(f"sessions={len(sessions)} feat_dims={feat_dims} dims={dims}")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = args.device
    model = MPSE_MM(feat_dims, mods, hidden=args.hidden, num_idx=len(dims)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sids = list(sessions)

    model.train()
    for ep in range(args.epochs):
        tot = 0.0
        for i in np.random.permutation(len(sids)):
            feats, Y, q, _ = sessions[sids[i]]
            ft = {m: torch.from_numpy(feats[m]).unsqueeze(0).to(dev) for m in mods}
            yt = torch.from_numpy(Y).unsqueeze(0).to(dev)
            qt = torch.from_numpy(q).view(1, -1).to(dev)
            mu, sigma, alpha, logvar = model(ft)
            loss = hetero_nll(mu, logvar, yt, weight=qt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        if (ep + 1) % 10 == 0:
            print(f"  ep {ep+1:>3d}/{args.epochs}  loss={tot/len(sids):.4f}")

    # 全量前向一次,拿 mu 分布来定 state_tag 的分位阈值
    model.eval()
    all_mu = []
    with torch.no_grad():
        for sid in sids:
            feats, _, _, _ = sessions[sid]
            ft = {m: torch.from_numpy(feats[m]).unsqueeze(0).to(dev) for m in mods}
            mu, _, _, _ = model(ft)
            all_mu.append(mu.squeeze(0).cpu().numpy())
    all_mu = np.concatenate(all_mu, axis=0)                       # (N, K)
    mus = [{d: float(row[k]) for k, d in enumerate(dims)} for row in all_mu]
    thr = fit_thresholds(mus)
    print("state_tag thresholds:", thr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "feat_dims": feat_dims, "modalities": list(mods), "dims": list(dims),
        "hidden": args.hidden, "thresholds": thr,
        "mu_mean": all_mu.mean(0).tolist(), "mu_std": all_mu.std(0).tolist(),
    }, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
