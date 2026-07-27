# -*- coding: utf-8 -*-
"""给现有 SFT 训练数据的 backchannel 目标大幅降权 (只改 sample_weight, 其余不动)。

backchannel = reply_behaviour=="other" 或 target<=3 词 (与烟雾测试同口径)。
new_w = old_sigma_weight * bc_weight(仅backchannel), 再归一化到 train 均值 1。
产出 train_bcw.jsonl, 原 train.jsonl 保留作对照。
"""
import json, argparse, os
import numpy as np

def nw(s): return len((s or "").split())

def parse_sid_tid(npz_path):
    name = os.path.basename(npz_path)
    if name.endswith(".npz"): name = name[:-4]
    sid, tid = name.rsplit("_", 1)
    return sid, int(tid)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/annomi/mm_sft_final/train.jsonl")
    ap.add_argument("--turns", default="data/annomi/turns_labeled.jsonl")
    ap.add_argument("--out",   default="data/annomi/mm_sft_final/train_bcw.jsonl")
    ap.add_argument("--bc_weight", type=float, default=0.1)
    args = ap.parse_args()

    beh = {}
    for l in open(args.turns, encoding="utf-8"):
        r = json.loads(l)
        beh[(str(r["session_id"]), int(r["turn_id"]))] = r.get("reply_behaviour")

    rows = [json.loads(l) for l in open(args.train, encoding="utf-8")]
    n = len(rows)

    is_bc = []
    for r in rows:
        sid, tid = parse_sid_tid(r["npz_path"])
        target = r["messages"][-1]["content"]
        b = beh.get((sid, tid))
        is_bc.append((b == "other") or (nw(target) <= 3))
    is_bc = np.array(is_bc)

    old_w = np.array([r["sample_weight"] for r in rows])
    new_w = old_w.copy()
    new_w[is_bc] *= args.bc_weight
    new_w = new_w / new_w.mean()          # 归一化到均值 1

    for r, w in zip(rows, new_w):
        r["sample_weight"] = float(w)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- 报告 ----
    def neff(w): return (w.sum()**2) / (w**2).sum()
    print(f"train 样本: {n}   backchannel: {is_bc.sum()} ({100*is_bc.mean():.0f}%)   bc_weight={args.bc_weight}")
    print(f"\n每样本权重 (归一化后, 均值1):")
    print(f"  backchannel : mean={new_w[is_bc].mean():.3f}")
    print(f"  substantive : mean={new_w[~is_bc].mean():.3f}")
    print(f"\n梯度'质量'占比 (Σw 份额 = 实际训练关注度):")
    print(f"  降权前: backchannel {100*old_w[is_bc].sum()/old_w.sum():.0f}%  /  substantive {100*old_w[~is_bc].sum()/old_w.sum():.0f}%")
    print(f"  降权后: backchannel {100*new_w[is_bc].sum()/new_w.sum():.0f}%  /  substantive {100*new_w[~is_bc].sum()/new_w.sum():.0f}%")
    print(f"\n有效样本量 N_eff (衡量降权后还剩多少'有效'数据):")
    print(f"  降权前 {neff(old_w):.0f} / {n}   ->   降权后 {neff(new_w):.0f} / {n}")
    print(f"\nwrote -> {args.out}")

if __name__ == "__main__":
    main()
