# -*- coding: utf-8 -*-
"""v2 精细加权: full 金标准子类(按 utterance join) + 会话质量 -> 每条 SFT 目标一个权重。
new_w = old_sigma_weight * behaviour_factor * session_factor, 归一化到 train 均值 1。"""
import json, csv, argparse, os
import numpy as np
from collections import defaultdict, Counter

def nw(s): return len((s or "").split())
def parse_sid_tid(p):
    n = os.path.basename(p)
    if n.endswith(".npz"): n = n[:-4]
    sid, tid = n.rsplit("_", 1); return sid, int(tid)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/annomi/mm_sft_final/train.jsonl")
    ap.add_argument("--turns", default="data/annomi/turns_labeled.jsonl")
    ap.add_argument("--full",  default="data/annomi/AnnoMI-full.csv")
    ap.add_argument("--out",   default="data/annomi/mm_sft_final/train_bcw2.jsonl")
    ap.add_argument("--bc_weight", type=float, default=0.02)
    args = ap.parse_args()

    # --- full: 去重, 按 session 排序, 记 therapist 子类 ---
    with open(args.full, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by = defaultdict(list)
    for r in rows: by[(r["transcript_id"], r["utterance_id"])].append(r)
    def maj(rs,c): return Counter(x[c] for x in rs).most_common(1)[0][0]
    sess = defaultdict(list)
    for k,rs in by.items():
        r0=rs[0]
        sess[r0["transcript_id"]].append({
            "uid": int(r0["utterance_id"]), "who": r0["interlocutor"],
            "refl": maj(rs,"reflection_exists"), "refl_s": maj(rs,"reflection_subtype"),
            "q": maj(rs,"question_exists"), "q_s": maj(rs,"question_subtype"),
            "inp": maj(rs,"therapist_input_exists"), "inp_s": maj(rs,"therapist_input_subtype")})
    for t in sess: sess[t].sort(key=lambda u:u["uid"])
    client_uids = {t: sorted(u["uid"] for u in sess[t] if u["who"]=="client") for t in sess}

    # --- turns_labeled: (sid,tid) -> client_uid, mi_quality, reply_text ---
    meta = {}
    for l in open(args.turns, encoding="utf-8"):
        r = json.loads(l)
        meta[(str(r["session_id"]), int(r["turn_id"]))] = (int(r["utterance_id"]), r["mi_quality"], r.get("therapist_reply",""))

    def reply_subtypes(sid, client_uid):
        us = sess.get(sid, [])
        cu = client_uids.get(sid, [])
        nxt = next((c for c in cu if c > client_uid), 10**9)
        flags = set()
        for u in us:
            if u["who"]!="therapist": continue
            if not (client_uid < u["uid"] < nxt): continue
            if u["refl"]=="True": flags.add("REFL:"+u["refl_s"])
            if u["q"]=="True": flags.add("Q:"+u["q_s"])
            if u["inp"]=="True": flags.add("INPUT:"+u["inp_s"])
        return flags

    def factor(sid, tid, reply_text):
        info = meta.get((sid,tid))
        if not info: return 1.0, 1.0, "?", set()
        cuid, mi, _ = info
        # behaviour factor
        if nw(reply_text) <= 3:
            bf, cat = args.bc_weight, "backchannel"
            flags = set()
        else:
            flags = reply_subtypes(sid, cuid)
            bf = 1.0
            if "REFL:complex" in flags: bf *= 1.3
            if "Q:open" in flags: bf *= 1.3
            if "Q:closed" in flags: bf *= 0.6
            if "INPUT:negotiation" in flags: bf *= 1.2
            if "INPUT:options" in flags: bf *= 1.1
            if "INPUT:advice" in flags: bf *= 0.4
            if "INPUT:information" in flags: bf *= 0.6
            if not flags: bf *= 0.5; cat="long-other"      # 无R/Q/I: 事务/过渡
            else: cat="mix"
            bf = float(np.clip(bf, 0.05, 1.5))
        sf = 0.4 if mi=="low" else 1.0
        return bf, sf, cat, flags

    # --- 应用到现有 train.jsonl ---
    tr = [json.loads(l) for l in open(args.train, encoding="utf-8")]
    bfs, sfs, cats = [], [], []
    for r in tr:
        sid, tid = parse_sid_tid(r["npz_path"])
        bf, sf, cat, flags = factor(sid, tid, r["messages"][-1]["content"])
        bfs.append(bf); sfs.append(sf); cats.append(cat)
    old_w = np.array([r["sample_weight"] for r in tr])
    new_w = old_w * np.array(bfs) * np.array(sfs)
    new_w = new_w / new_w.mean()
    for r, w in zip(tr, new_w): r["sample_weight"] = float(w)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in tr: f.write(json.dumps(r, ensure_ascii=False)+"\n")

    # --- 报告 ---
    n=len(tr); bfs=np.array(bfs); sfs=np.array(sfs)
    def neff(w): return (w.sum()**2)/(w**2).sum()
    print(f"train {n}   bc_weight={args.bc_weight}")
    print(f"backchannel: {(bfs==args.bc_weight).sum()}   低质量会话: {(sfs<1).sum()}")
    print("\nbehaviour_factor 分布:")
    for lo,hi,lab in [(0,0.03,'~0.02 backchannel'),(0.03,0.5,'0.05-0.5 强降'),(0.5,0.9,'0.5-0.9 降'),(0.9,1.1,'~1.0 中性'),(1.1,1.6,'1.1-1.5 升')]:
        m=((bfs>=lo)&(bfs<hi)).sum(); print(f"  {lab:18s}: {m:5d} ({100*m/n:4.0f}%)")
    print(f"\n梯度质量占比 (Σw):")
    up = new_w[bfs>1.05].sum(); dn = new_w[bfs<0.9].sum(); neu = new_w.sum()-up-dn
    print(f"  升权动作 {100*up/new_w.sum():.0f}%  /  中性 {100*neu/new_w.sum():.0f}%  /  降权 {100*dn/new_w.sum():.0f}%")
    print(f"有效样本量 N_eff: 旧 {neff(old_w):.0f} -> v2 {neff(new_w):.0f} / {n}")
    print(f"权重 min/max: {new_w.min():.3f} / {new_w.max():.3f}")
    print(f"\nwrote -> {args.out}")

if __name__ == "__main__":
    main()
