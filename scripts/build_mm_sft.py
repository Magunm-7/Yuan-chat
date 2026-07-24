"""
Build the B-stage (generator) SFT dataset from AnnoMI, by joining three files on
(session_id, turn_id):

  turns_labeled.jsonl  -> text (client utterance), therapist_reply (SFT target),
                          q_face, client_face, reply_behaviour
  feats/index.jsonl+npz-> audio_emb(768), video_emb(768) pooled per-turn embeddings
  pred_mm2.jsonl       -> mu{chg,aro,val} (aux target), sigma -> sample_weight

Emits, under data/annomi/mm_sft/:
  npz/<sid>_<tid>.npz  with the exact keys train_mm_sft expects:
      audio_feat(768,) video_feat(768,) alpha(2,) mu(3,)
  train.jsonl / holdout.jsonl  rows: {npz_path, messages, sample_weight}

The multimodal STATE enters only through the audio/video soft-prefix (alpha-gated),
never as numbers in the prompt text (numbers-in-text were verified to be noise).
Video is alpha-down-weighted when the active-speaker face is not the client.

  python scripts/build_mm_sft.py            # full build
  python scripts/build_mm_sft.py --limit 20 # smoke
"""
from __future__ import annotations
import os, json, argparse
import numpy as np
from mpse_mvp.mm.state_tag import fit_thresholds, state_tag

SYS = ("You are an experienced motivational interviewing (MI) counselor in an "
       "ongoing session. Given the conversation so far and the client's latest "
       "utterance (with the nonverbal cues provided), reply with a single brief, "
       "empathic counselor response that fits the topic and gently supports the "
       "client's own motivation to change.")


def _load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _key(r):
    return (str(r["session_id"]), int(r["turn_id"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default="data/annomi/turns_labeled.jsonl")
    ap.add_argument("--index", default="data/annomi/feats/index.jsonl")
    ap.add_argument("--pred", default="data/annomi/pred_mm2.jsonl")
    ap.add_argument("--splits", default="data/annomi/splits.json")
    ap.add_argument("--out_dir", default="data/annomi/mm_sft")
    ap.add_argument("--lam", type=float, default=8.0, help="sigma weight sharpness w=exp(-lam*sigma_bar)")
    ap.add_argument("--history", type=int, default=10, help="max prior (client,therapist) turns kept as dialogue context")
    ap.add_argument("--seq", action="store_true", help="store audio/video SEQUENCES + text_emb (for cross-attention); --index must point at feats_seq")
    ap.add_argument("--state_tag", action="store_true", help="prepend evaluator-mu state tag to current client utterance (mu -> text tag -> prompt)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    turns = {_key(r): r for r in _load_jsonl(args.turns)}
    feats = {_key(r): r for r in _load_jsonl(args.index)}
    preds = {_key(r): r for r in _load_jsonl(args.pred)}
    splits = json.load(open(args.splits, encoding="utf-8"))

    hold = splits.get("holdout", [])
    if isinstance(hold, dict):
        # holdout = {"test":[...], "val":[...], "train":[...]}; eval on unseen "test"
        hold = hold.get("test") or hold.get("sessions") or hold.get("holdout_sessions") or []
    hold_sessions = set(str(s) for s in hold)
    print(f"holdout sessions ({len(hold_sessions)}): {sorted(hold_sessions)[:8]}...")

    keys = sorted(turns.keys() & feats.keys() & preds.keys(),
                  key=lambda k: (int(k[0]) if k[0].isdigit() else 1e9, k[1]))
    if args.limit:
        keys = keys[:args.limit]

    # full dialogue-history pool: every client turn's (text, reply) per session, ordered
    hist_pool = {}
    for r in turns.values():
        ct = (r.get("text") or "").strip(); rp = (r.get("therapist_reply") or "").strip()
        if ct and rp:
            hist_pool.setdefault(str(r["session_id"]), []).append((int(r["turn_id"]), ct, rp))
    for s in hist_pool:
        hist_pool[s].sort()

    tag_thr = None
    if args.state_tag:
        tag_thr = fit_thresholds([preds[k]["mu"] for k in keys])
        print("state_tag tertile thresholds:", {d: [round(x, 3) for x in tag_thr[d]] for d in tag_thr})

    npz_dir = os.path.join(args.out_dir, "npz")
    os.makedirs(npz_dir, exist_ok=True)

    # first pass: assemble rows + raw sigma_bar, split by session
    rows, sbar_train = [], []
    q_a_all, q_v_all, cf_all = [], [], []
    skipped = 0
    for sid, tid in keys:
        t = turns[(sid, tid)]
        reply = (t.get("therapist_reply") or "").strip()
        client = (t.get("text") or "").strip()
        if not reply or not client:
            skipped += 1
            continue

        z = np.load(feats[(sid, tid)]["npz"])
        q_a = float(np.clip(z["q_audio"], 0.1, 1.0))
        # video is not speaker-attributed: trust it only when the active-speaker
        # face was the client (client_face==1), else strongly down-weight.
        client_face = int(t.get("client_face", 0))
        q_v = float(np.clip(z["q_video"], 0.1, 1.0)) * (1.0 if client_face else 0.3)
        alpha = np.array([q_a, q_v], dtype=np.float32)
        q_a_all.append(q_a); q_v_all.append(q_v); cf_all.append(client_face)

        p = preds[(sid, tid)]
        mu = np.array([p["mu"]["chg"], p["mu"]["aro"], p["mu"]["val"]], dtype=np.float32)
        sbar = float(np.mean([p["sigma"]["chg"], p["sigma"]["aro"], p["sigma"]["val"]]))

        npz_path = os.path.join(npz_dir, f"{sid}_{tid}.npz")
        if args.seq:
            np.savez(npz_path, audio_seq=z["audio_seq"].astype(np.float32),
                     video_seq=z["video_seq"].astype(np.float32),
                     text_emb=z["text_emb"].astype(np.float32), alpha=alpha, mu=mu)
        else:
            np.savez(npz_path, audio_feat=z["audio_emb"].astype(np.float32),
                     video_feat=z["video_emb"].astype(np.float32), alpha=alpha, mu=mu)

        # prepend up to --history prior (client, therapist) turns from this session
        hist = [(t2, c2, r2) for (t2, c2, r2) in hist_pool.get(sid, []) if t2 < tid][-args.history:]
        messages = [{"role": "system", "content": SYS}]
        for (_t, _c, _r) in hist:
            messages.append({"role": "user", "content": _c})
            messages.append({"role": "assistant", "content": _r})
        cur = (state_tag(preds[(sid, tid)]["mu"], tag_thr) + "\n" + client) if args.state_tag else client
        messages.append({"role": "user", "content": cur})         # current client (+ optional state tag)
        messages.append({"role": "assistant", "content": reply})  # target therapist reply
        is_hold = sid in hold_sessions
        rows.append({"npz_path": npz_path, "messages": messages,
                     "_sbar": sbar, "_hold": is_hold})
        if not is_hold:
            sbar_train.append(sbar)

    # sigma -> sample_weight, normalized to mean 1 over TRAIN rows
    sbar_train = np.asarray(sbar_train)
    w_raw = np.exp(-args.lam * sbar_train)
    w_mean = float(w_raw.mean()) if len(w_raw) else 1.0
    for r in rows:
        r["sample_weight"] = float(np.exp(-args.lam * r.pop("_sbar")) / w_mean)

    # write train / holdout indexes
    os.makedirs(args.out_dir, exist_ok=True)
    n_tr = n_ho = 0
    ftr = open(os.path.join(args.out_dir, "train.jsonl"), "w", encoding="utf-8")
    fho = open(os.path.join(args.out_dir, "holdout.jsonl"), "w", encoding="utf-8")
    for r in rows:
        is_hold = r.pop("_hold")
        line = json.dumps(r, ensure_ascii=False)
        if is_hold:
            fho.write(line + "\n"); n_ho += 1
        else:
            ftr.write(line + "\n"); n_tr += 1
    ftr.close(); fho.close()

    ws = np.array([r["sample_weight"] for r in rows])
    print(f"\n=== built {len(rows)} rows  (skipped {skipped} empty) ===")
    print(f"  train={n_tr}  holdout={n_ho}")
    print(f"  alpha q_audio: mean={np.mean(q_a_all):.3f}  q_video(gated): mean={np.mean(q_v_all):.3f}  "
          f"client_face rate={np.mean(cf_all):.2f}")
    print(f"  sample_weight: mean={ws.mean():.3f} std={ws.std():.3f} [{ws.min():.3f},{ws.max():.3f}]")
    print(f"  wrote {args.out_dir}/{{train,holdout}}.jsonl + npz/")


if __name__ == "__main__":
    main()
