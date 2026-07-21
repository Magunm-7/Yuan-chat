from __future__ import annotations
import os
import json
from typing import Dict, Any, Optional

from mpse_mvp.sft.build_sft import SYSTEM_PROMPT, format_state_block


def _load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_sft_from_pairs(
    turns_upgraded_path: str,
    pairs_jsonl: str,
    out_jsonl: str,
    inject_state_tokens: bool = True,
    skip_if_no_pair: bool = True,
) -> str:
    """
    Build SFT jsonl using *real* assistant replies from pairs.jsonl (dialogue mode).

    Inputs:
      - turns_upgraded_path: upgraded user turns with mu/sigma/alpha/p_ok/weight, each has turn_id
      - pairs_jsonl: produced by build_turns(dialogue_mode), each row includes:
            user_turn_id, user_text, assistant_text, session_id, ...
      - out_jsonl: output samples, each line:
            {"messages":[...], "sample_weight":..., "meta":{"session_id":..., "turn_id":...}}

    Pairing rule assumed:
      - pairs map user_turn_id -> assistant_text (the assistant reply *after* that user turn)

    Returns:
      - out_jsonl
    """
    rows = _load_jsonl(turns_upgraded_path)
    pairs = _load_jsonl(pairs_jsonl)
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)

    # Map: user_turn_id -> assistant_text
    turn2assist: Dict[int, str] = {}
    for p in pairs:
        try:
            tid = int(p["user_turn_id"])
        except Exception:
            continue
        atext = (p.get("assistant_text") or "").strip()
        if not atext:
            continue
        # If duplicates exist, keep the first by default (stable)
        if tid not in turn2assist:
            turn2assist[tid] = atext

    samples = []
    missing = 0

    for r in rows:
        tid = int(r["turn_id"])
        user_text_raw = (r.get("asr_text", "") or "").strip()

        assistant_text = turn2assist.get(tid, "").strip()
        if not assistant_text:
            missing += 1
            if skip_if_no_pair:
                continue
            assistant_text = "(ASSISTANT_MISSING)"

        if inject_state_tokens:
            user_text = (
                format_state_block(r["mu"], r["sigma"], r["alpha"], r["p_ok"])
                + "\nUSER: "
                + user_text_raw
            )
        else:
            user_text = "USER: " + user_text_raw

        sample = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ],
            "sample_weight": float(r.get("weight", 1.0)),
            "meta": {"session_id": r["session_id"], "turn_id": tid},
        }
        samples.append(sample)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(
        f"[build_sft_from_pairs] wrote {len(samples)} samples to {out_jsonl} "
        f"(missing_pairs={missing}, skip_if_no_pair={skip_if_no_pair})"
    )
    return out_jsonl
