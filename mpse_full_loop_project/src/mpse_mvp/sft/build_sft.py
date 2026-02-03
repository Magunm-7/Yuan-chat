from __future__ import annotations
import os, json

SYSTEM_PROMPT = """You are a supportive interview-style therapist. You should:
- respond concisely and empathetically
- ask one clear follow-up question
- avoid diagnosis; encourage seeking professional help if needed
"""

def format_state_block(mu: dict, sigma: dict, alpha: dict, p_ok: dict):
    return (
        f"[STATE] mu={mu}\n"
        f"[UNC] sigma={sigma}\n"
        f"[EVIDENCE] alpha={alpha}\n"
        f"[OK_PROB] p_ok={p_ok}\n"
    )

def build_sft(turns_upgraded_path: str, out_jsonl: str, inject_state_tokens: bool = True):
    rows = [json.loads(l) for l in open(turns_upgraded_path, encoding="utf-8")]
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)

    samples = []
    # We only have user utterances; therapist responses are placeholders.
    # Replace this later with real therapist transcripts (or generated therapist turns).
    for r in rows:
        user_text = r.get("asr_text","").strip()
        if inject_state_tokens:
            user_text = format_state_block(r["mu"], r["sigma"], r["alpha"], r["p_ok"]) + "\nUSER: " + user_text
        else:
            user_text = "USER: " + user_text

        assistant_text = "(THERAPIST_PLACEHOLDER)"  # replace when you have therapist turns

        sample = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ],
            "sample_weight": float(r.get("weight", 1.0)),
            "meta": {"session_id": r["session_id"], "turn_id": r["turn_id"]}
        }
        samples.append(sample)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return out_jsonl
