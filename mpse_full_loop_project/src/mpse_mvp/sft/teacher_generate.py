
from __future__ import annotations
import os, json
from tqdm import tqdm

SYSTEM_PROMPT = """You are a supportive interview-style therapist. You should:
- respond concisely and empathetically
- ask one clear follow-up question
- avoid diagnosis; encourage seeking professional help if needed
"""

def generate_teacher_sft(turns_upgraded_path: str, out_jsonl: str, base_model_dir: str,
                         max_new_tokens: int = 128, device: str = "cuda"):
    """
    Generate therapist responses using the base LLM (self-distillation) so SFT has real targets.
    This is ONLY for MVP closed-loop; replace with real therapist transcripts later.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(base_model_dir, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model_dir)
    model.to(device)
    model.eval()

    rows = [json.loads(l) for l in open(turns_upgraded_path, encoding="utf-8")]
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in tqdm(rows, desc="Teacher gen"):
            user_text = r.get("asr_text","").strip()
            prompt_msgs = [
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content": user_text},
            ]
            if hasattr(tok, "apply_chat_template"):
                prompt = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            else:
                prompt = SYSTEM_PROMPT + "\nUSER: " + user_text + "\nASSISTANT:"

            inputs = tok(prompt, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            full = tok.decode(out[0], skip_special_tokens=True)
            # naive extraction: take tail after last "ASSISTANT"
            assistant = full.split("ASSISTANT:")[-1].strip()
            if not assistant:
                assistant = "(THERAPIST_PLACEHOLDER)"

            sample = {
                "messages": [
                    {"role":"system","content":SYSTEM_PROMPT},
                    {"role":"user","content": user_text},
                    {"role":"assistant","content": assistant},
                ],
                "sample_weight": float(r.get("weight", 1.0)),
                "meta": {"session_id": r["session_id"], "turn_id": r["turn_id"]},
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return out_jsonl
