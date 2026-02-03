from __future__ import annotations
import re, json
import torch

RUBRIC = """You are a clinical-style rater. Given a user's utterance, score the following in [0,1]:
dep (low mood), sad (sadness), anx (anxiety), stress (stress).
Return STRICT JSON: {"dep":0.0,"sad":0.0,"anx":0.0,"stress":0.0}
"""

def load_llm(model_dir: str, device: str = "cuda"):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
    model.to(device); model.eval()
    return tok, model

@torch.no_grad()
def rate_text(tok, model, text: str, max_new_tokens: int = 128):
    prompt = RUBRIC + "\nUSER: " + (text or "") + "\nJSON:"
    inp = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    dec = tok.decode(out[0], skip_special_tokens=True)
    # find last JSON object
    m = re.findall(r"\{[^\}]+\}", dec)
    if not m:
        return None, dec
    try:
        obj = json.loads(m[-1])
        for k in ["dep","sad","anx","stress"]:
            if k in obj:
                obj[k] = float(max(0.0, min(1.0, obj[k])))
        return obj, dec
    except Exception:
        return None, dec
