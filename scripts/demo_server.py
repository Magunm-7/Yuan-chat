"""
实时多模态 MI 咨询 demo 服务(端到端跑通:音视频 -> 评估器 -> 状态标记 -> 生成器)。

  浏览器采集摄像头+麦克风 -> /infer
      Whisper  : ASR 文本 + audio_emb(768)
      CLIP     : video_emb(768)
      MiniLM   : text_emb(384)
      MPSE     : -> mu/sigma(chg/aro/val)
      state_tag: mu -> 自然语言状态标记, 进 prompt
      Qwen+LoRA: -> 咨询师回复

启动(服务器):
  python scripts/demo_server.py --port 8000
本地开隧道后浏览器访问 http://localhost:8000 :
  ssh -L 8000:localhost:8000 autodl-mpse
"""
from __future__ import annotations
import os, io, json, base64, argparse
import numpy as np
import torch

HF = os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
SYSTEM_PROMPT = ("You are an experienced motivational interviewing (MI) counselor in an ongoing "
                 "session. Given the conversation so far and the client's latest utterance (with "
                 "the nonverbal cues provided), reply with a single brief, empathic counselor "
                 "response that fits the topic and gently supports the client's own motivation to change.")

STATE: dict[str, dict] = {}          # session_id -> {feats:{text,audio,video}, history:[...]}
M: dict = {}                          # 已加载的模型

# ---- best-of-n 的复合 reward(与 scripts/reward_model.py 同一套口径)----
# 高温采样保证候选多样, 再用 reward 挑质量 —— 把"多样性"和"质量"解耦,
# 避免单一温度旋钮顾此失彼(0.7 滑向说教 / 0.4 退化成只会 Yeah)。
import re
BEHAV_REWARD = {"reflection": 1.0, "question": 0.6, "other": 0.2, "therapist_input": -0.5}
REL_MID, REL_SLOPE = 0.28, 12.0
W_REL, W_FAB, W_LEN, TARGET_LEN = 0.4, 1.0, 0.02, 15


def _norm(v):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


def score_candidates(user_text: str, replies: list[str], history_texts: list[str]):
    """复合 reward = behaviour x relevance门控 + relevance
                    - 忠实性(实体级, 仅 reflection) - 长度惩罚。
    忠实性用 faithfulness.score_faithfulness(实体级+concreteness), 只对 reflection 生效
    (question/therapist_input 合法引入信息, 查它们=误伤)。长度惩罚 W_LEN=0.02 拉回 gold≈12 词。"""
    from faithfulness import score_faithfulness
    ctx_text = " ".join([user_text] + [h for h in history_texts if h.strip()])
    v = _norm(np.asarray(M["text"].encode([user_text] + replies), dtype=np.float32))
    v_user, v_rep = v[0], v[1:]
    rel = v_rep @ v_user
    coef, intercept, keys = M["behav"]
    labs = [keys[i] for i in (v_rep @ coef.T + intercept).argmax(1)]
    out = []
    for i, (r, lab) in enumerate(zip(replies, labs)):
        faith = 0.0
        if lab == "reflection":
            faith, _ = score_faithfulness(r, ctx_text)
        words = len(r.split())
        gate = 1.0 / (1.0 + np.exp(-(float(rel[i]) - REL_MID) * REL_SLOPE))
        total = (BEHAV_REWARD[lab] * gate + W_REL * float(rel[i])
                 - W_FAB * faith - W_LEN * max(0, words - TARGET_LEN))
        out.append({"reply": r, "behaviour": lab, "rel": round(float(rel[i]), 3),
                    "faith": round(faith, 2), "words": words, "score": round(float(total), 3)})
    return sorted(out, key=lambda d: -d["score"])


def _decode_audio(b64: str) -> np.ndarray:
    """浏览器 MediaRecorder 的 webm/opus -> 16k 单声道 float32。"""
    import av
    raw = base64.b64decode(b64.split(",")[-1])
    container = av.open(io.BytesIO(raw))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []
    for frame in container.decode(audio=0):
        for f in resampler.resample(frame):
            chunks.append(f.to_ndarray().ravel())
    if not chunks:
        return np.zeros(16000, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def _decode_frames(b64_list: list[str]) -> np.ndarray:
    from PIL import Image
    out = []
    for b in b64_list:
        raw = base64.b64decode(b.split(",")[-1])
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        out.append(np.array(img, dtype=np.uint8))
    return np.stack(out) if out else np.zeros((0, 224, 224, 3), np.uint8)


def load_models(base_dir: str, lora_dir: str, evaluator_ckpt: str, device: str):
    from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                              AutoTokenizer, AutoModelForCausalLM)
    from sentence_transformers import SentenceTransformer
    from mpse_mvp.mm.encoders import CLIPVideoEncoder
    from mpse_mvp.mpse.model_mm import MPSE_MM

    print("[load] whisper ...")
    wdir = "openai/whisper-small"
    M["wproc"] = WhisperProcessor.from_pretrained(wdir)
    M["whisper"] = WhisperForConditionalGeneration.from_pretrained(wdir).to(device).eval()

    print("[load] clip / minilm ...")
    M["clip"] = CLIPVideoEncoder("openai/clip-vit-base-patch32", device=device)
    M["text"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    print("[load] evaluator ...")
    ck = torch.load(evaluator_ckpt, map_location="cpu")
    ev = MPSE_MM(ck["feat_dims"], tuple(ck["modalities"]), hidden=ck["hidden"],
                 num_idx=len(ck["dims"]))
    ev.load_state_dict(ck["state_dict"])
    M["ev"] = ev.to(device).eval()
    M["dims"] = ck["dims"]
    M["thr"] = {k: tuple(v) for k, v in ck["thresholds"].items()}

    print("[load] behaviour scorer ...")
    d = np.load("outputs/evaluator/behaviour_clf.npz", allow_pickle=True)
    M["behav"] = (d["coef"], d["intercept"], [str(k) for k in d["keys"]])

    print("[load] generator ...")
    tok = AutoTokenizer.from_pretrained(base_dir, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.bfloat16).to(device)
    from peft import PeftModel
    # 训练脚本把 adapter 存在 <out>/lora_adapter/,这里两种写法都接受
    if os.path.isdir(os.path.join(lora_dir, "lora_adapter")):
        lora_dir = os.path.join(lora_dir, "lora_adapter")
    lm = PeftModel.from_pretrained(lm, lora_dir).eval()
    M["tok"], M["lm"] = tok, lm
    M["device"] = device
    print("[load] done.")


@torch.no_grad()
def run_turn(session_id: str, audio_b64: str | None, frames_b64: list[str],
             typed_text: str | None, max_new_tokens: int, history_turns: int,
             temperature: float = 0.6, system_prompt: str | None = None,
             best_of: int = 8):
    dev = M["device"]
    st = STATE.setdefault(session_id, {"feats": {"text": [], "audio": [], "video": []},
                                       "history": []})

    # --- 1) 音频: ASR 文本 + audio_emb ---
    if audio_b64:
        wav = _decode_audio(audio_b64)
        feats = M["wproc"](wav, sampling_rate=16000, return_tensors="pt")
        inp = feats["input_features"].to(dev)
        ids = M["whisper"].generate(inp, language="en", task="transcribe", max_new_tokens=200)
        asr = M["wproc"].batch_decode(ids, skip_special_tokens=True)[0].strip()
        audio_emb = M["whisper"].model.encoder(inp).last_hidden_state.mean(1)[0].float().cpu().numpy()
    else:
        asr = ""
        audio_emb = np.zeros(M["ev"].proj["audio"][0].in_features, np.float32)

    user_text = (typed_text or "").strip() or asr
    if not user_text:
        return {"error": "没有听到内容,请重说或直接打字"}

    # --- 2) 视频 / 文本 embedding ---
    frames = _decode_frames(frames_b64)
    video_emb = M["clip"].encode(frames)[0][0].float().cpu().numpy()
    text_emb = M["text"].encode([user_text])[0].astype(np.float32)

    # --- 3) 评估器: 整段会话重跑(GRU 需要完整轮次序列), 取最新一轮的 mu ---
    f = st["feats"]
    f["text"].append(text_emb); f["audio"].append(audio_emb); f["video"].append(video_emb)
    ft = {m: torch.from_numpy(np.stack(f[m])).unsqueeze(0).float().to(dev)
          for m in M["ev"].mods}
    mu_t, sigma_t, alpha_t, _ = M["ev"](ft)
    mu = {d: float(mu_t[0, -1, k]) for k, d in enumerate(M["dims"])}
    sigma = {d: float(sigma_t[0, -1, k]) for k, d in enumerate(M["dims"])}
    alpha = [float(a) for a in alpha_t[0, -1]]

    # --- 4) 状态标记 -> prompt -> 生成 ---
    from mpse_mvp.mm.state_tag import state_tag
    tag = state_tag(mu, M["thr"])
    msgs = [{"role": "system", "content": system_prompt or SYSTEM_PROMPT}]
    msgs += st["history"][-2 * history_turns:]
    msgs.append({"role": "user", "content": tag + "\n" + user_text})

    tok = M["tok"]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to(dev)
    # best-of-n:高温采样保证候选多样,再由 reward 挑质量。
    # num_return_sequences 一次前向出 n 条,共享 prompt 计算,延迟远低于串行 n 次。
    out = M["lm"].generate(ids, max_new_tokens=max_new_tokens, do_sample=True,
                           temperature=max(temperature, 0.9 if best_of > 1 else temperature),
                           top_p=0.95, repetition_penalty=1.05,
                           num_return_sequences=max(1, best_of),
                           pad_token_id=tok.eos_token_id)
    cands = []
    for o in out:
        c = tok.decode(o[ids.shape[1]:], skip_special_tokens=True).strip()
        c = c.split("</think>")[-1].strip()               # Qwen3 若吐 think 块则剥掉
        if c and c not in cands:
            cands.append(c)
    hist_texts = [m["content"] for m in st["history"][-2 * history_turns:]]
    ranked = score_candidates(user_text, cands, hist_texts) if len(cands) > 1 else              [{"reply": cands[0] if cands else "", "behaviour": "-", "rel": 0.0,
               "fab": 0.0, "score": 0.0}]
    reply = ranked[0]["reply"]

    st["history"].append({"role": "user", "content": tag + "\n" + user_text})
    st["history"].append({"role": "assistant", "content": reply})

    return {"asr": asr, "user_text": user_text, "mu": mu, "sigma": sigma,
            "alpha": alpha, "tag": tag, "reply": reply,
            "candidates": ranked, "turn": len(st["history"]) // 2}


from pydantic import BaseModel      # noqa: E402  (需在模块级, 否则 FastAPI 解析不到注解)
from typing import Optional, List


class GenReq(BaseModel):
    messages: List[dict]
    n: int = 8
    temperature: float = 0.9
    max_new_tokens: int = 96


class Req(BaseModel):
    session_id: str = "demo"
    audio: Optional[str] = None
    frames: List[str] = []
    text: Optional[str] = None
    max_new_tokens: int = 96
    history_turns: int = 20
    temperature: float = 0.6
    system_prompt: Optional[str] = None
    best_of: int = 8


def build_app(html_path: str):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return open(html_path, encoding="utf-8").read()

    @app.post("/infer")
    def infer(r: Req):
        try:
            return run_turn(r.session_id, r.audio, r.frames, r.text,
                            r.max_new_tokens, r.history_turns, r.temperature,
                            r.system_prompt, r.best_of)
        except Exception as e:
            import traceback; traceback.print_exc()
            return {"error": f"{type(e).__name__}: {e}"}

    @app.post("/generate")
    def generate(r: GenReq):
        """离线批量采样:输入完整 messages,输出 n 个候选按 reward 排序。"""
        try:
            tok, dev = M["tok"], M["device"]
            try:
                text = tok.apply_chat_template(r.messages, tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                text = tok.apply_chat_template(r.messages, tokenize=False,
                                               add_generation_prompt=True)
            ids = tok(text, return_tensors="pt", truncation=True,
                      max_length=1536).input_ids.to(dev)
            with torch.no_grad():
                out = M["lm"].generate(ids, max_new_tokens=r.max_new_tokens, do_sample=True,
                                       temperature=r.temperature, top_p=0.95,
                                       repetition_penalty=1.05,
                                       num_return_sequences=max(1, r.n),
                                       pad_token_id=tok.eos_token_id)
            cands = []
            for o in out:
                c = tok.decode(o[ids.shape[1]:], skip_special_tokens=True).strip()
                c = c.split("</think>")[-1].strip()
                if c and c not in cands:
                    cands.append(c)
            if not cands:
                return {"candidates": []}
            users = [m["content"] for m in r.messages if m.get("role") == "user"]
            # 训练集 prompt 自带状态标记, relevance 要用标记之后的真实话语
            cur = users[-1].split("]" + chr(10), 1)[-1] if users else ""
            hist = [m["content"] for m in r.messages[:-1]]
            return {"candidates": score_candidates(cur, cands, hist), "user_text": cur}
        except Exception as e:
            import traceback; traceback.print_exc()
            return {"error": f"{type(e).__name__}: {e}"}

    @app.post("/reset")
    def reset(r: Req):
        STATE.pop(r.session_id, None)
        return {"ok": True}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--lora", default="outputs/mm_sft/qwen7b_v3_2048")
    ap.add_argument("--evaluator", default="outputs/evaluator/mpse_deploy.pt")
    ap.add_argument("--html", default="scripts/demo_ui.html")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    load_models(args.base, args.lora, args.evaluator, dev)
    import uvicorn
    uvicorn.run(build_app(args.html), host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
