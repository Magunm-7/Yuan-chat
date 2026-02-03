from __future__ import annotations
import json
import numpy as np

def fuse_labels(yT: dict, yA: dict, yV: dict, qT: float, qA: float, qV: float, idx_names: list[str]):
    denom = qT + qA + qV + 1e-8
    y = {}
    for k in idx_names:
        if k == "microexpr_rate":
            # microexpr_rate comes mainly from video heuristic itself
            y[k] = float(np.clip(yV.get(k, 0.0), 0.0, 1.0))
        else:
            y[k] = float((qT*yT.get(k,0.5) + qA*yA.get(k,0.5) + qV*yV.get(k,0.5)) / denom)
    return y

def text_rater_rule(text: str, idx_names: list[str]):
    t = (text or "").strip()
    # simple keyword heuristics for MVP (replace with LLM rater if enabled)
    low_words = ["没劲","难受","空","麻木","不想","睡不着","焦虑","压力","崩溃","烦"]
    score = 0
    for w in low_words:
        if w in t:
            score += 1
    base = min(1.0, 0.2 + 0.15*score)
    y = {}
    for k in idx_names:
        if k == "dep": y[k] = base
        elif k == "sad": y[k] = min(1.0, base*0.9 + 0.05)
        elif k == "anx": y[k] = min(1.0, base*0.7 + (0.2 if "焦虑" in t else 0.0))
        elif k == "stress": y[k] = min(1.0, base*0.6 + (0.2 if "压力" in t else 0.0))
        elif k == "microexpr_rate": y[k] = 0.0
    return y

def audio_rater_heuristic(q_audio: float, stress_proxy: float, idx_names: list[str]):
    # map stress_proxy to anx/stress; dep/sad weakly from low quality
    y = {}
    for k in idx_names:
        if k == "stress": y[k] = float(np.clip(0.2 + 0.7*stress_proxy, 0.0, 1.0))
        elif k == "anx": y[k] = float(np.clip(0.2 + 0.6*stress_proxy, 0.0, 1.0))
        elif k == "dep": y[k] = float(np.clip(0.4 - 0.2*q_audio, 0.0, 1.0))
        elif k == "sad": y[k] = float(np.clip(0.35 - 0.15*q_audio, 0.0, 1.0))
        elif k == "microexpr_rate": y[k] = 0.0
    return y

def video_rater_heuristic(q_video: float, microexpr_rate: float, idx_names: list[str]):
    # dep/sad slightly tied to low microexpr (flat affect); microexpr_rate is itself
    y = {}
    flat = 1.0 - microexpr_rate
    for k in idx_names:
        if k == "microexpr_rate": y[k] = float(np.clip(microexpr_rate,0,1))
        elif k == "dep": y[k] = float(np.clip(0.25 + 0.5*flat, 0.0, 1.0))
        elif k == "sad": y[k] = float(np.clip(0.25 + 0.45*flat, 0.0, 1.0))
        elif k == "anx": y[k] = float(np.clip(0.2 + 0.2*(1.0-q_video), 0.0, 1.0))
        elif k == "stress": y[k] = float(np.clip(0.2 + 0.2*(1.0-q_video), 0.0, 1.0))
    return y
