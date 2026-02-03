from __future__ import annotations
import os, json
import numpy as np
from tqdm import tqdm

from mpse_mvp.segment.io import load_wav
from mpse_mvp.segment.vad import energy_vad_segments
from mpse_mvp.features.audio_features import audio_quality_and_prosody
from mpse_mvp.features.video_features import face_visible_and_microexpr
from mpse_mvp.features.text_features import text_quality, basic_text_feats
from mpse_mvp.supervision.agents import (
    text_rater_rule, audio_rater_heuristic, video_rater_heuristic, fuse_labels
)

def _split_longest_until(segs, target_turns, min_len=0.4):
    segs = list(segs)
    while len(segs) < target_turns and len(segs) > 0:
        k = max(range(len(segs)), key=lambda i: segs[i][1] - segs[i][0])
        t0, t1 = segs[k]
        if (t1 - t0) < 2 * min_len:
            break
        mid = (t0 + t1) / 2.0
        segs[k] = (t0, mid)
        segs.insert(k + 1, (mid, t1))
    return segs

def _postprocess_segs(segs, vad_cfg, total_secs):
    # 1) merge close segments (gap <= merge_gap)
    merge_gap = float(vad_cfg.get("merge_gap_sec", vad_cfg.get("min_silence_ms", 400) / 1000.0))
    segs = sorted(segs)
    merged = []
    for t0, t1 in segs:
        if not merged:
            merged.append([t0, t1])
        else:
            if t0 - merged[-1][1] <= merge_gap:
                merged[-1][1] = max(merged[-1][1], t1)
            else:
                merged.append([t0, t1])
    segs = [(float(a), float(b)) for a, b in merged]

    # 2) pad each seg (optional)
    pad = float(vad_cfg.get("pad_sec", 0.2))
    if pad > 0:
        segs = [(max(0.0, t0 - pad), min(float(total_secs), t1 + pad)) for t0, t1 in segs]

    # 3) match target_turns by splitting (no repeating last seg!)
    target_turns = int(vad_cfg.get("target_turns", 0)) or 0
    if target_turns > 0:
        if len(segs) > target_turns:
            segs = segs[:target_turns]
        elif len(segs) < target_turns:
            segs = _split_longest_until(segs, target_turns, min_len=float(vad_cfg.get("min_len_sec", 0.4)))

    return segs


def build_turns(session_id: str, mp4_path: str, wav_path: str, turns_path: str, target_turns: int,
                vad_cfg: dict, idx_names: list[str], use_asr: bool,
                asr_model_dir: str | None, asr_device: str = "cpu", asr_compute_type: str = "int8",
                use_llm_rater: bool = False, llm_cfg: dict | None = None):
    os.makedirs(os.path.dirname(turns_path), exist_ok=True)

    wav, sr = load_wav(wav_path)
    total_secs = len(wav) / sr

    # 1) VAD first
    segs = energy_vad_segments(
        wav, sr,
        frame_ms=vad_cfg.get("frame_ms", 30),
        thr=vad_cfg.get("thr", 0.02),
        min_speech_ms=vad_cfg.get("min_speech_ms", 250),
        min_silence_ms=vad_cfg.get("min_silence_ms", 400),
    )

    # 2) postprocess (merge/pad/split to target)
    # pass target_turns into vad_cfg so _postprocess_segs can use it
    vad_cfg = dict(vad_cfg)
    vad_cfg["target_turns"] = target_turns
    segs = _postprocess_segs(segs, vad_cfg, total_secs)

    if len(segs) == 0:
        raise RuntimeError("VAD produced 0 segments. Try lower thr.")


    # ASR whole audio once (optional)
    asr_segs = None
    if use_asr:
        if not asr_model_dir:
            raise ValueError("asr_model_dir is required when asr.enabled=true")
        from mpse_mvp.asr.whisper_asr import transcribe_whole
        asr_segs = transcribe_whole(wav_path, asr_model_dir, device=asr_device, compute_type=asr_compute_type)

    # optional LLM rater init
    llm = None
    if use_llm_rater:
        from mpse_mvp.supervision.llm_rater import load_llm
        tok, model = load_llm(llm_cfg["model_dir"], device=llm_cfg.get("device","cuda"))
        llm = (tok, model)

    rows = []
    for i,(t0,t1) in enumerate(segs, start=1):
        # audio slice
        s0 = int(t0*sr); s1 = int(t1*sr)
        wav_seg = wav[s0:s1]

        q_audio, stress_proxy = audio_quality_and_prosody(wav_seg, sr)
        q_video, micro_rate = face_visible_and_microexpr(mp4_path, t0, t1, sample_fps=5.0)

        asr_text = ""
        if use_asr and asr_segs is not None:
            from mpse_mvp.asr.whisper_asr import gather_text
            asr_text = gather_text(asr_segs, t0, t1)
        q_text = text_quality(asr_text)

        # labelers
        if use_llm_rater and llm is not None:
            from mpse_mvp.supervision.llm_rater import rate_text
            obj, _raw = rate_text(llm[0], llm[1], asr_text, max_new_tokens=llm_cfg.get("max_new_tokens",128))
            yT = obj if obj else text_rater_rule(asr_text, idx_names)
        else:
            yT = text_rater_rule(asr_text, idx_names)

        yA = audio_rater_heuristic(q_audio, stress_proxy, idx_names)
        yV = video_rater_heuristic(q_video, micro_rate, idx_names)

        y = fuse_labels(yT,yA,yV,q_text,q_audio,q_video,idx_names)

        rows.append({
            "session_id": session_id,
            "turn_id": i,
            "t0": float(t0), "t1": float(t1),
            "asr_text": asr_text if asr_text else "(ASR_EMPTY)",
            "q_text": float(q_text), "q_audio": float(q_audio), "q_video": float(q_video),
            "microexpr_rate": float(micro_rate),
            "y_soft": y,
        })

    with open(turns_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return turns_path
