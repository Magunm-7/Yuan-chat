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

def build_turns(session_id: str, mp4_path: str, wav_path: str, turns_path: str, target_turns: int,
                vad_cfg: dict, idx_names: list[str], use_asr: bool,
                asr_model_dir: str | None, asr_device: str = "cpu", asr_compute_type: str = "int8",
                use_llm_rater: bool = False, llm_cfg: dict | None = None):
    os.makedirs(os.path.dirname(turns_path), exist_ok=True)

    wav, sr = load_wav(wav_path)
    \1
    total_secs = len(wav)/sr
    segs = _postprocess_segs(segs, vad_cfg, total_secs)
    if len(segs) == 0:
        raise RuntimeError("VAD produced 0 segments. Try lower thr.")
    if len(segs) > target_turns:
        segs = segs[:target_turns]
    if len(segs) < target_turns:
        # If fewer than target_turns, split longest segments to reach target (no duplication).
        segs = _split_longest_until(segs, target_turns, min_len_s=float(vad_cfg.get('min_len_s', 0.6)))

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
