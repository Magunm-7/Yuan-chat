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


def build_turns(
    session_id: str,
    mp4_path: str,
    wav_path: str,
    turns_path: str,
    target_turns: int,
    vad_cfg: dict,
    idx_names: list[str],
    use_asr: bool,
    asr_model_dir: str | None,
    asr_device: str = "cpu",
    asr_compute_type: str = "int8",
    use_llm_rater: bool = False,
    llm_cfg: dict | None = None,
    # --- NEW (optional): dialogue-mode support (robot+user alternating) ---
    dialogue_cfg: dict | None = None,
    turns_all_path: str | None = None,
    pairs_path: str | None = None,
    turns_user_path: str | None = None,
):
    """
    Backward compatible:
      - If dialogue_cfg is None or dialogue_cfg["enabled"] is False:
          behavior is unchanged; writes turns_path and returns turns_path.

    Dialogue mode (dialogue_cfg["enabled"]=True):
      - Treat VAD segments as alternating roles, starting from dialogue_cfg["start_role"] (default "assistant").
      - Export:
          * turns_all_path: all segments with role + ASR text (debug)
          * turns_user_path: only user segments with features/labels (for MPSE/upgrade)
          * pairs_path: pairs (user_i -> next assistant) for SFT
      - Do NOT force target_turns (turn count is unknown); no splitting to fixed length.
      - Strongly discourage merging/padding to avoid collapsing/overlapping user/assistant boundaries.
    """
    os.makedirs(os.path.dirname(turns_path), exist_ok=True)

    # --- dialogue config ---
    dlg = dialogue_cfg or {}
    dlg_enabled = bool(dlg.get("enabled", False))
    start_role = dlg.get("start_role", "assistant")  # robot starts by default
    keep_role = dlg.get("keep_role", "user")
    export_all = bool(dlg.get("export_all_turns", True))
    export_pairs = bool(dlg.get("export_pairs", True))
    target_user_turns = dlg.get("target_user_turns", None)  # None/-1/0 => no limit

    if dlg_enabled:
        # auto derive outputs if not provided
        base = turns_path
        if turns_all_path is None:
            turns_all_path = base.replace(".jsonl", "_all.jsonl") if base.endswith(".jsonl") else base + "_all.jsonl"
        if turns_user_path is None:
            turns_user_path = base.replace(".jsonl", "_user.jsonl") if base.endswith(".jsonl") else base + "_user.jsonl"
        if pairs_path is None:
            pairs_path = base.replace(".jsonl", "_pairs.jsonl") if base.endswith(".jsonl") else base + "_pairs.jsonl"

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

    # 2) postprocess (merge/pad/split)
    vad_cfg = dict(vad_cfg)

    if dlg_enabled:
        # turn count unknown => never split/truncate to target_turns
        vad_cfg["target_turns"] = 0
    else:
        # monologue mode: single speaker, single turn (collapse all VAD segs)
        force_single_turn = bool(vad_cfg.get("force_single_turn", False))
        if force_single_turn:
            # IMPORTANT: do not split/truncate segs by target_turns; we'll collapse later
            vad_cfg["target_turns"] = 0
        else:
            vad_cfg["target_turns"] = target_turns

    segs = _postprocess_segs(segs, vad_cfg, total_secs)

    # collapse to one turn if requested
    if (not dlg_enabled) and bool(vad_cfg.get("force_single_turn", False)):
        t0 = float(min(s[0] for s in segs))
        t1 = float(max(s[1] for s in segs))
        segs = [(t0, t1)]

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
        tok, model = load_llm(llm_cfg["model_dir"], device=llm_cfg.get("device", "cuda"))
        llm = (tok, model)

    def _toggle(role: str) -> str:
        return "user" if role == "assistant" else "assistant"

    # roles for each segment
    if dlg_enabled:
        roles = []
        cur = start_role  # e.g., "assistant"
        for _ in segs:
            roles.append(cur)
            cur = _toggle(cur)
    else:
        roles = ["user"] * len(segs)

    # --- build all_rows (role + ASR text only) ---
    all_rows = []
    for seg_id, ((t0, t1), role) in enumerate(zip(segs, roles), start=1):
        asr_text = ""
        if use_asr and asr_segs is not None:
            from mpse_mvp.asr.whisper_asr import gather_text
            asr_text = gather_text(asr_segs, t0, t1)
        all_rows.append({
            "session_id": session_id,
            "seg_id": seg_id,
            "role": role,
            "t0": float(t0), "t1": float(t1),
            "asr_text": asr_text if asr_text else "(ASR_EMPTY)",
        })

    # export turns_all.jsonl for debugging/inspection
    if dlg_enabled and export_all and turns_all_path:
        os.makedirs(os.path.dirname(turns_all_path), exist_ok=True)
        with open(turns_all_path, "w", encoding="utf-8") as f:
            for r in all_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- select user turns (for MPSE/upgrade), re-index turn_id ---
    if dlg_enabled:
        user_rows = [r for r in all_rows if r["role"] == keep_role]  # keep_role="user"

        if target_user_turns not in (None, -1, 0):
            user_rows = user_rows[: int(target_user_turns)]

        for tid, r in enumerate(user_rows, start=1):
            r["turn_id"] = tid
    else:
        user_rows = []
        for tid, r in enumerate(all_rows, start=1):
            r2 = dict(r)
            r2["turn_id"] = tid
            user_rows.append(r2)

    # --- compute features/labels ONLY on user_rows ---
    rows = []
    for r in tqdm(user_rows, desc="build turns"):
        t0 = float(r["t0"]); t1 = float(r["t1"])
        asr_text = r["asr_text"]

        # audio slice
        s0 = int(t0 * sr); s1 = int(t1 * sr)
        wav_seg = wav[s0:s1]

        q_audio, stress_proxy = audio_quality_and_prosody(wav_seg, sr)
        q_video, micro_rate = face_visible_and_microexpr(mp4_path, t0, t1, sample_fps=5.0)
        q_text = text_quality(asr_text)

        # labelers
        if use_llm_rater and llm is not None:
            from mpse_mvp.supervision.llm_rater import rate_text
            obj, _raw = rate_text(llm[0], llm[1], asr_text, max_new_tokens=llm_cfg.get("max_new_tokens", 128))
            yT = obj if obj else text_rater_rule(asr_text, idx_names)
        else:
            yT = text_rater_rule(asr_text, idx_names)

        yA = audio_rater_heuristic(q_audio, stress_proxy, idx_names)
        yV = video_rater_heuristic(q_video, micro_rate, idx_names)
        y = fuse_labels(yT, yA, yV, q_text, q_audio, q_video, idx_names)

        rows.append({
            "session_id": session_id,
            "turn_id": int(r["turn_id"]),
            "t0": float(t0), "t1": float(t1),
            "asr_text": asr_text if asr_text else "(ASR_EMPTY)",
            "q_text": float(q_text), "q_audio": float(q_audio), "q_video": float(q_video),
            "microexpr_rate": float(micro_rate),
            "y_soft": y,
        })

    # write turns (old mode -> turns_path; dialogue mode -> turns_user_path)
    out_turns = turns_user_path if dlg_enabled else turns_path
    os.makedirs(os.path.dirname(out_turns), exist_ok=True)
    with open(out_turns, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- build pairs.jsonl: (user_i -> next assistant) ---
    if dlg_enabled and export_pairs and pairs_path:
        os.makedirs(os.path.dirname(pairs_path), exist_ok=True)

        # seg_id -> turn_id for kept user turns
        seg2turn = {r["seg_id"]: r["turn_id"] for r in user_rows}

        pairs = []
        for i, u in enumerate(all_rows):
            if u["role"] != "user":
                continue
            if u["seg_id"] not in seg2turn:
                continue  # user was truncated/removed

            # find next assistant after this user seg
            j = i + 1
            while j < len(all_rows) and all_rows[j]["role"] != "assistant":
                j += 1
            if j >= len(all_rows):
                continue  # last user has no next assistant

            a = all_rows[j]
            pairs.append({
                "session_id": session_id,
                "user_turn_id": int(seg2turn[u["seg_id"]]),
                "user_seg_id": int(u["seg_id"]),
                "user_t0": float(u["t0"]), "user_t1": float(u["t1"]),
                "user_text": u["asr_text"],
                "assistant_seg_id": int(a["seg_id"]),
                "assistant_t0": float(a["t0"]), "assistant_t1": float(a["t1"]),
                "assistant_text": a["asr_text"],
            })

        with open(pairs_path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    return out_turns
