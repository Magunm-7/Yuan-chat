"""
Run the FULL closed-loop:
0) (optional) extract wav from mp4
1) VAD segment -> turns.jsonl (dialogue mode: turns_user.jsonl + turns_all.jsonl + pairs.jsonl)
2) build MPSE trainset npz
3) train MPSE -> mpse.pt
4) upgrade turns -> mu/sigma/alpha/p_ok/weight
5) build SFT jsonl (old: teacher targets; dialogue: from pairs.jsonl)
6) build multimodal cache (whisper+clip pooled embeddings)
7) train multimodal prefix SFT (freeze LLaMA, train projectors)

Usage:
  python -m pip install -e .
  python scripts/run_full_loop.py --config configs/default.yaml
  python scripts/run_full_loop.py --config configs/default.yaml --session_id S0002
"""
import argparse
import os
import numpy as np

from mpse_mvp.utils import load_yaml, fmt_path, ensure_dir
from mpse_mvp.segment.extract_audio import extract_wav_from_mp4
from mpse_mvp.pipeline.build_turns import build_turns
from mpse_mvp.pipeline.build_mpse_trainset import build_npz
from mpse_mvp.mpse.train import train_mpse
from mpse_mvp.upgrade.upgrade import upgrade
from mpse_mvp.sft.teacher_generate import generate_teacher_sft
from mpse_mvp.mm.cache_builder import build_mm_cache
from mpse_mvp.mm.train_mm_sft import train_mm_sft


def _require_exists(path: str, what: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[pipeline] Missing required {what}: {path}\n"
            f"-> You likely skipped the stage that generates it."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    # NEW: allow overriding session id for batch runs
    ap.add_argument("--session_id", default=None, help="Override cfg['session']['id'], e.g., S0001")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    sid = args.session_id or cfg["session"]["id"]
    raw_mp4 = fmt_path(cfg["paths"]["raw_video"], session_id=sid)
    work_dir = fmt_path(cfg["paths"]["work_dir"], session_id=sid)
    outputs_dir = cfg["paths"]["outputs_dir"]

    wav_path = os.path.join(work_dir, f"audio_{cfg['segment']['wav_sr']//1000}k.wav")
    turns_path = os.path.join(work_dir, "turns.jsonl")

    ensure_dir(work_dir)
    ensure_dir(outputs_dir)

    # dialogue outputs (only meaningful when cfg["dialogue"]["enabled"]=true)
    turns_all_path = os.path.join(work_dir, "turns_all.jsonl")
    turns_user_path = os.path.join(work_dir, "turns_user.jsonl")
    pairs_path = os.path.join(work_dir, "turns_pairs.jsonl")

    dialogue_cfg = cfg.get("dialogue", {}) or {}
    dlg_enabled = bool(dialogue_cfg.get("enabled", False))

    # ---------------------------
    # TRAIN-ONLY shortcut:
    # If pipeline requests ONLY mm_sft training, skip requiring mp4/wav/turns
    # and directly train from outputs/mm_cache/<sid>/mm_index.jsonl
    # ---------------------------
    pipe = cfg.get("pipeline", {}) or {}
    only_train = (
        bool(pipe.get("run_train_mm_sft", False))
        and (not bool(pipe.get("run_extract_wav", True)))
        and (not bool(pipe.get("run_build_turns", True)))
        and (not bool(pipe.get("run_build_mpse_npz", True)))
        and (not bool(pipe.get("run_train_mpse", True)))
        and (not bool(pipe.get("run_upgrade", True)))
        and (not bool(pipe.get("run_build_sft", True)))
        and (not bool(pipe.get("run_mm_cache", True)))
    )

    if only_train:
        mm_out = os.path.join(outputs_dir, "mm_cache", sid)
        index_jsonl = os.path.join(mm_out, "mm_index.jsonl")
        if not os.path.exists(index_jsonl):
            raise FileNotFoundError(
                f"[train-only] Missing mm_index.jsonl: {index_jsonl}\n"
                f"-> Make sure you ran merge_mm and the folder name matches session.id exactly (case-sensitive)."
            )

        print("[train-only] Using mm index:", index_jsonl)
        if not cfg.get("mm_sft", {}).get("enabled", True):
            raise RuntimeError("[train-only] mm_sft.enabled=false, cannot train")

        ckpt2 = train_mm_sft(
            index_jsonl=index_jsonl,
            base_model_dir=cfg["mm_sft"]["base_model_dir"],
            out_dir=fmt_path(cfg["mm_sft"]["out_dir"], session_id=sid),
            batch_size=int(cfg["mm_sft"].get("batch_size", 1)),
            lr=float(cfg["mm_sft"].get("lr", 2e-4)),
            epochs=int(cfg["mm_sft"].get("epochs", 1)),
            k_audio=int(cfg["mm"].get("k_audio", 8)),
            k_video=int(cfg["mm"].get("k_video", 8)),
            device=cfg["mm_sft"].get("device", "cuda"),
            max_len=int(cfg["mm_sft"].get("max_len", 1024)),
        )
        print("mm sft ckpt:", ckpt2)
        return


    # ---------------------------
    # Pipeline stage switches
    # ---------------------------
    pipe = cfg.get("pipeline", {}) or {}
    run_extract_wav = bool(pipe.get("run_extract_wav", True))
    run_build_turns = bool(pipe.get("run_build_turns", True))
    run_build_mpse_npz = bool(pipe.get("run_build_mpse_npz", True))
    run_train_mpse = bool(pipe.get("run_train_mpse", True))
    run_upgrade = bool(pipe.get("run_upgrade", True))
    run_build_sft = bool(pipe.get("run_build_sft", True))
    run_mm_cache = bool(pipe.get("run_mm_cache", True))
    run_train_mm_sft = bool(pipe.get("run_train_mm_sft", True))
    upgraded_policy = pipe.get("upgraded_policy", "overwrite")  # overwrite | skip_if_exists

    # Pre-compute all canonical output paths (so skipping still works)
    mpse_dir = os.path.join(outputs_dir, "mpse", sid)
    ensure_dir(mpse_dir)
    npz_path = os.path.join(mpse_dir, "train.npz")
    meta_path = os.path.join(mpse_dir, "meta.json")

    up_dir = os.path.join(outputs_dir, "upgrade", sid)
    ensure_dir(up_dir)
    up_path = os.path.join(up_dir, "turns_upgraded.jsonl")

    sft_dir = os.path.join(outputs_dir, "sft", sid)
    ensure_dir(sft_dir)
    sft_path = os.path.join(sft_dir, "sft_train.jsonl")

    mm_out = os.path.join(outputs_dir, "mm_cache", sid)
    ensure_dir(mm_out)

    # ---------------------------
    # [0] Extract WAV
    # ---------------------------
    if run_extract_wav:
        if not os.path.exists(wav_path):
            print("[0] Extract wav...")
            extract_wav_from_mp4(raw_mp4, wav_path, sr=cfg["segment"]["wav_sr"], ffmpeg_path=cfg.get("ffmpeg_path"))
        else:
            print("[0] Found wav:", wav_path)
    else:
        print("[0] Skip extract wav (pipeline.run_extract_wav=false)")
        _require_exists(wav_path, "wav")

    # ---------------------------
    # [1] Build turns
    # ---------------------------
    if run_build_turns:
        print("[1] Build turns ...")
        turns_used = build_turns(
            session_id=sid,
            mp4_path=raw_mp4,
            wav_path=wav_path,
            turns_path=turns_path,
            target_turns=cfg["segment"]["target_turns"],
            vad_cfg=cfg["segment"]["vad"],
            idx_names=cfg["indices"]["names"],
            use_asr=cfg["asr"]["enabled"],
            asr_model_dir=cfg["asr"]["model_dir"],
            asr_device=cfg["asr"].get("device", "cpu"),
            asr_compute_type=cfg["asr"].get("compute_type", "int8"),
            use_llm_rater=cfg.get("llm_rater", {}).get("enabled", False),
            llm_cfg=cfg.get("llm_rater", {}).get("llm_cfg", None),

            # dialogue-mode support
            dialogue_cfg=dialogue_cfg,
            turns_all_path=turns_all_path,
            pairs_path=pairs_path,
            turns_user_path=turns_user_path,
        )

        # IMPORTANT: downstream must use the actually-produced turns
        turns_path = turns_used

        print("turns used:", turns_path)
        if dlg_enabled:
            print("turns(all):", turns_all_path)
            print("pairs:", pairs_path)
    else:
        print("[1] Skip build turns (pipeline.run_build_turns=false)")
        _require_exists(turns_path, "turns.jsonl")
        if dlg_enabled:
            # dialogue mode expects these too
            _require_exists(turns_all_path, "turns_all.jsonl (dialogue)")
            _require_exists(pairs_path, "turns_pairs.jsonl (dialogue)")
            _require_exists(turns_user_path, "turns_user.jsonl (dialogue)")

    # ---------------------------
    # [2] Build MPSE trainset
    # ---------------------------
    in_dim = None
    if run_build_mpse_npz:
        print("[2] Build MPSE trainset ...")
        use_pretrained = cfg["mpse"].get("use_pretrained_encoders", False)
        enc_cfg = cfg["mpse"].get("encoders", {})

        print("use_pretrained =", use_pretrained)
        print("enc_cfg keys =", list(enc_cfg.keys()))
        print("enc_cfg =", enc_cfg)
        print("cfg mpse encoders =", cfg.get("mpse", {}).get("encoders", None))

        npz_path, in_dim = build_npz(
            turns_path=turns_path,
            out_npz=npz_path,
            idx_names=cfg["indices"]["names"],
            use_pretrained=use_pretrained,
            enc_cfg=enc_cfg,
        )
        print("npz:", npz_path)
    else:
        print("[2] Skip build MPSE trainset (pipeline.run_build_mpse_npz=false)")
        _require_exists(npz_path, "train.npz")

    # ---------------------------
    # [3] Train MPSE
    # ---------------------------
    ckpt = os.path.join(mpse_dir, "mpse.pt")
    if run_train_mpse:
        print("[3] Train MPSE ...")
        ckpt = train_mpse(
            npz_path=npz_path,
            out_dir=mpse_dir,
            epochs=cfg["mpse"]["epochs"],
            batch_size=cfg["mpse"]["batch_size"],
            lr=cfg["mpse"]["lr"],
            hidden_dim=cfg["mpse"]["hidden_dim"],
            dropout=cfg["mpse"]["dropout"],
            device=cfg["mpse"].get("device", "cpu"),
        )
        meta_path = os.path.join(mpse_dir, "meta.json")
        print("mpse ckpt:", ckpt)
    else:
        print("[3] Skip train MPSE (pipeline.run_train_mpse=false)")
        _require_exists(ckpt, "mpse.pt")
        _require_exists(meta_path, "meta.json")

    # ---------------------------
    # [4] Upgrade
    # ---------------------------
    print("[4] Upgrade dataset ...")
    if run_upgrade:
        if upgraded_policy == "skip_if_exists" and os.path.exists(up_path):
            print(f"[4] Skip upgrade: exists and upgraded_policy=skip_if_exists -> {up_path}")
        else:
            _require_exists(npz_path, "train.npz (for upgrade)")
            X = np.load(npz_path, allow_pickle=True)["X"]
            upgrade(
                turns_path=turns_path,
                X=X,
                ckpt=ckpt,
                meta_path=meta_path,
                idx_names=cfg["indices"]["names"],
                tau=cfg["indices"]["tau"],
                out_turns_path=up_path,
                sigma_lambda=cfg["upgrade"]["sigma_lambda"],
                sigma_max=cfg["upgrade"]["sigma_max"],
                inject_state_tokens=cfg["upgrade"]["inject_state_tokens"],
                device=cfg["mpse"].get("device", "cpu"),
            )
            print("upgraded turns:", up_path)
    else:
        print("[4] Skip upgrade (pipeline.run_upgrade=false)")

    # Regardless, downstream expects up_path to exist now
    _require_exists(up_path, "turns_upgraded.jsonl")

    # ---------------------------
    # [5] Build SFT jsonl
    # ---------------------------
    if run_build_sft:
        if dlg_enabled:
            # dialogue mode: use real assistant replies from pairs.jsonl
            print("[5] Build SFT from pairs ...")
            from mpse_mvp.sft.build_sft_from_pairs import build_sft_from_pairs
            build_sft_from_pairs(
                turns_upgraded_path=up_path,
                pairs_jsonl=pairs_path,
                out_jsonl=sft_path,
                inject_state_tokens=True,
            )
        else:
            # old mode: teacher targets or placeholder / manual therapist_reply
            if cfg.get("teacher", {}).get("enabled", True):
                print("[5] Teacher-generate SFT targets ...")
                generate_teacher_sft(
                    turns_upgraded_path=up_path,
                    out_jsonl=sft_path,
                    base_model_dir=cfg["teacher"]["base_model_dir"],
                    max_new_tokens=cfg["teacher"].get("max_new_tokens", 128),
                    device=cfg["teacher"].get("device", "cuda"),
                )
            else:
                from mpse_mvp.sft.build_sft import build_sft
                build_sft(up_path, sft_path, inject_state_tokens=True)

        print("sft jsonl:", sft_path)
    else:
        print("[5] Skip build SFT (pipeline.run_build_sft=false)")
        _require_exists(sft_path, "sft_train.jsonl")

    # ---------------------------
    # [6] Build MM cache
    # ---------------------------
    index_jsonl = None
    if run_mm_cache and cfg.get("mm", {}).get("enabled", True):
        print("[6] Build MM cache ...")
        index_jsonl = build_mm_cache(
            session_id=sid,
            mp4_path=raw_mp4,
            wav_path=wav_path,
            turns_upgraded_path=up_path,
            sft_jsonl=sft_path,
            out_dir=mm_out,
            whisper_dir=cfg["mm"]["whisper_dir"],
            clip_dir=cfg["mm"]["clip_dir"],
            idx_names=cfg["indices"]["names"],
            n_frames=int(cfg["mm"].get("n_frames", 8)),
            device=cfg["mm"].get("device", "cuda"),
        )
        print("mm index:", index_jsonl)
    else:
        if not run_mm_cache:
            print("[6] Skip MM cache (pipeline.run_mm_cache=false)")
        else:
            print("[6] Skip MM cache (mm.enabled=false)")
        # If skipping, try to reuse an existing index if present
        # (not required unless you also train MM-SFT)
        index_guess = os.path.join(mm_out, "mm_index.jsonl")
        if os.path.exists(index_guess):
            index_jsonl = index_guess
            print("[6] Reuse existing mm index:", index_jsonl)

    # ---------------------------
    # [7] Train MM SFT
    # ---------------------------
    if run_train_mm_sft and index_jsonl and cfg.get("mm_sft", {}).get("enabled", True):
        print("[7] Train MM-SFT ...")
        ckpt2 = train_mm_sft(
            index_jsonl=index_jsonl,
            base_model_dir=cfg["mm_sft"]["base_model_dir"],
            out_dir=fmt_path(cfg["mm_sft"]["out_dir"], session_id=sid),
            batch_size=int(cfg["mm_sft"].get("batch_size", 1)),
            lr=float(cfg["mm_sft"].get("lr", 2e-4)),
            epochs=int(cfg["mm_sft"].get("epochs", 1)),
            k_audio=int(cfg["mm"].get("k_audio", 8)),
            k_video=int(cfg["mm"].get("k_video", 8)),
            device=cfg["mm_sft"].get("device", "cuda"),
            max_len=int(cfg["mm_sft"].get("max_len", 1024)),
        )
        print("mm sft ckpt:", ckpt2)
    else:
        if not run_train_mm_sft:
            print("[7] Skip MM-SFT train (pipeline.run_train_mm_sft=false)")
        elif not index_jsonl:
            print("[7] Skip MM-SFT train (no mm index)")
        else:
            print("[7] Skip MM-SFT train (mm_sft.enabled=false)")

    print("\nDONE.\n")
    print("Key outputs:")
    print(f"- turns: {turns_path}")
    if dlg_enabled:
        print(f"- turns_all: {turns_all_path}")
        print(f"- pairs: {pairs_path}")
    print(f"- upgraded turns: {up_path}")
    print(f"- sft: {sft_path}")
    if index_jsonl:
        print(f"- mm cache index: {index_jsonl}")


if __name__ == "__main__":
    main()
