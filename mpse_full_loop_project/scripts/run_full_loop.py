
"""
Run the FULL closed-loop:
0) (optional) extract wav from mp4
1) VAD segment -> turns.jsonl
2) build MPSE trainset npz
3) train MPSE -> mpse.pt
4) upgrade turns -> mu/sigma/alpha/p_ok/weight
5) build SFT jsonl (teacher-generated targets for MVP)
6) build multimodal cache (whisper+clip pooled embeddings)
7) train multimodal prefix SFT (freeze LLaMA, train projectors)

Usage:
  # install editable (or set PYTHONPATH)
  python -m pip install -e .
  python scripts/run_full_loop.py --config configs/default.yaml
"""
import argparse, os

from mpse_mvp.utils import load_yaml, fmt_path, ensure_dir
from mpse_mvp.segment.extract_audio import extract_wav_from_mp4
from mpse_mvp.pipeline.build_turns import build_turns
from mpse_mvp.pipeline.build_mpse_trainset import build_npz
from mpse_mvp.mpse.train import train_mpse
from mpse_mvp.upgrade.upgrade import upgrade_turns
from mpse_mvp.sft.teacher_generate import generate_teacher_sft
from mpse_mvp.mm.cache_builder import build_mm_cache
from mpse_mvp.mm.train_mm_sft import train_mm_sft

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    sid = cfg["session"]["id"]
    raw_mp4 = fmt_path(cfg["paths"]["raw_video"], session_id=sid)
    work_dir = fmt_path(cfg["paths"]["work_dir"], session_id=sid)
    outputs_dir = cfg["paths"]["outputs_dir"]

    wav_path = os.path.join(work_dir, f"audio_{cfg['segment']['wav_sr']//1000}k.wav")
    turns_path = os.path.join(work_dir, "turns.jsonl")

    ensure_dir(work_dir)
    ensure_dir(outputs_dir)

    # [0] Extract WAV if missing
    if not os.path.exists(wav_path):
        print("[0] Extract wav...")
        extract_wav_from_mp4(raw_mp4, wav_path, sr=cfg["segment"]["wav_sr"], ffmpeg_path=cfg.get("ffmpeg_path"))
    else:
        print("[0] Found wav:", wav_path)

    # [1] Build turns
    print("[1] Build turns ...")
    build_turns(
        session_id=sid,
        mp4_path=raw_mp4,
        wav_path=wav_path,
        turns_path=turns_path,
        target_turns=cfg["segment"]["target_turns"],
        vad_cfg=cfg["segment"]["vad"],
        idx_names=cfg["indices"]["names"],
        use_asr=cfg["asr"]["enabled"],
        asr_model_dir=cfg["asr"]["model_dir"],
        asr_device=cfg["asr"].get("device","cpu"),
        asr_compute_type=cfg["asr"].get("compute_type","int8"),
        use_llm_rater=cfg["llm_rater"]["enabled"],
        llm_cfg=cfg.get("llm_rater",{}),
    )
    print("turns saved:", turns_path)

    # [2] Build MPSE trainset
    print("[2] Build MPSE trainset ...")
    mpse_dir = os.path.join(outputs_dir, "mpse", sid)
    ensure_dir(mpse_dir)
    npz_path = os.path.join(mpse_dir, "train.npz")
    meta_path = os.path.join(mpse_dir, "meta.json")
    build_npz(
        turns_path=turns_path,
        out_npz=npz_path,
        idx_names=cfg["indices"]["names"],
        use_pretrained=cfg["mpse"]["use_pretrained_encoders"],
        enc_cfg=cfg.get("mpse_encoders", {}),
    )
    print("npz:", npz_path)

    # [3] Train MPSE
    print("[3] Train MPSE ...")
    ckpt = train_mpse(
        npz_path=npz_path,
        out_ckpt=os.path.join(mpse_dir, "mpse.pt"),
        out_meta=meta_path,
        idx_names=cfg["indices"]["names"],
        epochs=cfg["mpse"]["epochs"],
        lr=cfg["mpse"]["lr"],
        device=cfg["mpse"].get("device","cpu"),
    )
    print("mpse ckpt:", ckpt)

    # [4] Upgrade
    print("[4] Upgrade dataset ...")
    up_dir = os.path.join(outputs_dir, "upgrade", sid)
    ensure_dir(up_dir)
    up_path = os.path.join(up_dir, "turns_upgraded.jsonl")
    upgrade_turns(
        turns_path=turns_path,
        mpse_ckpt=ckpt,
        mpse_meta=meta_path,
        out_path=up_path,
        idx_names=cfg["indices"]["names"],
        soft_end_cfg=cfg["soft_end"],
        safety_cfg=cfg["safety"],
        device=cfg["mpse"].get("device","cpu"),
    )
    print("upgraded turns:", up_path)

    # [5] Build SFT jsonl (teacher targets)
    sft_dir = os.path.join(outputs_dir, "sft", sid)
    ensure_dir(sft_dir)
    sft_path = os.path.join(sft_dir, "sft_train.jsonl")
    if cfg.get("teacher",{}).get("enabled", True):
        print("[5] Teacher-generate SFT targets ...")
        generate_teacher_sft(
            turns_upgraded_path=up_path,
            out_jsonl=sft_path,
            base_model_dir=cfg["teacher"]["base_model_dir"],
            max_new_tokens=cfg["teacher"].get("max_new_tokens",128),
            device=cfg["teacher"].get("device","cuda"),
        )
    else:
        from mpse_mvp.sft.build_sft import build_sft
        build_sft(up_path, sft_path, inject_state_tokens=True)
    print("sft jsonl:", sft_path)

    # [6] Build MM cache
    if cfg.get("mm",{}).get("enabled", True):
        print("[6] Build MM cache ...")
        mm_out = os.path.join(outputs_dir, "mm_cache", sid)
        ensure_dir(mm_out)
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
            device=cfg["mm"].get("device","cuda"),
        )
        print("mm index:", index_jsonl)
    else:
        index_jsonl = None

    # [7] Train MM SFT
    if index_jsonl and cfg.get("mm_sft",{}).get("enabled", True):
        print("[7] Train MM-SFT ...")
        ckpt2 = train_mm_sft(
            index_jsonl=index_jsonl,
            base_model_dir=cfg["mm_sft"]["base_model_dir"],
            out_dir=fmt_path(cfg["mm_sft"]["out_dir"], session_id=sid),
            batch_size=int(cfg["mm_sft"].get("batch_size",1)),
            lr=float(cfg["mm_sft"].get("lr",2e-4)),
            epochs=int(cfg["mm_sft"].get("epochs",1)),
            k_audio=int(cfg["mm"].get("k_audio",8)),
            k_video=int(cfg["mm"].get("k_video",8)),
            device=cfg["mm_sft"].get("device","cuda"),
            max_len=int(cfg["mm_sft"].get("max_len",1024)),
        )
        print("mm sft ckpt:", ckpt2)

    print("\nDONE.\n")
    print("Key outputs:")
    print(f"- turns: {turns_path}")
    print(f"- upgraded turns: {up_path}")
    print(f"- sft: {sft_path}")
    if index_jsonl:
        print(f"- mm cache index: {index_jsonl}")

if __name__ == "__main__":
    main()
