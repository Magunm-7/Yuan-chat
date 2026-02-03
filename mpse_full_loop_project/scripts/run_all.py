import argparse, os, json
import numpy as np

from mpse_mvp.utils import load_yaml, fmt_path, ensure_dir
from mpse_mvp.segment.extract_audio import extract_wav_from_mp4
from mpse_mvp.pipeline.build_turns import build_turns
from mpse_mvp.pipeline.build_mpse_trainset import build_npz
from mpse_mvp.mpse.train import train_mpse
from mpse_mvp.upgrade.upgrade import upgrade
from mpse_mvp.sft.build_sft import build_sft

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--session_id", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    sid = args.session_id or cfg["session"]["id"]

    raw_mp4 = fmt_path(cfg["paths"]["raw_video"], sid)
    work_dir = fmt_path(cfg["paths"]["work_dir"], sid)
    ensure_dir(work_dir)

    wav_path = os.path.join(work_dir, "audio_16k.wav")
    turns_path = os.path.join(work_dir, "turns.jsonl")

    # Step 0: ensure wav
    if not os.path.exists(wav_path):
        if os.path.exists(raw_mp4):
            print("[0] Extract wav from mp4 ...")
            extract_wav_from_mp4(raw_mp4, wav_path, sr=cfg["segment"]["wav_sr"])
        else:
            raise FileNotFoundError(f"Missing both wav and mp4. Expected mp4 at: {raw_mp4} or wav at: {wav_path}")
    else:
        print("[0] Found wav:", wav_path)

    # Step 1: build turns with weak labels
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
        asr_model_dir=cfg["asr"]["model_dir"] if cfg["asr"]["enabled"] else None,
        asr_device=cfg["asr"]["device"],
        asr_compute_type=cfg["asr"]["compute_type"],
        use_llm_rater=cfg["agents"]["use_llm_text_rater"],
        llm_cfg=cfg["llm"],
    )
    print("turns saved:", turns_path)

    # Step 2: build mpse train npz (features X + soft labels Y)
    print("[2] Build MPSE trainset ...")
    npz_path = os.path.join(cfg["paths"]["outputs_dir"], "mpse", sid, "train.npz")
    out_dir = os.path.join(cfg["paths"]["outputs_dir"], "mpse", sid)
    ensure_dir(out_dir)

    out_npz, in_dim = build_npz(
        turns_path, npz_path, idx_names=cfg["indices"]["names"],
        use_pretrained=cfg["mpse"]["use_pretrained_encoders"],
        enc_cfg=cfg["mpse"]["encoders"]
    )
    print("npz:", out_npz, "in_dim:", in_dim)

    # Step 3: train mpse
    print("[3] Train MPSE ...")
    ckpt = train_mpse(
        npz_path=out_npz,
        out_dir=out_dir,
        epochs=cfg["mpse"]["epochs"],
        batch_size=cfg["mpse"]["batch_size"],
        lr=float(cfg["mpse"]["lr"]),
        hidden_dim=cfg["mpse"]["hidden_dim"],
        dropout=cfg["mpse"]["dropout"],
        device="cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES") is not None) else "cpu"
    )
    meta_path = os.path.join(out_dir, "meta.json")
    print("mpse ckpt:", ckpt)

    # Step 4: upgrade dataset
    print("[4] Upgrade dataset ...")
    d = np.load(out_npz, allow_pickle=True)
    X = d["X"].astype(np.float32)
    up_dir = os.path.join(cfg["paths"]["outputs_dir"], "upgrade", sid)
    ensure_dir(up_dir)
    turns_up = os.path.join(up_dir, "turns_upgraded.jsonl")

    upgrade(
        turns_path=turns_path,
        X=X,
        ckpt=ckpt,
        meta_path=meta_path,
        idx_names=cfg["indices"]["names"],
        tau=cfg["indices"]["tau"],
        out_turns_path=turns_up,
        sigma_lambda=float(cfg["upgrade"]["sigma_lambda"]),
        sigma_max=float(cfg["upgrade"]["sigma_max"]),
        inject_state_tokens=bool(cfg["upgrade"]["inject_state_tokens"]),
        device="cpu"
    )
    print("upgraded turns:", turns_up)

    # Step 5: build SFT jsonl
    print("[5] Build SFT jsonl ...")
    sft_dir = os.path.join(cfg["paths"]["outputs_dir"], "sft", sid)
    ensure_dir(sft_dir)
    sft_jsonl = os.path.join(sft_dir, "sft_train.jsonl")
    build_sft(turns_up, sft_jsonl, inject_state_tokens=bool(cfg["upgrade"]["inject_state_tokens"]))
    print("sft jsonl:", sft_jsonl)

    print("\nDONE. Next:")
    print("- Inspect outputs/upgrade/{sid}/turns_upgraded.jsonl (mu/sigma/alpha/weight)")
    print("- Replace therapist placeholders in SFT when you have therapist turns.")
    print("- If ASR is noisy, adjust VAD thresholds in configs/default.yaml.")

if __name__ == "__main__":
    main()
