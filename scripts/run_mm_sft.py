"""
Thin launcher for B-stage multimodal-prefix + LoRA SFT on AnnoMI.

Trains from the consolidated index built by build_mm_sft.py (all sessions in one
file), unlike run_full_loop.py which is per-session. Produces:
  <out>/mm_prefix.pt  +  <out>/lora_adapter/

  # multimodal (default)
  python scripts/run_mm_sft.py --base /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
      --index data/annomi/mm_sft/train.jsonl --out outputs/mm_sft/qwen7b_mm

  # text-only control (same steps/arch, alpha gated to 0)
  python scripts/run_mm_sft.py ... --out outputs/mm_sft/qwen7b_text --text_only
"""
from __future__ import annotations
import argparse
from mpse_mvp.mm.train_mm_sft import train_mm_sft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/annomi/mm_sft/train.jsonl")
    ap.add_argument("--base", required=True, help="base_model_dir (local path)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--k_audio", type=int, default=8)
    ap.add_argument("--k_video", type=int, default=8)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--no_aux_mu", action="store_true")
    ap.add_argument("--text_only", action="store_true")
    ap.add_argument("--state_prefix", action="store_true", help="inject evaluator mu as prefix soft tokens (route A->B)")
    ap.add_argument("--k_state", type=int, default=4)
    ap.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (for long seqs / history)")
    args = ap.parse_args()

    ckpt = train_mm_sft(
        index_jsonl=args.index,
        base_model_dir=args.base,
        out_dir=args.out,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_len=args.max_len,
        k_audio=args.k_audio,
        k_video=args.k_video,
        lora_r=args.lora_r,
        aux_mu=not args.no_aux_mu,
        text_only=args.text_only,
        use_state_prefix=args.state_prefix,
        k_state=args.k_state,
        gradient_checkpointing=args.grad_ckpt,
    )
    print("saved:", ckpt)


if __name__ == "__main__":
    main()
