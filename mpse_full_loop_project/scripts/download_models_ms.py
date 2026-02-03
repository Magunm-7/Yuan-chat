
"""
Download required models via ModelScope.

Default model_ids (ModelScope):
- Llama: LLM-Research/Llama-3.2-3B-Instruct
- Whisper: openai-mirror/whisper-small
- CLIP: openai-mirror/clip-vit-base-patch32

Usage:
  python scripts/download_models_ms.py --root models
"""
import argparse, os
from modelscope.hub.snapshot_download import snapshot_download

DEFAULTS = {
    "llama": "LLM-Research/Llama-3.2-3B-Instruct",
    "whisper": "openai-mirror/whisper-small",
    "clip": "openai-mirror/clip-vit-base-patch32",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="models")
    ap.add_argument("--llama_id", default=DEFAULTS["llama"])
    ap.add_argument("--whisper_id", default=DEFAULTS["whisper"])
    ap.add_argument("--clip_id", default=DEFAULTS["clip"])
    args = ap.parse_args()

    os.makedirs(args.root, exist_ok=True)
    for name, mid in [("llama", args.llama_id), ("whisper", args.whisper_id), ("clip", args.clip_id)]:
        out = os.path.join(args.root, os.path.basename(mid))
        print(f"[{name}] downloading {mid} -> {out}")
        local_dir = snapshot_download(model_id=mid, cache_dir=out)
        print(f"[{name}] done: {local_dir}")

if __name__ == "__main__":
    main()
