import argparse
import os
import sys
import urllib.request
from modelscope.hub.snapshot_download import snapshot_download


DEFAULTS = {
    "clip_id": "openai-mirror/clip-vit-base-patch32",
    "faster_whisper_id": "pengzhendong/faster-whisper-small",
    "llama_id": "LLM-Research/Llama-3.2-3B-Instruct",
    "qwen_id": "Qwen/Qwen3-8B",
    "whisper_id": "openai-mirror/whisper-small",
    "mediapipe_url": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    ),
}

FOLDERS = {
    "clip": "clip-vit-base-patch32",
    "faster_whisper": "faster_whisper",
    "llama": "llama3.2",
    "mediapipe": "mediapipe",
    "qwen": "Qwen3-8B",
    "whisper": "whisper-small",
}


def repo_root_from_scripts_dir() -> str:
    """
    This script is located at: <repo_root>/scripts/download_models_ms.py
    Therefore repo_root is:     <repo_root>/scripts/..
    """
    scripts_dir = os.path.dirname(os.path.abspath(__file__))  # <repo_root>/scripts
    return os.path.abspath(os.path.join(scripts_dir, ".."))   # <repo_root>


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _dir_nonempty(p: str) -> bool:
    return os.path.isdir(p) and any(os.scandir(p))


def _snapshot_to(model_id: str, out_dir: str):
    _ensure_dir(out_dir)
    try:
        local_dir = snapshot_download(
            model_id=model_id,
            local_dir=out_dir,
            local_dir_use_symlinks=False,
        )
        return local_dir
    except TypeError:
        return snapshot_download(model_id=model_id, cache_dir=out_dir)


def _download_url(url: str, dst_path: str):
    _ensure_dir(os.path.dirname(dst_path))
    tmp_path = dst_path + ".tmp"
    print(f"[mediapipe] downloading {url} -> {dst_path}")
    urllib.request.urlretrieve(url, tmp_path)
    os.replace(tmp_path, dst_path)
    print("[mediapipe] done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="Models root dir. Default: <repo_root>/models")

    ap.add_argument("--clip_id", default=DEFAULTS["clip_id"])
    ap.add_argument("--faster_whisper_id", default=DEFAULTS["faster_whisper_id"])
    ap.add_argument("--llama_id", default=DEFAULTS["llama_id"])
    ap.add_argument("--qwen_id", default=DEFAULTS["qwen_id"])
    ap.add_argument("--whisper_id", default=DEFAULTS["whisper_id"])

    ap.add_argument("--mediapipe_url", default=DEFAULTS["mediapipe_url"])
    ap.add_argument("--skip_mediapipe", action="store_true")

    ap.add_argument("--skip_if_exists", action="store_true")
    ap.add_argument("--skip_llm", action="store_true")
    ap.add_argument("--skip_encoders", action="store_true")

    args = ap.parse_args()

    repo_root = repo_root_from_scripts_dir()
    root = args.root or os.path.join(repo_root, "models")
    _ensure_dir(root)

    jobs = []
    if not args.skip_encoders:
        jobs += [
            ("clip", args.clip_id, os.path.join(root, FOLDERS["clip"])),
            ("whisper", args.whisper_id, os.path.join(root, FOLDERS["whisper"])),
            ("faster_whisper", args.faster_whisper_id, os.path.join(root, FOLDERS["faster_whisper"])),
        ]
    if not args.skip_llm:
        jobs += [
            ("qwen", args.qwen_id, os.path.join(root, FOLDERS["qwen"])),
            ("llama", args.llama_id, os.path.join(root, FOLDERS["llama"])),
        ]

    for name, mid, out in jobs:
        if args.skip_if_exists and _dir_nonempty(out):
            print(f"[{name}] skip (exists): {out}")
            continue
        print(f"[{name}] downloading {mid} -> {out}")
        _snapshot_to(mid, out)
        print(f"[{name}] done")

    if not args.skip_mediapipe:
        dst = os.path.join(root, FOLDERS["mediapipe"], "face_landmarker.task")
        if args.skip_if_exists and os.path.exists(dst):
            print(f"[mediapipe] skip (exists): {dst}")
        else:
            _download_url(args.mediapipe_url, dst)

    print("\nAll done ✅")
    print("Models root:", root)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
