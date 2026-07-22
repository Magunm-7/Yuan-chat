"""
Download AnnoMI source videos (YouTube/Vimeo) and extract 16 kHz mono wav.

AnnoMI ships only URLs (it can't redistribute video); we fetch them with yt-dlp.
Downloads at <=480p (CLIP wants 224px, face landmarks work fine at 480p -> saves
disk/bandwidth). Resumable: skips sessions whose wav already exists.

Run on the server (GPU-less "无卡模式" is fine — this is network/CPU bound):
  python scripts/download_annomi_videos.py                 # all live sessions
  python scripts/download_annomi_videos.py --limit 3       # smoke test first
"""
from __future__ import annotations
import os
import json
import glob
import argparse
import subprocess


def _run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stderr or "") + (p.stdout or "")


def download_video(url: str, out_base: str, max_height: int) -> str | None:
    """yt-dlp download to out_base.<ext> capped at max_height. Returns path or None."""
    existing = glob.glob(out_base + ".*")
    existing = [p for p in existing if not p.endswith(".part")]
    if existing:
        return existing[0]
    fmt = (f"bv*[height<={max_height}]+ba/"
           f"b[height<={max_height}]/b")
    code, log = _run([
        "yt-dlp", "--no-playlist", "--no-progress", "--retries", "3",
        "-f", fmt, "--merge-output-format", "mp4",
        "-o", out_base + ".%(ext)s", url,
    ])
    if code != 0:
        print(f"    yt-dlp failed: {log.strip().splitlines()[-1] if log.strip() else code}")
        return None
    hits = [p for p in glob.glob(out_base + ".*") if not p.endswith(".part")]
    return hits[0] if hits else None


def extract_wav(video_path: str, wav_path: str) -> bool:
    if os.path.exists(wav_path):
        return True
    code, log = _run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", wav_path,
    ])
    if code != 0:
        print(f"    ffmpeg failed: {log.strip().splitlines()[-1] if log.strip() else code}")
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description="Download AnnoMI videos + extract wav")
    ap.add_argument("--manifest", default="data/annomi/manifest.json")
    ap.add_argument("--video_dir", default="data/annomi/video")
    ap.add_argument("--wav_dir", default="data/annomi/wav")
    ap.add_argument("--report", default="data/annomi/download_report.json")
    ap.add_argument("--max_height", type=int, default=480)
    ap.add_argument("--limit", type=int, default=0, help="only first N live sessions (0 = all)")
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(args.wav_dir, exist_ok=True)

    live = [m for m in manifest if m.get("video_alive", True)]
    if args.limit:
        live = live[:args.limit]

    report = []
    ok = 0
    for i, m in enumerate(live, 1):
        sid = m["session_id"]
        url = m["video_url"]
        wav_path = os.path.join(args.wav_dir, f"{sid}.wav")
        print(f"[{i}/{len(live)}] session {sid}  {url}")
        if os.path.exists(wav_path):
            print("    wav exists, skip")
            report.append({"session_id": sid, "status": "cached"}); ok += 1
            continue

        vpath = download_video(url, os.path.join(args.video_dir, sid), args.max_height)
        if not vpath:
            report.append({"session_id": sid, "status": "download_failed"}); continue
        if not extract_wav(vpath, wav_path):
            report.append({"session_id": sid, "status": "wav_failed", "video": vpath}); continue

        report.append({"session_id": sid, "status": "ok", "video": vpath, "wav": wav_path})
        ok += 1

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    from collections import Counter
    print(f"\n=== done: {ok}/{len(live)} sessions have wav ===")
    print("status:", dict(Counter(r["status"] for r in report)))
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
