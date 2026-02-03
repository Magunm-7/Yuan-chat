import os, subprocess, shutil

def extract_wav_from_mp4(mp4_path: str, wav_path: str, sr: int = 16000, ffmpeg_path: str | None = None):
    """Extract mono 16k wav from mp4. Requires ffmpeg executable."""
    exe = ffmpeg_path or shutil.which("ffmpeg")
    if exe is None:
        raise FileNotFoundError(
            "ffmpeg not found. Options:\n"
            "1) conda install ffmpeg (recommended)\n"
            "2) extract wav on your local machine and upload it\n"
            "3) pass ffmpeg_path to this function\n"
        )
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    cmd = [exe, "-y", "-i", mp4_path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", wav_path]
    subprocess.run(cmd, check=True)
