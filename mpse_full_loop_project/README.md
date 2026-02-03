# MPSE + Dataset Upgrading + SFT (MVP Project)

This repo implements the pipeline we agreed on:

0) Segment interview video into user turns (A1: single-speaker video, only user speaks).
1) Generate 3 types of weak supervision via agents/heuristics.
2) Train MPSE to predict (mu, sigma, alpha) per turn.
3) Upgrade dataset using MPSE outputs -> weighted SFT jsonl.
4) (Optional) run a lightweight text-only SFT baseline.

## Directory layout (default)
- data/raw/{session_id}.mp4
- data/derived/{session_id}/audio_16k.wav
- data/derived/{session_id}/turns.jsonl
- outputs/mpse/
- outputs/upgrade/
- outputs/sft/

## Quick start (single sample)
1) Put your video at: `data/raw/S0001.mp4`
2) Prepare env (miniforge recommended):
   - `conda env create -f environment.yml`
   - `conda activate mpse-mvp`
3) (Server needs ffmpeg only if you want mp4->wav on server)
   - If no ffmpeg on server: extract wav on your local machine and upload `audio_16k.wav` to `data/derived/S0001/`.
4) Run end-to-end:
   - `python scripts/run_all.py --session_id S0001`

## Offline model downloads (ModelScope)
- ASR (required if you want real text from audio):
  - faster-whisper-small (CT2) repository (you already found one on ModelScope)
  - download to local dir, set `asr.model_dir` in `configs/default.yaml`
- Text base LLM for labeler + (optional) SFT baseline:
  - default: Llama-3.2-3B-Instruct (ModelScope id example you provided)
  - set `llm.model_dir` in `configs/default.yaml`
- Encoders (optional but recommended):
  - text encoder: any local sentence-embedding model dir
  - audio encoder: any local wav2vec2/Hubert encoder dir
  - video encoder: any local CLIP/ViT image encoder dir
  (Search these on ModelScope, download with snapshot_download, and point to local dirs.)

## What you must edit first
Open `configs/default.yaml` and set:
- `paths.raw_video`
- `paths.work_dir`
- `asr.model_dir` (if using ASR)
- `llm.model_dir` (if using LLM labeler)
- encoder paths if you want pretrained embeddings



## Full closed-loop (includes multimodal SFT)

1) Download models via ModelScope (recommended for offline HF):
```bash
python scripts/download_models_ms.py --root models
```

2) Edit `configs/default.yaml` to point to your local model dirs:
- `asr.model_dir` (CT2 faster-whisper) or disable ASR
- `mm.whisper_dir` (HuggingFace-format whisper-small)
- `mm.clip_dir` (HuggingFace-format CLIP vision)
- `teacher.base_model_dir` and `mm_sft.base_model_dir` (Llama-3.2-3B-Instruct)

3) Install the project:
```bash
python -m pip install -e .
```

4) Run end-to-end:
```bash
python scripts/run_full_loop.py --config configs/default.yaml
```

### Notes
- This MVP uses **teacher-generated therapist responses** (self-distillation) to produce SFT targets. Replace with real therapist transcripts later.
- VAD parameters are under `segment.vad`:
  - `frame_ms`: frame size for energy computation (smaller -> more sensitive, noisier)
  - `thr`: energy threshold (lower -> more speech detected)
  - `min_speech_ms`: minimum speech duration for a segment
  - `min_silence_ms`: minimum silence to split segments (larger -> fewer, longer segments)
