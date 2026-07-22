#!/usr/bin/env bash
# Smoke test: verify every moving part installs & runs on this server BEFORE
# committing to a longer billing term. Downloads the two encoders the real
# pipeline uses (whisper-small, clip-vit-base-patch32), so it also warms the cache.
#
# Run after scripts/setup_server.sh. A GPU instance is preferred (tests CUDA),
# but it also passes on CPU (just slower).
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src:${PYTHONPATH:-}

python - <<'PY'
import sys, numpy as np, torch

def ok(msg): print(f"  [ OK ] {msg}")
def bad(msg, e): print(f"  [FAIL] {msg}: {e}"); sys.exit(1)

print("1) torch / GPU")
print(f"     torch {torch.__version__}, cuda_available={torch.cuda.is_available()}")
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cuda":
    ok(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("     [warn] no CUDA — fine for CPU-only steps, but training needs GPU")

print("2) Whisper audio encoder (whisper-small)")
try:
    from transformers import WhisperModel, WhisperFeatureExtractor
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    wm = WhisperModel.from_pretrained("openai/whisper-small").to(dev).eval()
    wav = np.random.randn(16000 * 3).astype("float32")  # 3s dummy
    feats = fe(wav, sampling_rate=16000, return_tensors="pt").input_features.to(dev)
    with torch.no_grad():
        h = wm.encoder(feats).last_hidden_state
    ok(f"whisper encoder out {tuple(h.shape)}")
except Exception as e:
    bad("whisper", e)

print("3) CLIP vision encoder (clip-vit-base-patch32)")
try:
    from transformers import CLIPVisionModel, CLIPImageProcessor
    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    cm = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
    img = (np.random.rand(224, 224, 3) * 255).astype("uint8")
    px = proc(images=[img], return_tensors="pt").pixel_values.to(dev)
    with torch.no_grad():
        v = cm(pixel_values=px).last_hidden_state
    ok(f"clip vision out {tuple(v.shape)}")
except Exception as e:
    bad("clip", e)

print("4) MediaPipe face mesh")
try:
    import mediapipe as mp
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    fm.process((np.random.rand(256, 256, 3) * 255).astype("uint8"))  # no face -> None, but must run
    fm.close()
    ok("mediapipe ran")
except Exception as e:
    bad("mediapipe", e)

print("5) librosa prosody")
try:
    import librosa
    y = np.random.randn(16000).astype("float32")
    c = librosa.feature.spectral_centroid(y=y, sr=16000)
    ok(f"librosa centroid shape {c.shape}")
except Exception as e:
    bad("librosa", e)

print("6) project import + numpy eval self-test")
try:
    from mpse_mvp.eval.metrics import make_synthetic_predictions, spearman, talk_ordinal
    rows = make_synthetic_predictions()
    mu = np.array([r["mu"]["chg"] for r in rows])
    rho = spearman(mu, talk_ordinal([r["talk_type"] for r in rows]))
    assert rho > 0
    ok(f"mpse_mvp import + eval works (rho={rho:+.3f})")
except Exception as e:
    bad("project import", e)

print("\nALL SMOKE CHECKS PASSED")
PY

echo "yt-dlp:" && yt-dlp --version && echo "ffmpeg:" && ffmpeg -version | head -1
