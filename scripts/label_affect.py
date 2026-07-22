"""
Strong affect weak labels: aro (audio arousal) + val (client-face valence).

aro : audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim -> arousal(0..1) from
      the client's audio slice (audio is already client-attributed by the turn).

val : the video is NOT speaker-attributed, so we must pick the CLIENT's face. During
      a client turn the client is the one talking, so among faces in the window we
      pick the actively-speaking one (highest lip-motion), crop it, and read valence
      from a facial-expression model. If no talking face is found, val is unreliable
      (client_face=0) and we fall back to neutral 0.5.

Writes turns_labeled.jsonl (aro_weak, val_weak, q_face, client_face) and reports
dynamic range + the client-face-found rate (how trustworthy val is on this data).

  python scripts/label_affect.py --limit 2   # smoke first
"""
from __future__ import annotations
import os, json, argparse
import numpy as np

AUD_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
FER_MODEL = "trpakov/vit-face-expression"
# 7-class FER -> valence in [-1,1]
VAL_MAP = {"happy": 1.0, "surprise": 0.4, "neutral": 0.0,
           "sad": -0.6, "fear": -0.6, "angry": -0.7, "disgust": -0.6}


# ---- audio arousal (audeering custom regression head) ----
def load_audio_model(dev):
    import torch.nn as nn
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class RegressionHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        def forward(self, x):
            x = self.dropout(x); x = torch.tanh(self.dense(x))
            x = self.dropout(x); return self.out_proj(x)

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.init_weights()
        def forward(self, x):
            h = self.wav2vec2(x)[0].mean(1)
            return self.classifier(h)  # (B,3) = arousal, dominance, valence

    import torch
    from transformers import AutoConfig
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    proc = Wav2Vec2Processor.from_pretrained(AUD_MODEL)
    model = EmotionModel(AutoConfig.from_pretrained(AUD_MODEL))
    sd = load_file(hf_hub_download(AUD_MODEL, "model.safetensors"))
    # remap old weight_norm (weight_g/weight_v) -> new parametrizations format
    new = {}
    for k, v in sd.items():
        if k.endswith("pos_conv_embed.conv.weight_g"):
            k = k.replace("weight_g", "parametrizations.weight.original0")
        elif k.endswith("pos_conv_embed.conv.weight_v"):
            k = k.replace("weight_v", "parametrizations.weight.original1")
        new[k] = v
    missing, unexpected = model.load_state_dict(new, strict=False)
    crit = [m for m in missing if "pos_conv" in m or "classifier" in m]
    print(f"[audio] remapped load; critical-missing={crit}", flush=True)
    return proc, model.to(dev).eval()


def arousal(proc, model, wav, sr, dev):
    import torch
    if len(wav) < sr // 10:
        return 0.5
    x = proc(wav, sampling_rate=sr, return_tensors="pt").input_values.to(dev)
    with torch.no_grad():
        out = model(x)[0].cpu().numpy()   # [arousal, dominance, valence]
    return float(np.clip(out[0], 0, 1))


# ---- video: pick the speaking (client) face, read valence ----
def faces_in_frame(frame, face_mesh):
    res = face_mesh.process(frame)
    out = []
    if res.multi_face_landmarks:
        H, W = frame.shape[:2]
        for lm in res.multi_face_landmarks:
            p = lm.landmark
            xs = [q.x for q in p]; ys = [q.y for q in p]
            out.append({
                "cx": float(np.mean(xs)),
                "mouth": float(abs(p[13].y - p[14].y)),         # lip gap (normalized)
                "bbox": (min(xs) * W, min(ys) * H, max(xs) * W, max(ys) * H),
            })
    return out


def pick_speaker_valence(frames, face_mesh, fer_proc, fer_model, id2val, dev):
    """Track faces by x across frames, pick the one whose mouth moves most, read valence."""
    import torch
    from PIL import Image
    tracks = []  # each: {"cx":..., "mouths":[], "crops":[(frame,bbox)]}
    for fr in frames:
        for f in faces_in_frame(fr, face_mesh):
            # assign to nearest existing track by cx, else new
            hit = None
            for tk in tracks:
                if abs(tk["cx"] - f["cx"]) < 0.15:
                    hit = tk; break
            if hit is None:
                hit = {"cx": f["cx"], "mouths": [], "crops": []}
                tracks.append(hit)
            hit["cx"] = 0.7 * hit["cx"] + 0.3 * f["cx"]
            hit["mouths"].append(f["mouth"])
            hit["crops"].append((fr, f["bbox"]))
    tracks = [t for t in tracks if len(t["mouths"]) >= 2]
    if not tracks:
        return 0.5, 0.0, 0        # neutral, q_face=0, client_face=0
    # speaker = most lip motion
    spk = max(tracks, key=lambda t: float(np.var(t["mouths"])))
    # if the 'speaker' barely moves and there are multiple faces, low confidence
    client_face = 1 if (np.var(spk["mouths"]) > 1e-5 or len(tracks) == 1) else 0

    vals = []
    for fr, (x0, y0, x1, y1) in spk["crops"]:
        m = 0.15 * (x1 - x0)
        x0, y0 = max(0, int(x0 - m)), max(0, int(y0 - m))
        x1, y1 = int(x1 + m), int(y1 + m)
        crop = fr[y0:y1, x0:x1]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            continue
        inp = fer_proc(images=Image.fromarray(crop), return_tensors="pt").to(dev)
        with torch.no_grad():
            p = fer_model(**inp).logits.softmax(-1)[0].cpu().numpy()
        v = sum(p[i] * id2val[i] for i in range(len(p)))   # [-1,1]
        vals.append((v + 1) / 2)                            # -> [0,1]
    if not vals:
        return 0.5, float(len(spk["crops"])) / max(1, len(frames)), 0
    q_face = float(len(spk["mouths"])) / max(1, len(frames))
    return float(np.mean(vals)), q_face, client_face


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default="data/annomi/turns_chg.jsonl")
    ap.add_argument("--video_dir", default="data/annomi/video")
    ap.add_argument("--wav_dir", default="data/annomi/wav")
    ap.add_argument("--out", default="data/annomi/turns_labeled.jsonl")
    ap.add_argument("--n_frames", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch, mediapipe as mp
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    from mpse_mvp.segment.io import load_wav
    from mpse_mvp.mm.encoders import sample_video_frames
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    aud_proc, aud_model = load_audio_model(dev)
    fer_proc = AutoImageProcessor.from_pretrained(FER_MODEL)
    fer_model = AutoModelForImageClassification.from_pretrained(FER_MODEL).to(dev).eval()
    id2label = fer_model.config.id2label
    id2val = {i: VAL_MAP.get(id2label[i].lower(), 0.0) for i in id2label}
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=3,
                                                min_detection_confidence=0.5)

    rows = [json.loads(l) for l in open(args.turns, encoding="utf-8")]
    by = {}
    for r in rows:
        by.setdefault(r["session_id"], []).append(r)
    sids = [s for s in by
            if os.path.exists(os.path.join(args.wav_dir, f"{s}.wav"))
            and os.path.exists(os.path.join(args.video_dir, f"{s}.mp4"))]
    sids.sort(key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
    if args.limit:
        sids = sids[:args.limit]

    out_rows = []
    for si, sid in enumerate(sids, 1):
        wav, sr = load_wav(os.path.join(args.wav_dir, f"{sid}.wav"))
        mp4 = os.path.join(args.video_dir, f"{sid}.mp4")
        print(f"[{si}/{len(sids)}] session {sid}: {len(by[sid])} turns", flush=True)
        for r in sorted(by[sid], key=lambda r: r["turn_id"]):
            t0, t1 = float(r["t0"]), float(r["t1"])
            r["aro_weak"] = arousal(aud_proc, aud_model, wav[int(t0 * sr):int(t1 * sr)].astype(np.float32), sr, dev)
            frames = sample_video_frames(mp4, t0, t1, n_frames=args.n_frames)
            r["val_weak"], r["q_face"], r["client_face"] = pick_speaker_valence(
                frames, face_mesh, fer_proc, fer_model, id2val, dev)
            out_rows.append(r)
    face_mesh.close()

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    aro = np.array([r["aro_weak"] for r in out_rows])
    val = np.array([r["val_weak"] for r in out_rows])
    cf = np.array([r["client_face"] for r in out_rows])
    print(f"\n=== {len(out_rows)} turns ===")
    print(f"  aro: mean {aro.mean():.3f} std {aro.std():.3f} [{aro.min():.2f},{aro.max():.2f}]")
    print(f"  val: mean {val.mean():.3f} std {val.std():.3f} [{val.min():.2f},{val.max():.2f}]")
    print(f"  client-face found: {100 * cf.mean():.1f}% of turns  <- val reliability")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
