
from __future__ import annotations
import os
import numpy as np
import torch

def _to_device(x, device: str):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, device=device)

class WhisperAudioEncoder:
    """
    Thin wrapper around transformers Whisper encoder to produce turn-level audio embeddings.
    Expected input: float32 mono waveform at 16kHz.
    Output: pooled embedding (B, C) and optionally token-level states (B, T, C).
    """
    def __init__(self, model_dir: str, device: str = "cpu", dtype: str = "float32"):
        from transformers import WhisperModel, WhisperFeatureExtractor
        self.device = device
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.feat = WhisperFeatureExtractor.from_pretrained(model_dir)
        self.model = WhisperModel.from_pretrained(model_dir)
        self.model.to(device)
        self.model.eval()
        # freeze
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, wav16k: np.ndarray, sr: int = 16000, return_sequence: bool = False):
        if sr != 16000:
            raise ValueError(f"WhisperAudioEncoder expects 16k audio, got sr={sr}")
        # WhisperFeatureExtractor expects float array in [-1,1]
        feats = self.feat(wav16k, sampling_rate=16000, return_tensors="pt")
        input_features = feats["input_features"].to(self.device)
        out = self.model.encoder(input_features=input_features)
        hs = out.last_hidden_state  # (B, T, C)
        pooled = hs.mean(dim=1)     # (B, C)
        if return_sequence:
            return pooled, hs
        return pooled, None

class CLIPVideoEncoder:
    """
    Wrapper around transformers CLIPVisionModel to encode a list of RGB frames.
    Input: frames as uint8 numpy array (N, H, W, 3) RGB.
    Output: pooled embedding (B, C) and optionally patch tokens.
    """
    def __init__(self, model_dir: str, device: str = "cpu", dtype: str = "float32"):
        from transformers import CLIPVisionModel, CLIPImageProcessor
        self.device = device
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.proc = CLIPImageProcessor.from_pretrained(model_dir)
        self.model = CLIPVisionModel.from_pretrained(model_dir)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, frames_rgb: np.ndarray, return_sequence: bool = False):
        # frames_rgb: (N,H,W,3) in RGB uint8
        if frames_rgb is None or len(frames_rgb) == 0:
            # return zeros to keep pipeline running
            pooled = torch.zeros((1, self.model.config.hidden_size), device=self.device)
            return pooled, None
        inputs = self.proc(images=[f for f in frames_rgb], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        out = self.model(pixel_values=pixel_values)
        hs = out.last_hidden_state  # (N, T, C) where T includes CLS + patches
        pooled = hs[:,0,:].mean(dim=0, keepdim=True)  # average CLS over frames -> (1,C)
        if return_sequence:
            return pooled, hs
        return pooled, None

def sample_video_frames(mp4_path: str, t0: float, t1: float, n_frames: int = 8) -> np.ndarray:
    """
    Sample n_frames evenly from [t0, t1] in a video. Returns RGB uint8 frames (N,H,W,3).
    Requires cv2.
    """
    import cv2
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps if total > 0 else None

    if duration is not None:
        t0 = max(0.0, min(t0, duration))
        t1 = max(0.0, min(t1, duration))
    if t1 <= t0:
        t1 = t0 + 0.1

    ts = np.linspace(t0, t1, num=n_frames, endpoint=False)
    frames = []
    for tt in ts:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(tt) * 1000.0)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    cap.release()
    if len(frames) == 0:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)
    return np.stack(frames, axis=0)
