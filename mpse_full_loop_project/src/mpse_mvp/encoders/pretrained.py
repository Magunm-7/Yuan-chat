from __future__ import annotations
import numpy as np
import torch

class TextEncoder:
    def __init__(self, model_dir: str, device: str = "cpu"):
        from transformers import AutoTokenizer, AutoModel
        self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        inp = self.tok(texts, padding=True, truncation=True, return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}
        out = self.model(**inp)
        # mean pool
        h = out.last_hidden_state
        m = inp["attention_mask"].unsqueeze(-1)
        emb = (h*m).sum(dim=1) / (m.sum(dim=1) + 1e-8)
        return emb.detach().cpu().numpy().astype(np.float32)

class AudioEncoder:
    def __init__(self, model_dir: str, device: str = "cpu"):
        from transformers import AutoProcessor, AutoModel
        self.proc = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
        self.model.to(device); self.model.eval()
        self.device = device

    @torch.no_grad()
    def encode(self, wavs: list[np.ndarray], srs: list[int]) -> np.ndarray:
        # assumes same sr; if not, pre-resample outside
        inputs = self.proc(wavs, sampling_rate=srs[0], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k,v in inputs.items()}
        out = self.model(**inputs)
        h = out.last_hidden_state.mean(dim=1)
        return h.detach().cpu().numpy().astype(np.float32)

class VideoEncoder:
    def __init__(self, model_dir: str, device: str = "cpu"):
        from transformers import AutoProcessor, AutoModel
        self.proc = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
        self.model.to(device); self.model.eval()
        self.device = device

    @torch.no_grad()
    def encode(self, frames: list[np.ndarray]) -> np.ndarray:
        # frames: list of HxWx3 RGB np.uint8
        inputs = self.proc(images=frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k,v in inputs.items()}
        out = self.model(**inputs)
        # try pooled output if exists, else mean pool
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            h = out.pooler_output
        else:
            h = out.last_hidden_state.mean(dim=1)
        return h.detach().cpu().numpy().astype(np.float32)
