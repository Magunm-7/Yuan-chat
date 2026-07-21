from __future__ import annotations
import os, json
import numpy as np
from tqdm import tqdm

from mpse_mvp.features.text_features import basic_text_feats

def build_npz(turns_path: str, out_npz: str, idx_names: list[str],
              use_pretrained: bool, enc_cfg: dict):
    rows = [json.loads(l) for l in open(turns_path, encoding="utf-8")]

    # Targets Y: indices in order
    Y = np.array([[r["y_soft"].get(k,0.5) for k in idx_names] for r in rows], dtype=np.float32)

    # Quality Q: combine
    Q = np.array([0.34*r["q_text"] + 0.33*r["q_audio"] + 0.33*r["q_video"] for r in rows], dtype=np.float32)

    feats = []
    if use_pretrained:
        # Local pretrained encoders (must exist)
        from mpse_mvp.encoders.pretrained import TextEncoder
        te = TextEncoder(enc_cfg["text_model_dir"], device=enc_cfg.get("device","cpu"))
        texts = [r["asr_text"] for r in rows]
        T = te.encode(texts)

        # For MVP: still include simple quality scalars; audio/video pretrained can be added later
        # We'll concatenate: text_emb + [q_text,q_audio,q_video,microexpr_rate]
        scal = np.array([[r["q_text"], r["q_audio"], r["q_video"], r["microexpr_rate"]] for r in rows], dtype=np.float32)
        X = np.concatenate([T, scal], axis=1).astype(np.float32)
    else:
        # Fallback: basic text feats + scalars
        tf = np.stack([basic_text_feats(r["asr_text"]) for r in rows], axis=0)
        scal = np.array([[r["q_text"], r["q_audio"], r["q_video"], r["microexpr_rate"]] for r in rows], dtype=np.float32)
        X = np.concatenate([tf, scal], axis=1).astype(np.float32)

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez(out_npz, X=X, Y=Y, Q=Q)
    return out_npz, X.shape[1]
