import numpy as np
import soundfile as sf

def load_wav(path: str):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    return x, sr
