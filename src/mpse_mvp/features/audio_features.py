import numpy as np
import librosa

def audio_quality_and_prosody(wav: np.ndarray, sr: int):
    """Return q_audio in [0,1] and a stress proxy in [0,1]."""
    wav = wav.astype(np.float32)
    wav = wav / (np.max(np.abs(wav)) + 1e-8)

    # SNR-ish proxy: ratio of RMS to median absolute deviation
    rms = float(np.sqrt(np.mean(wav**2)) + 1e-8)
    mad = float(np.median(np.abs(wav - np.median(wav))) + 1e-8)
    snr_proxy = rms / mad
    q_audio = float(np.clip((snr_proxy - 1.0) / 8.0, 0.0, 1.0))  # heuristic

    # stress proxy: spectral centroid normalized
    centroid = librosa.feature.spectral_centroid(y=wav, sr=sr)
    c = float(np.mean(centroid))
    stress_proxy = float(np.clip((c - 1500.0) / 2500.0, 0.0, 1.0))
    return q_audio, stress_proxy
