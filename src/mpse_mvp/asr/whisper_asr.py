from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ASRSegment:
    t0: float
    t1: float
    text: str

def transcribe_whole(wav_path: str, model_dir: str, device: str = "cpu", compute_type: str = "int8"):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_dir, device=device, compute_type=compute_type, local_files_only=True)
    segments, _info = model.transcribe(wav_path, vad_filter=False)
    out = []
    for s in segments:
        out.append(ASRSegment(float(s.start), float(s.end), (s.text or "").strip()))
    return out

def gather_text(asr_segments: list[ASRSegment], t0: float, t1: float) -> str:
    texts = []
    for s in asr_segments:
        if s.t1 >= t0 and s.t0 <= t1:
            if s.text:
                texts.append(s.text)
    return " ".join(texts).strip()
