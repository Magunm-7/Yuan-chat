import os, yaml
from dataclasses import dataclass

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def fmt_path(tpl: str, session_id: str) -> str:
    return tpl.format(session_id=session_id)

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p
