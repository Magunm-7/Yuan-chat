# scripts/merge_mm_index.py
import json, glob
from pathlib import Path

def merge(out_path: str, pattern: str = "outputs/mm_cache/S*/mm_index.jsonl", make_abs: bool = True):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paths = sorted(glob.glob(pattern))
    assert paths, f"No index found by pattern: {pattern}"

    with out_path.open("w", encoding="utf-8") as w:
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if make_abs:
                        rec["npz_path"] = str(Path(rec["npz_path"]).resolve())
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Merged {len(paths)} index files -> {out_path}")

if __name__ == "__main__":
    merge("outputs/mm_cache/ALL/mm_index.jsonl")
