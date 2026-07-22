"""
AnnoMI -> turns.jsonl adapter.

AnnoMI ships expert-annotated MI counselling transcripts as a CSV. We turn it
into the per-client-turn representation the MPSE pipeline consumes, and attach
the gold labels used for evaluation (see docs/eval-design.md).

What replaces the old audio-only pipeline (docs/eval-design.md 2.1):
  - turn boundaries  <- `timestamp` (start of each utterance; end = next start)
  - speaker role     <- `interlocutor`
  - transcript text  <- `utterance_text`   (ASR becomes optional)
  - therapist reply  <- the therapist utterance(s) following a client turn

Only CLIENT turns become MPSE turns. Each carries:
  - talk_type       : gold external truth for the `chg` dimension
  - mi_quality      : session-level gold (high/low)
  - therapist_reply : concatenated following therapist utterance(s) (generator target)

Cleaning (docs/eval-design.md 2.2):
  - dur = next_utterance.start - this.start   (last utterance has no next -> dropped)
  - drop dur <= 0  (timestamp inversions)
  - drop dur < min_dur (default 2.0s: too short to slice audio/video)
"""
from __future__ import annotations
import os
import csv
import json
import argparse
from collections import Counter, defaultdict

# talk_type as an ordinal aligned with mu (lower = more change-oriented = "better")
TALK_SCORE = {"change": -1, "neutral": 0, "sustain": +1}

# Video URLs found dead in the 2026-07-22 liveness probe (notes.md 6.8).
# Sessions using these keep their text/gold labels but have no usable video.
KNOWN_DEAD_VIDEOS = {
    "https://www.youtube.com/watch?v=KNnwfMfkZek",
    "https://www.youtube.com/watch?v=mbheToXlm2Y",
}


def parse_ts(s: str) -> float | None:
    """'HH:MM:SS' (or 'MM:SS') -> seconds. Returns None if unparseable."""
    s = (s or "").strip()
    if not s:
        return None
    parts = s.split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h, m, sec = 0, parts[0], parts[1]
    elif len(parts) == 1:
        h, m, sec = 0, 0, parts[0]
    else:
        return None
    return float(h * 3600 + m * 60 + sec)


def load_sessions(csv_path: str) -> dict[str, list[dict]]:
    """Group utterances by transcript_id, ordered by utterance_id."""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    by = defaultdict(list)
    for r in rows:
        by[r["transcript_id"]].append(r)
    for tid in by:
        by[tid].sort(key=lambda r: int(r["utterance_id"]))
    return by


def _clean_beh(b: str) -> str | None:
    b = (b or "").strip()
    return None if b in ("", "n/a") else b


def build_client_turns(utts: list[dict], min_dur: float) -> tuple[list[dict], Counter]:
    """
    From one session's ordered utterances, produce cleaned client turns.
    Returns (kept_turns, drop_reasons).
    """
    n = len(utts)
    drops = Counter()
    kept = []
    turn_id = 0
    for i, u in enumerate(utts):
        if u["interlocutor"] != "client":
            continue
        tt = u["client_talk_type"]
        if tt not in TALK_SCORE:
            drops["no_talk_type"] += 1
            continue

        t0 = parse_ts(u["timestamp"])
        if i + 1 >= n:
            drops["last_utt_no_end"] += 1  # boundary artifact: no next start
            continue
        t1 = parse_ts(utts[i + 1]["timestamp"])
        if t0 is None or t1 is None:
            drops["bad_timestamp"] += 1
            continue
        dur = t1 - t0
        if dur < 0:
            drops["inverted_timestamp"] += 1  # genuinely corrupt ordering
            continue
        if dur == 0:
            drops["zero_dur_same_second"] += 1  # 1s-resolution artifact (recoverable later)
            continue
        if dur < min_dur:
            drops["too_short"] += 1
            continue

        # therapist reply = therapist utterance(s) between this client turn and the next client turn
        reply_texts, reply_behs = [], []
        for j in range(i + 1, n):
            if utts[j]["interlocutor"] == "client":
                break
            reply_texts.append(utts[j]["utterance_text"])
            b = _clean_beh(utts[j]["main_therapist_behaviour"])
            if b:
                reply_behs.append(b)
        reply_beh = Counter(reply_behs).most_common(1)[0][0] if reply_behs else None

        turn_id += 1
        kept.append({
            "session_id": u["transcript_id"],
            "mi_quality": u["mi_quality"],
            "turn_id": turn_id,
            "utterance_id": int(u["utterance_id"]),
            "t0": round(t0, 3), "t1": round(t1, 3), "dur": round(dur, 3),
            "text": u["utterance_text"],
            "talk_type": tt,
            "therapist_reply": " ".join(reply_texts).strip(),
            "reply_behaviour": reply_beh,
        })
    return kept, drops


def build(csv_path: str, out_jsonl: str, out_manifest: str,
          min_dur: float = 2.0, min_session_turns: int = 8) -> dict:
    """
    Full build. Writes:
      - out_jsonl    : one line per kept client turn (all sessions)
      - out_manifest : per-session summary (mi_quality, video_url, counts, video_alive)
    Prints and returns cleaning stats (the deliverable: usable turns / usable sessions).
    """
    sessions = load_sessions(csv_path)
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)

    all_turns = []
    manifest = []
    total_drops = Counter()
    for tid, utts in sessions.items():
        turns, drops = build_client_turns(utts, min_dur)
        total_drops.update(drops)
        vurl = utts[0]["video_url"]
        manifest.append({
            "session_id": tid,
            "mi_quality": utts[0]["mi_quality"],
            "video_url": vurl,
            "video_alive": vurl not in KNOWN_DEAD_VIDEOS,
            "n_client_turns_kept": len(turns),
            "usable_for_h2": len(turns) >= min_session_turns,
        })
        all_turns.extend(turns)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for t in all_turns:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ---- deliverable stats ----
    usable_sessions = [m for m in manifest if m["usable_for_h2"]]
    alive_sessions = [m for m in manifest if m["video_alive"]]
    q = Counter(m["mi_quality"] for m in manifest)
    q_usable = Counter(m["mi_quality"] for m in usable_sessions)
    tt = Counter(t["talk_type"] for t in all_turns)

    stats = {
        "sessions_total": len(manifest),
        "sessions_high": q["high"], "sessions_low": q["low"],
        "client_turns_kept": len(all_turns),
        "sessions_usable_for_h2": len(usable_sessions),
        "usable_high": q_usable["high"], "usable_low": q_usable["low"],
        "sessions_video_alive": len(alive_sessions),
        "talk_type_dist": dict(tt),
        "drops": dict(total_drops),
        "min_dur": min_dur, "min_session_turns": min_session_turns,
    }

    print("=== AnnoMI build ===")
    print(f"sessions:            {stats['sessions_total']}  (high {q['high']} / low {q['low']})")
    print(f"kept client turns:   {stats['client_turns_kept']}")
    print(f"  talk_type:         {dict(tt)}")
    print(f"dropped:             {dict(total_drops)}")
    print(f"sessions >= {min_session_turns} turns:  {len(usable_sessions)}"
          f"  (high {q_usable['high']} / low {q_usable['low']})   <- H2 sample")
    print(f"sessions video-alive:{len(alive_sessions)} / {len(manifest)}")
    print(f"wrote: {out_jsonl}")
    print(f"wrote: {out_manifest}")
    return stats


def main():
    ap = argparse.ArgumentParser(description="Build turns.jsonl from AnnoMI CSV")
    ap.add_argument("--csv", default="data/annomi/AnnoMI-simple.csv")
    ap.add_argument("--out_jsonl", default="data/annomi/turns_client.jsonl")
    ap.add_argument("--out_manifest", default="data/annomi/manifest.json")
    ap.add_argument("--min_dur", type=float, default=2.0)
    ap.add_argument("--min_session_turns", type=int, default=8)
    args = ap.parse_args()
    build(args.csv, args.out_jsonl, args.out_manifest,
          min_dur=args.min_dur, min_session_turns=args.min_session_turns)


if __name__ == "__main__":
    main()
