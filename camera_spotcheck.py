"""
camera_spotcheck.py — daily WF/EF still spot-check via Pushover.

Picks the first N full-match events with wf_entry_time >= START_HOUR for the
given date (default: today), grabs a frame from each camera's clip, and
pushes a combined WF|EF still per event to Pushover — a quick visual check
for fogged/streaked windows or an out-of-focus camera. Sends at most once
per day (tracked in spotcheck_state.json).

Usage:
    python camera_spotcheck.py [YYYY_MM_DD]

Requires PUSHOVER_USER_KEY and PUSHOVER_API_TOKEN in the environment.
"""
import os
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import requests

import traffic_utils as tu

PROJECT_ROOT = Path("/Users/jrill/Documents/traffic_project")
DB_PATH      = PROJECT_ROOT / "traffic_events.db"
STATE_PATH   = PROJECT_ROOT / "spotcheck_state.json"
YOLO_MODEL   = PROJECT_ROOT / "yolov8n.pt"

START_HOUR    = 8   # only consider events at/after this hour
TARGET_COUNT  = 3   # stills per day
FALLBACK_HOUR = 11  # if still short of TARGET_COUNT by this hour, send what we have
N_SAMPLE      = 15  # frames sampled per clip when hunting for the vehicle

PUSHOVER_USER  = os.environ.get("PUSHOVER_USER_KEY")
PUSHOVER_TOKEN = os.environ.get("PUSHOVER_API_TOKEN")

model = tu.init_model(YOLO_MODEL)


def _bbox_score(x, y, w, h, frame_w, frame_h):
    x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
    cut = (max(0, -x1) + max(0, x2 - frame_w) +
           max(0, -y1) + max(0, y2 - frame_h))
    completeness = max(0.0, 1.0 - cut / (2 * (w + h)))
    x_central = 1.0 if 0.25 <= (x / frame_w) <= 0.75 else 0.1
    return w * h * completeness * x_central


def best_vehicle_frame(clip_path, n_sample=N_SAMPLE):
    """Sample frames across the clip and return (frame, bbox_xywh) for the best-scored vehicle."""
    cap   = cv2.VideoCapture(str(clip_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return None, None
    idxs = [int(total * i / (n_sample + 1)) for i in range(1, n_sample + 1)]
    best_frame, best_box, best_score = None, None, 0.0
    fallback_frame = None
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        if fallback_frame is None:
            fallback_frame = frame.copy()
        fh, fw = frame.shape[:2]
        results = model(frame, classes=tu.YOLO_CLASSES, verbose=False)[0]
        if results.boxes and len(results.boxes):
            scores = [
                _bbox_score(*results.boxes.xywh[i].tolist(), fw, fh)
                for i in range(len(results.boxes))
            ]
            bi    = int(np.argmax(scores))
            score = scores[bi]
            if score > best_score:
                best_score = score
                best_frame = frame.copy()
                best_box   = results.boxes.xywh[bi].tolist()
    cap.release()
    return (best_frame if best_frame is not None else fallback_frame), best_box


def _load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def _save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2))


def _fetch_events(date_str):
    sql_date = date_str.replace("_", "-")
    cutoff = f"{sql_date}T{START_HOUR:02d}:00:00"
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        """SELECT wf_entry_time, direction, wf_clip, ef_clip
           FROM traffic_events
           WHERE date = ? AND wf_entry_time >= ?
           ORDER BY wf_entry_time ASC
           LIMIT ?""",
        (sql_date, cutoff, TARGET_COUNT),
    ).fetchall()
    con.close()
    return rows


def _combined_still(wf_path, ef_path, label):
    H = 480

    def render(path, tag):
        if path is None or not path.exists():
            ph = np.full((H, 640, 3), 60, dtype=np.uint8)
            cv2.putText(ph, f"NO {tag} CLIP", (140, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (160, 160, 160), 2)
            return ph
        frame, _box = best_vehicle_frame(path)
        if frame is None:
            return np.zeros((H, 640, 3), dtype=np.uint8)
        h, w  = frame.shape[:2]
        scale = H / h
        frame = cv2.resize(frame, (int(w * scale), H))
        cv2.putText(frame, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame

    combined = np.hstack([render(wf_path, "WF"), render(ef_path, "EF")])
    header = np.zeros((36, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return np.vstack([header, combined])


def _send_pushover(image_path, title, message):
    with open(image_path, "rb") as f:
        resp = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={"token": PUSHOVER_TOKEN, "user": PUSHOVER_USER,
                  "title": title, "message": message},
            files={"attachment": (image_path.name, f, "image/jpeg")},
            timeout=30,
        )
    resp.raise_for_status()
    return resp.json()


def main():
    date_str = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y_%m_%d")

    if not PUSHOVER_USER or not PUSHOVER_TOKEN:
        print("[spotcheck] PUSHOVER_USER_KEY / PUSHOVER_API_TOKEN not set — skipping")
        return

    state = _load_state()
    if state.get(date_str) == "sent":
        print(f"[spotcheck] already sent for {date_str} — skipping")
        return

    now = datetime.now()
    if now.hour < START_HOUR:
        print("[spotcheck] before start hour — skipping")
        return

    events = _fetch_events(date_str)
    if not events:
        print(f"[spotcheck] no qualifying events for {date_str} yet")
        return
    if len(events) < TARGET_COUNT and now.hour < FALLBACK_HOUR:
        print(f"[spotcheck] only {len(events)}/{TARGET_COUNT} events so far — waiting")
        return

    data_dir = PROJECT_ROOT / "traffic_data" / date_str
    out_path = PROJECT_ROOT / "spotcheck_tmp.jpg"
    labels, panels = [], []
    for wf_entry, direction, wf_clip, ef_clip in events:
        wf_path = (data_dir / "WF" / "mp4" / wf_clip) if wf_clip else None
        ef_path = (data_dir / "EF" / "mp4" / ef_clip) if ef_clip else None
        ts = datetime.fromisoformat(wf_entry)
        label = f"{ts:%H:%M:%S}  {direction}"
        labels.append(label)
        panels.append(_combined_still(wf_path, ef_path, label))

    max_w = max(p.shape[1] for p in panels)
    padded = [
        p if p.shape[1] == max_w
        else np.hstack([p, np.zeros((p.shape[0], max_w - p.shape[1], 3), dtype=np.uint8)])
        for p in panels
    ]
    stacked = np.vstack(padded)
    cv2.imwrite(str(out_path), stacked)

    try:
        _send_pushover(out_path, "Camera Spot-Check", f"{date_str}  —  " + ", ".join(labels))
        print(f"[spotcheck] sent {len(events)} stills for {date_str} in one push")
        state[date_str] = "sent"
        _save_state(state)
    except Exception as e:
        print(f"[spotcheck] send failed: {e}")
    finally:
        out_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
