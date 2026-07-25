#!/usr/bin/env python3
"""
cam_sweep.py — WF nighttime camera settings sweep (block experiment)

    --start     Arm sweep; auto-detects experiment day from calendar
    --step WFS  Called from run_download.sh after each pipeline run
    --stop      Disarm sweep (settings remain as-is)
    --status    Show current block and recent results
    --report    Print full results for all completed blocks
    --set-night --gain-max N --exp N
                Apply AI SSA baseline with manual gain/shutter overrides
"""

import hashlib, json, os, re, subprocess, sys
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path

PROJECT      = Path(__file__).parent
STATE_FILE   = PROJECT / "sweep_state.json"
PIPELINE_LOG = PROJECT / "pipeline.log"

WF_HOST           = "http://192.168.50.50"
USERNAME          = "admin"
SWEEP_START_DATE  = date(2026, 7, 6)
WINDOW_START_HOUR = 21
WINDOW_END_HOUR   = 4
BLOCK_MINS        = 15

# Fixed night settings derived from AI SSA baseline
NIGHT_BASELINE = {
    "WideDynamicRange":      50,
    "WideDynamicRangeMode":  1,
    "WhiteBalance":          "ManualDatum",
    "WhiteBalanceDatumRect": [1219, 1522, 7315, 7347],
    "GainBlue":              68,
    "GainRed":               49,
    "ExposureMode":          4,
}

SWEEP = {
    1: {
        "name": "Shutter Speed",
        "var":  "shutter_max",
        "blocks": [
            {"shutter_min": 0.1, "shutter_max": 15, "gain_min": 10, "gain_max": 80, "nr_3d": 40},
            {"shutter_min": 0.1, "shutter_max": 12, "gain_min": 10, "gain_max": 80, "nr_3d": 40},
            {"shutter_min": 0.1, "shutter_max":  9, "gain_min": 10, "gain_max": 80, "nr_3d": 40},
        ],
    },
    2: {
        "name": "Gain",
        "var":  "gain_max",
        "blocks": [
            {"shutter_min": 0.1, "shutter_max": 15, "gain_min": 10, "gain_max":  60, "nr_3d": 40},
            {"shutter_min": 0.1, "shutter_max": 15, "gain_min": 10, "gain_max":  80, "nr_3d": 40},
            {"shutter_min": 0.1, "shutter_max": 15, "gain_min": 10, "gain_max": 100, "nr_3d": 40},
        ],
    },
    3: {
        "name": "3D NR",
        "var":  "nr_3d",
        "blocks": [
            {"shutter_min": 0.1, "shutter_max": 15, "gain_min": 10, "gain_max": 80, "nr_3d":  0},
            {"shutter_min": 0.1, "shutter_max": 15, "gain_min": 10, "gain_max": 80, "nr_3d": 20},
            {"shutter_min": 0.1, "shutter_max": 15, "gain_min": 10, "gain_max": 80, "nr_3d": 40},
        ],
    },
}


# ── Auth / RPC ────────────────────────────────────────────────────────────────

def _post(endpoint, payload):
    r = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{WF_HOST}/{endpoint}",
         "-H", "Content-Type: application/json",
         "-d", json.dumps(payload)],
        capture_output=True, text=True,
    )
    return json.loads(r.stdout) if r.stdout.strip() else {}

def login(password):
    r1 = _post("RPC2_Login", {
        "method": "global.login",
        "params": {"userName": USERNAME, "password": "", "clientType": "Web5.0"},
        "id": 1,
    })
    p         = r1["params"]
    pwd_md5   = hashlib.md5(f"{USERNAME}:{p['realm']}:{password}".encode()).hexdigest().upper()
    auth_hash = hashlib.md5(f"{USERNAME}:{p['random']}:{pwd_md5}".encode()).hexdigest().upper()
    r2 = _post("RPC2_Login", {
        "method": "global.login",
        "params": {"userName": USERNAME, "password": auth_hash,
                   "clientType": "Web5.0", "authorityType": "Default"},
        "id": 2, "session": r1["session"],
    })
    if not r2.get("result"):
        raise RuntimeError(f"WF login failed: {r2}")
    return r2["session"]

def rpc(session, method, params):
    r = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{WF_HOST}/RPC2",
         "-H", "Content-Type: application/json",
         "-H", f"Cookie: DhWebClientSessionID={session}",
         "-d", json.dumps({"method": method, "params": params, "id": 1, "session": session})],
        capture_output=True, text=True,
    )
    return json.loads(r.stdout) if r.stdout.strip() else {}


# ── Camera ops ────────────────────────────────────────────────────────────────

def apply_denoise(session, nr_3d):
    r = rpc(session, "configManager.getConfig", {"name": "VideoInDenoise"})
    table = r["params"]["table"]
    night = next(p for p in table[0] if p.get("Name") == "Night")
    if nr_3d == 0:
        night["3DType"] = "Off"
    else:
        night["3DType"] = "Manual"
        night["3DManulType"]["TnfLevel"] = nr_3d
    result = rpc(session, "configManager.setConfig", {"name": "VideoInDenoise", "table": table})
    if not result.get("result"):
        raise RuntimeError(f"setConfig VideoInDenoise failed: {result}")

def apply_block(block, password):
    session = login(password)
    r = rpc(session, "configManager.getConfig", {"name": "VideoInOptions"})
    table = r["params"]["table"]
    night = table[0]["NightOptions"]
    for k, v in NIGHT_BASELINE.items():
        night[k] = v
    night["ExposureValue1"] = block["shutter_min"]
    night["ExposureValue2"] = block["shutter_max"]
    night["GainMin"]        = block["gain_min"]
    night["GainMax"]        = block["gain_max"]
    result = rpc(session, "configManager.setConfig", {"name": "VideoInOptions", "table": table})
    if not result.get("result"):
        raise RuntimeError(f"setConfig failed: {result}")
    if "nr_3d" in block:
        apply_denoise(session, block["nr_3d"])


# ── Time helpers ──────────────────────────────────────────────────────────────

def get_day_number(today=None):
    d = today or date.today()
    return ((d - SWEEP_START_DATE).days % 3) + 1

def in_window(now=None):
    t = (now or datetime.now()).time()
    return t >= dtime(WINDOW_START_HOUR, 0) or t < dtime(WINDOW_END_HOUR, 0)

def experiment_date(now=None):
    t = now or datetime.now()
    return (t - timedelta(days=1)).date() if t.hour < WINDOW_END_HOUR else t.date()

def mins_since_9pm(now=None):
    t = now or datetime.now()
    total = t.hour * 60 + t.minute
    start = WINDOW_START_HOUR * 60
    return total - start if total >= start else total + (24 * 60 - start)

def block_and_cycle(now=None):
    total = int(mins_since_9pm(now) // BLOCK_MINS)
    return total % 3, total // 3

def block_window_mins(block_idx, cycle):
    total = cycle * 3 + block_idx
    return total * BLOCK_MINS, (total + 1) * BLOCK_MINS

def block_label(var, block):
    if var == "shutter_max": return f"shutter_max={block['shutter_max']}ms"
    if var == "gain_max":    return f"gain_max={block['gain_max']}"
    if var == "nr_3d":       return f"nr_3d={block.get('nr_3d', '?')}"
    return str(block)

def block_start_dt(now, cycle, block_idx):
    base_date = (now - timedelta(days=1)).date() if now.hour < WINDOW_END_HOUR else now.date()
    base = datetime.combine(base_date, dtime(WINDOW_START_HOUR, 0))
    return base + timedelta(minutes=(cycle * 3 + block_idx) * BLOCK_MINS)


# ── Log parsing ───────────────────────────────────────────────────────────────

def clip_mins_since_9pm(h, m):
    total = h * 60 + m
    start = WINDOW_START_HOUR * 60
    if total >= start:
        return total - start
    elif total < WINDOW_END_HOUR * 60:
        return total + (24 * 60 - start)
    return None  # daytime clip

def collect_block_clips(log_text, start_mins, end_mins):
    clips = []
    track_re = re.compile(
        r'^ {2}\[tracks WF (\d{2})\.(\d{2})\.(\d{2})[^\n]+\.mp4\]\n((?:^ {4}[^\n]*\n)*)',
        re.MULTILINE,
    )
    for m in track_re.finditer(log_text):
        h, mn = int(m.group(1)), int(m.group(2))
        cm = clip_mins_since_9pm(h, mn)
        if cm is None or not (start_mins <= cm < end_mins):
            continue
        body     = m.group(4)
        n_tracks = len(re.findall(r'tid=\d+', body))
        n_frames = sum(int(x) for x in re.findall(r'\bn=(\d+)\b', body))
        clips.append({"n_tracks": n_tracks, "n_frames": n_frames})
    return clips


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_block_report(result):
    day     = result["day"]
    day_cfg = SWEEP[day]
    block   = day_cfg["blocks"][result["block_idx"]]
    bs      = datetime.fromisoformat(result["block_start"])
    be      = datetime.fromisoformat(result["block_end"])
    cycle   = result.get("cycle", 0)

    date_str   = bs.strftime("%-d %b")
    time_range = f"{bs.strftime('%-H:%M')}-{be.strftime('%-H:%M')}"
    label      = block_label(day_cfg["var"], block)
    cycle_str  = f" (cycle {cycle + 1})" if cycle > 0 else ""

    clips    = result["clips"]
    n        = len(clips)
    tracks   = [c["n_tracks"] for c in clips]
    frames   = [c["n_frames"] for c in clips]
    mean_t   = round(sum(tracks) / n, 2) if n else 0
    mean_f   = round(sum(frames) / n, 2) if n else 0

    print(f"\n{date_str} (Day {day} – {day_cfg['name']}){cycle_str}")
    print(f"{time_range} | {label}")
    print(f"n_triggered_events: {n}")
    print(f"mean_yolo_tracks: {mean_t}  {tracks}")
    print(f"mean_yolo_frames: {mean_f}  {frames}")


# ── State ─────────────────────────────────────────────────────────────────────

def load_state():
    return json.loads(STATE_FILE.read_text())

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_start():
    password = os.environ.get("CAMERA_PASSWORD") or input("Camera password: ").strip()
    previous_results = []
    if STATE_FILE.exists():
        existing = load_state()
        if existing.get("active"):
            print("[sweep] Already armed. Run --stop first.")
            sys.exit(1)
        previous_results = existing.get("results", [])

    day     = get_day_number()
    day_cfg = SWEEP[day]
    apply_block(day_cfg["blocks"][0], password)

    state = {
        "active":            True,
        "start_date":        SWEEP_START_DATE.isoformat(),
        "window_entered":    False,
        "current_block_idx": 0,
        "current_cycle":     0,
        "results":           previous_results,
    }
    save_state(state)
    lbl = block_label(day_cfg["var"], day_cfg["blocks"][0])
    print(f"[sweep] Armed — Day {day} ({day_cfg['name']}), block 1 settings applied ({lbl}).")
    print(f"[sweep] Window opens at 9 PM.")


def cmd_step(wf_stats):
    if not STATE_FILE.exists():
        return
    state = load_state()
    if not state["active"]:
        return

    now      = datetime.now()
    password = os.environ.get("CAMERA_PASSWORD", "")

    if not in_window(now):
        if state.get("window_entered"):
            state["window_entered"] = False
            save_state(state)
            print("[sweep] Window ended for the night.")
        return

    day     = get_day_number(experiment_date(now))
    day_cfg = SWEEP[day]
    block_idx, cycle = block_and_cycle(now)

    if not state.get("window_entered"):
        state["window_entered"]    = True
        state["current_block_idx"] = block_idx
        state["current_cycle"]     = cycle
        if password:
            apply_block(day_cfg["blocks"][block_idx], password)
        else:
            print("[sweep] WARNING: CAMERA_PASSWORD not set; settings not applied.", file=sys.stderr)
        lbl = block_label(day_cfg["var"], day_cfg["blocks"][block_idx])
        print(f"[sweep] Window opened — Day {day} block {block_idx + 1} ({lbl})")
        save_state(state)
        return

    last_idx   = state.get("current_block_idx", 0)
    last_cycle = state.get("current_cycle", 0)

    if block_idx != last_idx or cycle != last_cycle:
        # Collect and report completed block
        s_mins, e_mins = block_window_mins(last_idx, last_cycle)
        log_text = PIPELINE_LOG.read_text(errors="replace")
        clips    = collect_block_clips(log_text, s_mins, e_mins)
        bs       = block_start_dt(now, last_cycle, last_idx)
        result   = {
            "day":         day,
            "cycle":       last_cycle,
            "block_idx":   last_idx,
            "block_start": bs.isoformat(),
            "block_end":   (bs + timedelta(minutes=BLOCK_MINS)).isoformat(),
            "clips":       clips,
        }
        state["results"].append(result)
        print_block_report(result)

        # Apply new block settings
        if password:
            apply_block(day_cfg["blocks"][block_idx], password)
            lbl = block_label(day_cfg["var"], day_cfg["blocks"][block_idx])
            print(f"[sweep] Block {block_idx + 1} applied ({lbl})")
        else:
            print("[sweep] WARNING: CAMERA_PASSWORD not set; settings not applied.", file=sys.stderr)

        state["current_block_idx"] = block_idx
        state["current_cycle"]     = cycle
        save_state(state)


def cmd_stop():
    if not STATE_FILE.exists():
        print("[sweep] Not armed.")
        return
    state = load_state()
    state["active"] = False
    save_state(state)
    print("[sweep] Stopped. Camera settings unchanged.")


def cmd_status():
    if not STATE_FILE.exists():
        print("[sweep] Not armed.")
        return
    state = load_state()
    if not state["active"]:
        print("[sweep] Inactive.")
        return
    day     = get_day_number()
    day_cfg = SWEEP[day]
    now     = datetime.now()
    if in_window(now):
        block_idx, cycle = block_and_cycle(now)
        block = day_cfg["blocks"][block_idx]
        bs    = block_start_dt(now, cycle, block_idx)
        be    = bs + timedelta(minutes=BLOCK_MINS)
        print(f"[sweep] Active — Day {day} ({day_cfg['name']}), "
              f"block {block_idx + 1}/3 cycle {cycle + 1}: "
              f"{block_label(day_cfg['var'], block)}")
        print(f"  {bs.strftime('%-H:%M')} – {be.strftime('%-H:%M')}")
    else:
        print(f"[sweep] Armed — waiting for 9 PM. Day {day} ({day_cfg['name']})")
    print(f"  Completed blocks: {len(state.get('results', []))}")


def cmd_report():
    if not STATE_FILE.exists():
        print("[sweep] No state file.")
        return
    state = load_state()
    results = state.get("results", [])
    if not results:
        print("[sweep] No completed blocks yet.")
        return
    for result in results:
        print_block_report(result)


def cmd_set_night(gain_max, exp_value2, nr_3d):
    password = os.environ.get("CAMERA_PASSWORD") or input("Camera password: ").strip()
    session  = login(password)
    r        = rpc(session, "configManager.getConfig", {"name": "VideoInOptions"})
    table    = r["params"]["table"]
    night    = table[0]["NightOptions"]
    for k, v in NIGHT_BASELINE.items():
        night[k] = v
    night["GainMin"]        = 10
    night["GainMax"]        = gain_max
    night["ExposureValue1"] = 0.1
    night["ExposureValue2"] = exp_value2
    result = rpc(session, "configManager.setConfig", {"name": "VideoInOptions", "table": table})
    if not result.get("result"):
        raise RuntimeError(f"setConfig failed: {result}")
    apply_denoise(session, nr_3d)
    print(f"[set-night] WF → GainMax={gain_max}  ExposureValue2={exp_value2}ms  WDR=50  WB=ManualDatum  nr_3d={nr_3d}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "--set-night":
        args = sys.argv[2:]
        gain_max, exp_val, nr_3d = None, None, 40
        for i, a in enumerate(args):
            if a == "--gain-max" and i + 1 < len(args): gain_max = int(args[i + 1])
            if a == "--exp"      and i + 1 < len(args): exp_val  = int(args[i + 1])
            if a == "--nr-3d"    and i + 1 < len(args): nr_3d    = int(args[i + 1])
        if gain_max is None or exp_val is None:
            print("Usage: cam_sweep.py --set-night --gain-max N --exp N [--nr-3d N]")
            sys.exit(1)
        cmd_set_night(gain_max, exp_val, nr_3d)
    elif cmd == "--start":  cmd_start()
    elif cmd == "--step":   cmd_step(sys.argv[2] if len(sys.argv) > 2 else "")
    elif cmd == "--stop":   cmd_stop()
    elif cmd == "--status": cmd_status()
    elif cmd == "--report": cmd_report()
    else:
        print(__doc__)
        sys.exit(1)
