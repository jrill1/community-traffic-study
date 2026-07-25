"""
lookup_make_model.py — look up make/model/year for each vehicle in traffic_data/make_model/
using the platetovin.com API.

Usage:
    python lookup_make_model.py                  # all dates
    python lookup_make_model.py 2026_07_08       # specific date folder
    python lookup_make_model.py --dry-run        # print plates without calling API

Requires:
    PLATETOVIN_API_KEY env var  (https://platetovin.com)
    pip install requests pandas

Output:
    traffic_data/make_model/vehicle_cache_platetovin.json  — plate → API result (persists across runs)
    traffic_data/make_model/make_model.tsv                 — one row per car event (EF/WF combined)

The cache means each unique plate is only looked up once, ever.  EF/WF image pairs
for the same event are merged into a single TSV row with filename_ef / filename_wf columns.
"""

import json
import os
import sys
import time
import logging
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
MAKE_MODEL_DIR = PROJECT_ROOT / "traffic_data" / "make_model"
CACHE_PATH = MAKE_MODEL_DIR / "vehicle_cache_platetovin.json"
TSV_PATH = MAKE_MODEL_DIR / "make_model.tsv"
CORRECTIONS_PATH = MAKE_MODEL_DIR / "plate_corrections.json"

API_BASE = "https://platetovin.com/api/convert"
DEFAULT_STATE = "NJ"
EVENTS_TSV = PROJECT_ROOT / "traffic_events.tsv"
CONF_THRESHOLD = 0.95
MAX_PAIRS_PER_PLATE = 2  # must match extract_vehicle_crops.py

REQUEST_DELAY = 0.25

TSV_COLS = [
    "date", "timestamp", "plate", "filename_ef", "filename_wf",
    "make", "model", "year", "trim", "drivetrain", "engine",
    "api_success", "api_error", "source", "reviewed", "flagged", "thresholded",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_stem(stem: str) -> tuple[str, str, str, str]:
    """Return (date_part, timestamp, plate, direction) from a make_model filename stem.

    Filename format: YYYYMMDD_HHMMSS_PLATE_DIRECTION
    e.g. 20260708_073943_A12BCD_EF  →  ('20260708', '073943', 'A12BCD', 'EF')
    """
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {stem!r}")
    return parts[0], parts[1], parts[2], parts[3]


def load_cache() -> dict:
    if CACHE_PATH.exists():
        with CACHE_PATH.open() as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    MAKE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w") as f:
        json.dump(cache, f, indent=2)


def lookup_plate(plate: str, api_key: str, state: str) -> dict:
    """Call the platetovin.com API.  Returns the parsed JSON response."""
    resp = requests.post(
        API_BASE,
        headers={"Authorization": api_key, "Content-Type": "application/json"},
        json={"state": state, "plate": plate},
        timeout=10,
    )
    resp.raise_for_status()
    result = resp.json()
    result.get("vin", {}).pop("vin", None)
    result["source"] = "platetovin"
    return result


def load_confidence_index() -> dict[tuple, float]:
    """Return {(date_dir, timestamp, plate): max_conf} from traffic_events.tsv."""
    if not EVENTS_TSV.exists():
        log.warning("%s not found — confidence filter disabled.", EVENTS_TSV)
        return {}
    df = pd.read_csv(EVENTS_TSV, sep="\t", dtype=str).fillna("")
    index = {}
    for _, r in df.iterrows():
        try:
            ef = float(r.get("ef_plate_conf") or 0)
            wf = float(r.get("wf_plate_conf") or 0)
        except ValueError:
            continue
        parts = r.get("event_id", "").split("_")
        if len(parts) < 2:
            continue
        d = parts[0]  # "20260604"
        date_dir = f"{d[:4]}_{d[4:6]}_{d[6:]}"  # "2026_06_04"
        timestamp = parts[1]                       # "092401"
        plate = r.get("plate", "")
        if plate:
            index[(date_dir, timestamp, plate)] = max(ef, wf)
    return index


def collect_image_files(date_filter: str | None) -> list[Path]:
    """Return sorted list of .jpg files under make_model/, optionally filtered by date."""
    if date_filter:
        dirs = [MAKE_MODEL_DIR / date_filter]
    else:
        dirs = sorted(p for p in MAKE_MODEL_DIR.iterdir() if p.is_dir())

    files = []
    for d in dirs:
        if not d.exists():
            log.error("Directory not found: %s", d)
            continue
        files.extend(sorted(d.glob("*.jpg")))
    return files


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    dry_run = "--dry-run" in sys.argv
    debug   = "--debug"   in sys.argv

    limit: int | None = None
    for arg in sys.argv[1:]:
        if arg.startswith("--limit="):
            limit = int(arg.split("=", 1)[1])

    date_filter = next(
        (a for a in sys.argv[1:] if not a.startswith("--")), None
    )

    api_key = os.environ.get("PLATETOVIN_API_KEY", "")
    if not api_key and not dry_run:
        log.error("PLATETOVIN_API_KEY env var not set.  Export it or use --dry-run.")
        sys.exit(1)

    state = os.environ.get("PLATETOVIN_STATE", DEFAULT_STATE)

    images = collect_image_files(date_filter)
    if not images:
        log.error("No images found under %s", MAKE_MODEL_DIR)
        sys.exit(1)

    log.info("Found %d jpg files on disk (%s)",
             len(images),
             date_filter or "all dates")

    conf_index = load_confidence_index()

    cache = load_cache()

    corrections: set[str] = (
        set(json.loads(CORRECTIONS_PATH.read_text()))
        if CORRECTIONS_PATH.exists() else set()
    )
    if corrections:
        log.info("Correction queue: %d plate(s) — %s", len(corrections), ", ".join(sorted(corrections)))

    # Pre-load existing TSV rows so we can skip them and preserve their data.
    existing_keys: set[tuple] = set()
    events: dict[tuple, dict] = {}
    if TSV_PATH.exists():
        existing_df = pd.read_csv(TSV_PATH, sep="\t", dtype=str).fillna("")
        if "thresholded" not in existing_df.columns:
            existing_df["thresholded"] = ""
        for _, r in existing_df.iterrows():
            key = (r["date"], r["timestamp"], r["plate"])
            existing_keys.add(key)
            events[key] = r.to_dict()

    from collections import Counter
    plate_event_count = Counter(
        v["plate"] for v in events.values()
        if not (v.get("flagged") == "1" and v.get("api_success") == "1")
    )

    eligible = set()
    for img_path in images:
        try:
            _, ts, pl, _ = parse_stem(img_path.stem)
        except ValueError:
            continue
        ek = (img_path.parent.name, ts, pl)
        if ek in existing_keys:
            continue
        if conf_index and pl not in corrections and conf_index.get(ek, 0.0) <= CONF_THRESHOLD:
            continue
        eligible.add(ek)
    log.info("%d new events on disk passing conf >%.2f", len(eligible), CONF_THRESHOLD)

    # Process corrected plates first so they're never bumped by --limit.
    if corrections:
        images.sort(key=lambda p: (
            0 if len(p.stem.split("_")) >= 3 and p.stem.split("_")[2] in corrections else 1,
            str(p),
        ))

    new_lookups = 0
    new_event_count = 0
    processed_corrections: set[str] = set()

    for img_path in images:
        date_dir = img_path.parent.name
        stem = img_path.stem
        try:
            _, timestamp, plate, direction = parse_stem(stem)
        except ValueError as e:
            log.warning("Skipping %s: %s", img_path.name, e)
            continue

        event_key = (date_dir, timestamp, plate)

        # Skip low-confidence OCR reads (bypass for correction queue plates).
        if conf_index and plate not in corrections:
            conf = conf_index.get(event_key, 0.0)
            if conf <= CONF_THRESHOLD:
                log.debug("Skipping %s: conf %.3f <= %.2f", plate, conf, CONF_THRESHOLD)
                continue

        # Skip events already in the TSV (they don't count toward the limit).
        if event_key in existing_keys:
            continue

        if limit is not None and event_key not in events and new_event_count >= limit:
            continue

        # Look up plate if not already cached
        if plate not in cache:
            if dry_run:
                log.info("[dry-run] would look up: %s (%s)", plate, state)
                cache[plate] = {"_dry_run": True}
            else:
                log.info("Looking up %s ...", plate)
                try:
                    result = lookup_plate(plate, api_key, state)
                    if debug:
                        print(json.dumps(result, indent=2))
                    cache[plate] = result
                    new_lookups += 1
                    save_cache(cache)       # persist after each call
                    time.sleep(REQUEST_DELAY)
                except requests.HTTPError as e:
                    if e.response is not None and e.response.status_code == 402:
                        log.error("Out of credits (402). Stopping without caching plate %s.", plate)
                        sys.exit(1)
                    not_found = e.response is not None and e.response.status_code == 404
                    log.warning("%s for plate %s: %s", "Not found" if not_found else "HTTP error", plate, e)
                    cache[plate] = {"success": False, "not_found": not_found, "error": str(e)}
                    save_cache(cache)
                except Exception as e:
                    log.warning("Error for plate %s: %s", plate, e)
                    cache[plate] = {"success": False, "error": str(e)}
                    save_cache(cache)
        else:
            log.debug("Cache hit: %s", plate)

        result = cache[plate]
        success = result.get("success", False) and not result.get("_dry_run")
        vin_data = result.get("vin", {}) if success else {}

        if event_key not in events:
            over_cap = plate_event_count.get(plate, 0) >= MAX_PAIRS_PER_PLATE
            if over_cap and plate not in corrections:
                log.debug("Over cap (%d pairs): %s — skipping", MAX_PAIRS_PER_PLATE, plate)
                continue
            new_event_count += 1
            if plate in corrections:
                processed_corrections.add(plate)
            if over_cap:
                log.info("Over cap (%d pairs): %s — correction queue, marking thresholded", MAX_PAIRS_PER_PLATE, plate)
            events[event_key] = {
                "date":        date_dir,
                "timestamp":   timestamp,
                "plate":       plate,
                "filename_ef": "",
                "filename_wf": "",
                "make":        vin_data.get("make", ""),
                "model":       vin_data.get("model", ""),
                "year":        vin_data.get("year", ""),
                "trim":        vin_data.get("trim", ""),
                "drivetrain":  vin_data.get("driveType", ""),
                "engine":      vin_data.get("engine", ""),
                "api_success": "1" if success else "0",
                "api_error":   result.get("error", ""),
                "source":      result.get("source", ""),
                "reviewed":    "1" if over_cap else "",
                "flagged":     "",
                "thresholded": "1" if over_cap else "",
            }
            plate_event_count[plate] += 1

        # Slot this image into the correct direction column
        if direction == "EF":
            events[event_key]["filename_ef"] = img_path.name
        elif direction == "WF":
            events[event_key]["filename_wf"] = img_path.name
        else:
            log.warning("Unknown direction %r in %s", direction, img_path.name)

    rows = list(events.values())

    # Write TSV atomically — build in a temp file then rename so a crash mid-write
    # never leaves the TSV truncated.
    MAKE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = TSV_PATH.with_suffix(".tmp")
    with tmp_path.open("w") as f:
        f.write("\t".join(TSV_COLS) + "\n")
        for row in rows:
            f.write("\t".join(str(row[c]) for c in TSV_COLS) + "\n")
    tmp_path.replace(TSV_PATH)

    if processed_corrections:
        pending = set(json.loads(CORRECTIONS_PATH.read_text())) - processed_corrections
        if pending:
            CORRECTIONS_PATH.write_text(json.dumps(sorted(pending), indent=2))
        else:
            CORRECTIONS_PATH.unlink()
        log.info("Correction queue: cleared %s", ", ".join(sorted(processed_corrections)))

    log.info(
        "Done. %d new events (+%d existing), %d unique plates, %d new API calls.",
        new_event_count,
        len(existing_keys),
        len(cache),
        new_lookups,
    )
    log.info("Cache: %s", CACHE_PATH)
    log.info("TSV:   %s", TSV_PATH)

    if dry_run:
        unique_plates = sorted({k[2] for k in events})
        print("\nPlates that would be looked up:")
        for p in unique_plates:
            if p not in load_cache() or load_cache().get(p, {}).get("_dry_run"):
                print(f"  {p}")


if __name__ == "__main__":
    main()
