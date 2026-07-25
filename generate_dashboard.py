"""
generate_dashboard.py — regenerate dashboard.html from traffic_events.tsv.

Usage:
    python generate_dashboard.py              # most recent date in TSV
    python generate_dashboard.py 2026-06-23   # specific date
    python generate_dashboard.py all          # aggregate all dates
"""
import sys
import json
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd

PROJECT_ROOT = Path(__file__).parent
TSV_PATH     = PROJECT_ROOT / "traffic_events.tsv"
HTML_PATH    = PROJECT_ROOT / "dashboard.html"

SPEED_MIN   = 10
BAND_EDGES  = [11, 16, 21, 26, 31, 36, 41, 46, 51]   # 8 bands [low, high)
N_BANDS     = len(BAND_EDGES) - 1
N_HOURS     = 24


def band_index(speed_mph):
    for i in range(N_BANDS):
        if BAND_EDGES[i] <= speed_mph < BAND_EDGES[i + 1]:
            return i
    return None


def build_leaf(df):
    """
    Build the LEAF dict: {"{dir}_{path}_{parking}": [[count]*24]*8}
    'parked' key = events where contains_parked == 1.
    'all'    key = all events regardless of parked flag.
    """
    Z = lambda: [[0] * N_HOURS for _ in range(N_BANDS)]

    leaf = defaultdict(Z)
    for _, row in df.iterrows():
        speed = row.get("speed_mph")
        if pd.isna(speed) or speed < SPEED_MIN:
            continue
        bi = band_index(speed)
        if bi is None:
            continue

        direction = row.get("direction", "")
        if direction not in ("etw", "wte"):
            continue

        motion = row.get("motion_type", "")
        if motion not in ("straight", "turning"):
            continue

        hour = pd.to_datetime(row["event_start"]).hour
        parked = int(row.get("contains_parked", 0) or 0)

        leaf[f"{direction}_{motion}_all"][bi][hour] += 1
        if parked:
            leaf[f"{direction}_{motion}_parked"][bi][hour] += 1

    # Ensure all 8 keys exist even if empty
    for d in ("etw", "wte"):
        for m in ("straight", "turning"):
            for pk in ("all", "parked"):
                _ = leaf[f"{d}_{m}_{pk}"]

    return dict(leaf)


def leaf_to_js(leaf):
    lines = ["const LEAF = {"]
    for key, bands in sorted(leaf.items()):
        band_strs = [f"    {json.dumps(row)}" for row in bands]
        lines.append(f"  {key}: [")
        lines.append(",\n".join(band_strs))
        lines.append("  ],")
    lines.append("};")
    return "\n".join(lines)


def format_date_title(date_str):
    """'2026-06-23' → '23 Jun 2026'"""
    import datetime
    d = datetime.date.fromisoformat(date_str)
    return d.strftime("%-d %b %Y")


def build_all_leaves_js(df, available_dates):
    lines = ["const ALL_LEAVES = {"]
    for date_str in available_dates:
        date_df = df[df["date"] == date_str].copy()
        leaf = build_leaf(date_df)
        lines.append(f'  "{date_str}": {{')
        for key, bands in sorted(leaf.items()):
            band_strs = [f"    {json.dumps(row)}" for row in bands]
            lines.append(f"  {key}: [")
            lines.append(",\n".join(band_strs))
            lines.append("  ],")
        lines.append("  },")
    lines.append("};")
    return "\n".join(lines)


def main():
    df = pd.read_csv(TSV_PATH, sep="\t", low_memory=False)

    # Determine target date(s)
    arg = sys.argv[1] if len(sys.argv) > 1 else "latest"
    available_dates = sorted(df["date"].dropna().unique())

    if arg == "all":
        target_df   = df.copy()
        date_label  = f"{format_date_title(available_dates[0])} – {format_date_title(available_dates[-1])}"
        title_date  = date_label
    else:
        date_str = available_dates[-1] if arg == "latest" else arg
        if date_str not in available_dates:
            print(f"Error: date '{date_str}' not found. Available: {available_dates}")
            sys.exit(1)
        target_df  = df[df["date"] == date_str].copy()
        date_label = format_date_title(date_str)
        title_date = date_label

    total_events = len(target_df)
    qualifying   = int((target_df["speed_mph"].fillna(0) >= SPEED_MIN).sum())

    # Build per-date stats and ALL_LEAVES
    date_stats = {}
    for d in available_dates:
        d_df = df[df["date"] == d]
        date_stats[d] = {
            "total": len(d_df),
            "qualifying": int((d_df["speed_mph"].fillna(0) >= SPEED_MIN).sum()),
        }

    all_leaves_js = build_all_leaves_js(df, available_dates)

    html = HTML_PATH.read_text()

    # ── Patch <title> ──────────────────────────────────────────────────────────
    html = re.sub(
        r"<title>.*?</title>",
        f"<title>Corridor speed dashboard — {title_date}</title>",
        html,
    )

    # ── Patch AVAILABLE_DATES ─────────────────────────────────────────────────
    dates_js = json.dumps(list(available_dates))
    html = re.sub(
        r"const AVAILABLE_DATES = \[.*?\];",
        f"const AVAILABLE_DATES = {dates_js};",
        html,
    )

    # ── Patch DATE_STATS ──────────────────────────────────────────────────────
    html = re.sub(
        r"const DATE_STATS = \{.*?^};",
        f"const DATE_STATS = {json.dumps(date_stats, indent=2)};",
        html,
        flags=re.DOTALL | re.MULTILINE,
    )

    # ── Patch ALL_LEAVES ──────────────────────────────────────────────────────
    html = re.sub(
        r"const ALL_LEAVES = \{.*?^};",
        all_leaves_js,
        html,
        flags=re.DOTALL | re.MULTILINE,
    )

    # ── Patch footer ───────────────────────────────────────────────────────────
    html = re.sub(
        r"Source:.*?</footer>",
        f'Source: <code>traffic_events.db</code> — '
        f'<span id="footerDate"></span><span id="footerStats"></span>\n  </footer>',
        html,
        flags=re.DOTALL,
    )

    HTML_PATH.write_text(html)
    print(f"dashboard.html updated — {date_label}  ({qualifying} events ≥ {SPEED_MIN} mph)")


if __name__ == "__main__":
    main()
