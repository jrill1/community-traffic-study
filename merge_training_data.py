#!/usr/bin/env python3
"""
Merge a remote training data file (Mini) with a local annotated copy (Laptop).

For TSV files (make_model.tsv, annotations.tsv):
  - Remote wins on new rows and factual columns
  - Local wins on annotation columns (reviewed, flagged, thresholded)

For JSON files (annotations.json):
  - Remote wins on new keys
  - Local wins on annotation fields within shared keys

Usage:
    python merge_training_data.py --local <path> --remote <path> --output <path>
"""

import argparse
import csv
import json
from pathlib import Path

MAKE_MODEL_ANNOTATION_COLS = {'reviewed', 'flagged', 'thresholded'}
MAKE_MODEL_TSV_COLS = [
    "date", "timestamp", "plate", "filename_ef", "filename_wf",
    "make", "model", "year", "trim", "drivetrain", "engine",
    "api_success", "api_error", "source", "reviewed", "flagged", "thresholded",
]


def load_tsv(path: Path) -> tuple[dict, list, list]:
    """Return (rows_by_key, fieldnames, key_cols)."""
    if not path.exists():
        return {}, [], []
    with path.open() as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames or []
        key_cols = ['file'] if 'file' in fieldnames else ['date', 'timestamp', 'plate']
        rows = {}
        for r in reader:
            key = tuple(r[c] for c in key_cols)
            rows[key] = r
    return rows, fieldnames, key_cols


def merge_tsv(local_path: Path, remote_path: Path, output_path: Path) -> None:
    local,  local_fields,  key_cols = load_tsv(local_path)
    remote, remote_fields, _        = load_tsv(remote_path)

    # Determine which columns are annotations (all non-key cols for annotations.tsv;
    # fixed set for make_model.tsv)
    if 'file' in (remote_fields or []):
        annotation_cols = {'label'}
        fieldnames = remote_fields or local_fields
        sort_key = lambda r: tuple(r[c] for c in key_cols)
    else:
        annotation_cols = MAKE_MODEL_ANNOTATION_COLS
        fieldnames = MAKE_MODEL_TSV_COLS
        sort_key = lambda r: (r['date'], r['timestamp'])

    merged = {}

    for key, row in remote.items():
        merged[key] = dict(row)
        if key in local:
            for col in annotation_cols:
                if col in local[key]:
                    merged[key][col] = local[key][col]

    for key, row in local.items():
        if key not in merged:
            merged[key] = dict(row)

    rows = sorted(merged.values(), key=sort_key)

    tmp = output_path.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t',
                                lineterminator='\n', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(output_path)

    new = sum(1 for k in remote if k not in local)
    print(f"  {output_path.name}: {len(rows)} rows total ({new} new from Mini)")


def merge_json(local_path: Path, remote_path: Path, output_path: Path) -> None:
    local  = json.loads(local_path.read_text())  if local_path.exists()  else {}
    remote = json.loads(remote_path.read_text()) if remote_path.exists() else {}

    if not isinstance(local, dict) or not isinstance(remote, dict):
        raise ValueError(f"Expected JSON objects (dicts) in {local_path.name}")

    merged = dict(remote)
    for key, local_val in local.items():
        if key not in merged:
            merged[key] = local_val
        elif isinstance(local_val, dict) and isinstance(merged[key], dict):
            # Local wins on all annotation fields for shared keys
            merged[key].update(local_val)

    tmp = output_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(merged, indent=2))
    tmp.replace(output_path)

    new = sum(1 for k in remote if k not in local)
    print(f"  {output_path.name}: {len(merged)} keys total ({new} new from Mini)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local',  required=True, type=Path)
    parser.add_argument('--remote', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()

    suffix = args.local.suffix.lower()
    if suffix == '.tsv':
        merge_tsv(args.local, args.remote, args.output)
    elif suffix == '.json':
        merge_json(args.local, args.remote, args.output)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


if __name__ == '__main__':
    main()
