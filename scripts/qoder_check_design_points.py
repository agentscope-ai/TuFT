#!/usr/bin/env python3
"""Validate TuFT P0 design points before long campaign runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qoder_build_paper_tables import collect

EXPECTED = {
    "ablation_fastloop_8gpu": {"holds_armed": "positive"},
    "ablation_sampling_tuned_8gpu": {
        "l1_a0_merged": "positive",
        "l2_dispatch_runs": "positive",
        "holds_armed": "positive",
    },
    "ablation_reactive_8gpu": {
        "l1_a0_merged": "positive",
        "duty_to_sampling": "positive",
    },
    "slowloop_static_1p3_8gpu": {
        "l1_a0_merged": "positive",
        "duty_to_sampling": "positive",
    },
    "slowloop_grow_1p3_8gpu": {
        "l1_a0_merged": "positive",
        "duty_to_sampling": "positive",
        "conversions": "positive",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = collect(root)
    by_arm = {row["arm"].split("/")[0]: row for row in rows}
    failures: list[str] = []
    selected: list[dict] = []
    for arm, checks in EXPECTED.items():
        row = by_arm.get(arm)
        if row is None:
            failures.append(f"{arm}: missing result")
            continue
        selected.append(row)
        if not row["complete"]:
            failures.append(f"{arm}: incomplete tenants")
        for field, expectation in checks.items():
            if expectation == "positive" and row[field] <= 0:
                failures.append(f"{arm}: {field}={row[field]}, expected > 0")
    # These zeros are deliberate ablations, not failed design points.
    fast = by_arm.get("ablation_fastloop_8gpu")
    if fast:
        for field in ("l1_a0_merged", "duty_to_sampling", "conversions"):
            if fast[field] != 0:
                failures.append(f"ablation_fastloop_8gpu: {field}={fast[field]}, expected 0")
    static = by_arm.get("slowloop_static_1p3_8gpu")
    if static and static["conversions"] != 0:
        failures.append(
            f"slowloop_static_1p3_8gpu: conversions={static['conversions']}, expected 0"
        )
    output = args.output or root / "design_point_check.json"
    output.write_text(json.dumps({"ok": not failures, "failures": failures, "rows": selected}, indent=2) + "\n")
    for row in selected:
        print(
            f"{row['arm'].split('/')[0]:<34} complete={row['complete']} "
            f"L1={row['l1_a0_merged']} duty>smp={row['duty_to_sampling']} "
            f"conv={row['conversions']} holds={row['holds_armed']}"
        )
    if failures:
        print("P0 GATE FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("P0 GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
