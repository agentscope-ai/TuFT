"""Run lightweight, deterministic checks over matched stage samples.

These checks are descriptive diagnostics, not a validated sarcasm classifier or
a substitute for blinded human/model-judge evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


OVERT_SARCASM_CUES = {
    "oh_absolutely": re.compile(r"\boh,\s+absolutely\b", re.IGNORECASE),
    "nothing_says": re.compile(r"\bnothing says\b", re.IGNORECASE),
    "bravo": re.compile(r"\bbravo\b", re.IGNORECASE),
    "groundbreaking": re.compile(r"\bgroundbreaking\b", re.IGNORECASE),
    "masterclass": re.compile(r"\bmasterclass\b", re.IGNORECASE),
    "pinnacle": re.compile(r"\bpinnacle\b", re.IGNORECASE),
    "revolutionary": re.compile(r"\brevolutionary\b", re.IGNORECASE),
}


def _load_arms(path: Path) -> dict[str, list[dict]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    arms = payload.get("arms")
    if not isinstance(arms, dict) or not arms:
        raise ValueError(f"{path} does not contain a non-empty 'arms' object")
    return arms


def _validate_matched(arms: dict[str, list[dict]]) -> None:
    labels = list(arms)
    expected = [(row["prompt"], row["seed"]) for row in arms[labels[0]]]
    for label in labels[1:]:
        observed = [(row["prompt"], row["seed"]) for row in arms[label]]
        if observed != expected:
            raise ValueError(f"arm {label!r} does not use the same ordered prompts and seeds")


def _evaluate_arm(rows: list[dict]) -> dict:
    cue_hits = []
    for row in rows:
        matches = [
            name
            for name, pattern in OVERT_SARCASM_CUES.items()
            if pattern.search(row["response"])
        ]
        cue_hits.append({"prompt": row["prompt"], "cues": matches})

    count = len(rows)
    cue_count = sum(bool(hit["cues"]) for hit in cue_hits)
    complete_count = sum(not row["hit_cap"] for row in rows)
    return {
        "responses": count,
        "overt_sarcasm_cue_count": cue_count,
        "overt_sarcasm_cue_rate": cue_count / count,
        "completed_without_token_cap_count": complete_count,
        "completed_without_token_cap_rate": complete_count / count,
        "median_response_tokens": statistics.median(row["tokens"] for row in rows),
        "cue_hits": cue_hits,
    }


def evaluate(path: Path) -> dict:
    arms = _load_arms(path)
    _validate_matched(arms)
    return {
        "schema_version": 1,
        "source": str(path),
        "prompt_count": len(next(iter(arms.values()))),
        "same_prompt_and_seed_across_arms": True,
        "metrics": {
            "overt_sarcasm_cue_rate": (
                "Fraction of responses containing at least one published English surface cue; "
                "this is a conservative heuristic, not a sarcasm classifier."
            ),
            "completed_without_token_cap_rate": (
                "Fraction of responses that ended before the configured generation cap."
            ),
            "median_response_tokens": "Median generated-token count within an arm.",
        },
        "cue_patterns": list(OVERT_SARCASM_CUES),
        "arms": {label: _evaluate_arm(rows) for label, rows in arms.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="sample_outputs.json or companion_run.json")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()

    report = evaluate(args.input)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
