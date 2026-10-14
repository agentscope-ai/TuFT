"""Historical duplicate borrowing must never be reported as physical sharing."""

import pytest

from scripts.summarize_loopweave import comparison_runs


def test_historical_comparison_excludes_duplicate_borrowing_without_mutating_data():
    static = {"policy": "round_robin"}
    duplicate = {"policy": "loopweave", "shared_weights": None}
    shared = {"policy": "loopweave_shared", "shared_weights": {"awake": {}}}
    data = {"runs": [static, duplicate, shared]}
    selected = comparison_runs(data)
    assert selected == [static, {**shared, "policy": "loopweave"}]
    assert data["runs"] == [static, duplicate, shared]
    assert shared["policy"] == "loopweave_shared"


def test_current_static_comparison_requires_alias_evidence():
    rows = [
        {"policy": "round_robin"},
        {"policy": "loopweave", "shared_weights": {"awake": {}}},
    ]
    assert comparison_runs({"runs": rows}) == rows
    rows[1]["shared_weights"] = None
    with pytest.raises(ValueError, match="physically shared"):
        comparison_runs({"runs": rows})
