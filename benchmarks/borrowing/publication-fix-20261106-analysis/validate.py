import hashlib
import json
import sys
from pathlib import Path


root = Path(sys.argv[1])
out = Path(sys.argv[2])
checks = {}
for arm in ("off", "on"):
    d = json.loads((root / f"tenants-8-{arm}.json").read_text())
    before = d["scheduler_before"]
    after = d["scheduler_after"]
    events = [e for e in after["decision_trace"] if e["elapsed_s"] >= before["elapsed_s"]]
    cs = [e for e in events if e["event"] == "exclusive" and e["kind"] == "checkpoint"]
    ps = [p for p in d["publications"] if p.get("start_s") is not None]
    c = dict(
        frozen_base_unchanged=d["frozen_base_unchanged"],
        copied_parameters=d["shared"]["copied_parameters"],
        requests=len(d["requests"]),
        updates=len(d["updates"]),
        all_tenants_updated=all(any(u["tenant"] == i for u in d["updates"]) for i in range(8)),
        output_limited=sum(r["hit_output_limit"] for r in d["requests"]),
        measurement_window=d["independent_clients"]["measurement_window_s"],
        checkpoint_events=len(cs),
        checkpoint_queue_max_s=max((e["queue_s"] for e in cs), default=0),
        checkpoint_service_max_s=max((e["service_s"] for e in cs), default=0),
        publication_count=len(ps),
        global_publication_gate_events=sum(
            e["event"] == "exclusive" and e["kind"] == "publication" for e in events
        ),
    )
    assert c["frozen_base_unchanged"] and c["copied_parameters"] == 0
    assert c["requests"] == c["updates"] * 32 and c["all_tenants_updated"]
    assert c["measurement_window"] == [240, 840]
    assert c["global_publication_gate_events"] == 0
    checks[arm] = c
x = json.loads((root / "tenants-8-off.json").read_text())
y = json.loads((root / "tenants-8-on.json").read_text())
assert x["initial_adapter_fingerprints"] == y["initial_adapter_fingerprints"]
checks["matching_initial_adapters"] = True
checks["raw_sha256"] = {
    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in root.glob("tenants-8-*.json")
    if "partial" not in p.name
}
out.write_text(json.dumps(checks, indent=2) + "\n")
print(json.dumps(checks, indent=2))
