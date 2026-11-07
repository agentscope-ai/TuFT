"""Offline trace reconstruction and CPU-only reproduction of queue barriers."""

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from tuft.mutation_queue import MutationQueue


async def queue_order(checkpoint_tenant):
    queue = MutationQueue(max_overtakes=2)
    order = []

    async def operation(name, tenant, priority):
        async with queue.reserve(tenant, priority):
            order.append(name)

    async with queue.reserve("owner", False):
        tasks = []
        for name, tenant, priority in (
            ("backward", "b", False),
            ("checkpoint", checkpoint_tenant, True),
            ("optimizer", "c", True),
        ):
            tasks.append(asyncio.create_task(operation(name, tenant, priority)))
            await asyncio.sleep(0)
    await asyncio.gather(*tasks)
    return order


def main(directory, output):
    result = {"arms": {}}
    for arm in ("off", "on"):
        data = json.loads((directory / f"tenants-8-{arm}.json").read_text())
        origin = data["scheduler_before"]["elapsed_s"]
        operations = []
        for event in data["scheduler_after"]["decision_trace"]:
            if event["event"] != "exclusive" or event["elapsed_s"] < origin:
                continue
            finish = event["elapsed_s"] - origin
            service_start = finish - event["service_s"]
            acquired = service_start - event["phase_s"] - event["drain_s"]
            operations.append(
                dict(
                    event,
                    finish_s=finish,
                    submitted_s=acquired - event["queue_s"],
                    acquired_s=acquired,
                    drain_finished_s=acquired + event["drain_s"],
                )
            )
        phases = []
        for tenant in range(data["tenants"]):
            training = [
                u
                for u in data["updates"]
                if u["tenant"] == tenant
                and u["finished_s"] - u["duration_s"] <= 750 < u["finished_s"]
            ]
            publishing = [
                p
                for p in data["publications"]
                if p["tenant"] == f"agent-{tenant}"
                and p.get("start_s") is not None
                and p["start_s"] <= 750 < p["start_s"] + p["duration_s"]
            ]
            phases.append(dict(tenant=tenant, training=training, publishing=publishing))
        result["arms"][arm] = dict(
            mean_prompt_tokens=statistics.mean(r["prompt_tokens"] for r in data["requests"]),
            mean_output_tokens=statistics.mean(r["output_tokens"] for r in data["requests"]),
            total_phase_switch_s=sum(e["phase_s"] for e in operations),
            drain_events=[e for e in operations if e["drain_s"] > 0.1],
            critical_operations=[e for e in operations if 500 < e["finish_s"] < 835],
            tenant_states_at_750_s=phases,
        )
    actual = asyncio.run(queue_order(None))
    scoped = asyncio.run(queue_order("a"))
    assert actual == ["backward", "checkpoint", "optimizer"]
    assert scoped == ["checkpoint", "optimizer", "backward"]
    result["cpu_queue_reproduction"] = dict(global_checkpoint=actual, tenant_checkpoint=scoped)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["cpu_queue_reproduction"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.directory, args.output)
