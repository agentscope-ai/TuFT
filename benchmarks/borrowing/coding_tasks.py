"""Incremental order-report coding tasks; generated programs run only in Modal Sandboxes."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass, field


MAX_ATTEMPTS = 8


MILESTONES = [
    "Implement build_report(orders). Each order has customer, sku, qty and price_cents. "
    "Return orders (count), units, gross_cents, customers (sorted unique names), "
    "and by_sku (a dictionary of total units for each sku). Empty input returns zeros, [] and {}. "
    "Input for this milestone contains only valid orders.",
    "Keep the previous outputs and add rejected and cancelled counts. A valid order is a dict "
    "with nonempty string customer and sku (strip surrounding whitespace), qty a positive int, "
    "and price_cents a nonnegative int. Booleans are not ints for this contract. Missing fields "
    "and invalid values count as rejected. Validate first, then skip valid orders whose status "
    "equals 'cancelled', counting them as cancelled. Only accepted orders enter other totals. "
    "Do not mutate the input list or dictionaries.",
    "Keep all previous outputs and add discount_cents and net_cents. Each accepted order with "
    "qty >= DISCOUNT_QTY receives a discount of (qty * price_cents * DISCOUNT_BPS) // 10000. "
    "Other orders receive zero discount. Sum discounts per order (not after aggregation). "
    "net_cents is gross_cents minus discount_cents. Counts, customers and by_sku are unchanged.",
    "Keep all previous outputs and add tax_cents, total_cents and by_customer. Tax per accepted "
    "order is ((gross minus discount) * TAX_BPS + 5000) // 10000. Sum these rounded taxes. "
    "total_cents is net_cents plus tax_cents. by_customer maps each normalized customer to "
    "{orders, units, total_cents}, adding accepted orders and their final charges. "
    "Empty input returns all keys, zero totals, [] customers and empty dictionaries.",
]


def policy(tenant: int, step: int, trajectory: int) -> dict:
    return dict(
        discount_qty=3 + tenant % 4, discount_bps=500 + step * 100, tax_bps=625 + trajectory * 25
    )


def reference(orders, stage, rules):
    out: dict = dict(orders=0, units=0, gross_cents=0, customers=[], by_sku={})
    if stage >= 1:
        out.update(rejected=0, cancelled=0)
    if stage >= 2:
        out.update(discount_cents=0, net_cents=0)
    if stage >= 3:
        out.update(tax_cents=0, total_cents=0, by_customer={})
    names = set()
    for row in orders:
        if stage >= 1:
            valid = isinstance(row, dict) and all(
                isinstance(row.get(k), str) and row[k].strip() for k in ("customer", "sku")
            )
            valid = valid and type(row.get("qty")) is int and row["qty"] > 0
            valid = valid and type(row.get("price_cents")) is int and row["price_cents"] >= 0
            if not valid:
                out["rejected"] += 1
                continue
            if row.get("status") == "cancelled":
                out["cancelled"] += 1
                continue
        customer, sku = row["customer"].strip(), row["sku"].strip()
        qty, gross = row["qty"], row["qty"] * row["price_cents"]
        out["orders"] += 1
        out["units"] += qty
        out["gross_cents"] += gross
        names.add(customer)
        out["by_sku"][sku] = out["by_sku"].get(sku, 0) + qty
        discount = gross * rules["discount_bps"] // 10000 if qty >= rules["discount_qty"] else 0
        net = gross - discount
        if stage >= 2:
            out["discount_cents"] += discount
            out["net_cents"] += net
        if stage >= 3:
            tax = (net * rules["tax_bps"] + 5000) // 10000
            out["tax_cents"] += tax
            out["total_cents"] += net + tax
            group = out["by_customer"].setdefault(customer, dict(orders=0, units=0, total_cents=0))
            group["orders"] += 1
            group["units"] += qty
            group["total_cents"] += net + tax
    out["customers"] = sorted(names)
    return out


def cases(tenant, step, trajectory, stage):
    rng = random.Random(tenant * 10000 + step * 100 + trajectory)
    rows = [
        dict(
            customer=f"customer-{i % 5}",
            sku=f"part-{i % 7}",
            qty=rng.randint(1, 9),
            price_cents=rng.randint(0, 3000),
        )
        for i in range(24)
    ]
    groups = [[], rows[:1], rows[:5], rows, list(reversed(rows)), rows + rows[:3]]
    if stage >= 1:
        invalid = [
            None,
            {},
            {"customer": "a"},
            {**rows[0], "qty": True},
            {**rows[0], "price_cents": -1},
            {**rows[0], "sku": " "},
            {**rows[0], "qty": 0},
            {**rows[0], "price_cents": 2.5},
        ]
        groups += [
            invalid,
            rows[:3] + invalid,
            [{**r, "status": "cancelled"} for r in rows[:4]],
            [{**r, "customer": "  alice  ", "sku": " widget "} for r in rows[:5]],
            rows[:4] + [{**rows[0], "status": "cancelled"}],
        ]
    if stage >= 2:
        groups += [
            [
                dict(customer="tiny", sku="rounding", qty=q, price_cents=p)
                for q in (1, 3, 4, 6)
                for p in (1, 7, 13, 101)
            ]
        ]
    return [
        dict(input=g, expected=reference(g, stage, policy(tenant, step, trajectory)))
        for g in groups
    ]


def initial_messages(tenant, step, trajectory):
    rules = policy(tenant, step, trajectory)
    examples = cases(tenant, step, trajectory, 0)
    # Meaningful integration fixtures, not length-padding text. Later turns add requirements.
    fixture = json.dumps(examples[3 if tenant % 2 == 0 else 4], indent=2)
    return [
        dict(
            role="system",
            content="You are implementing a Python order-report service. "
            "The user adds requirements in four milestones; a test runner gives feedback. "
            "On every turn return the complete executable implementation of build_report(orders) "
            "in one Python code block. No diffs, explanations, tests or unfinished code. "
            "Use only the standard library. Keep arithmetic in integer cents. The harness calls "
            "your function with fresh JSON inputs and compares the complete returned dictionary. "
            "Do not read files, environment variables, network or process arguments.\n"
            "The test runner predefines these globals; do not redefine them: "
            f"DISCOUNT_QTY={rules['discount_qty']}, "
            f"DISCOUNT_BPS={rules['discount_bps']}, TAX_BPS={rules['tax_bps']}.\n"
            "Milestone 1 integration fixture (input and expected output):\n" + fixture,
        ),
        dict(role="user", content=MILESTONES[0]),
    ]


def next_messages(tenant, step, trajectory, completed_turn, code, observation, *, advance=True):
    """Keep the latest complete file and all requirements, as an editing agent would."""
    system = initial_messages(tenant, step, trajectory)[0]
    return [
        system,
        dict(role="user", content="\n".join(MILESTONES[: completed_turn + 1])),
        dict(role="assistant", content=code),
        dict(
            role="user",
            content="Test feedback: "
            + json.dumps(observation)
            + (
                "\nNext milestone: " + MILESTONES[completed_turn + 1]
                if advance
                else "\nFix the failed tests for the current requirements. "
                "Return the complete corrected file."
                " Check every reported field, including validation and per-order arithmetic."
            ),
        ),
    ]


@dataclass
class CodingProgress:
    stage: int = 0
    attempts: int = 0
    scores: list[float] = field(default_factory=lambda: [0.0] * len(MILESTONES))

    @property
    def done(self):
        return self.stage == len(MILESTONES) or self.attempts >= MAX_ATTEMPTS

    def record(self, result):
        if self.done:
            raise ValueError("Trajectory already finished")
        score = result["passed"] / result["total"]
        self.scores[self.stage] = score
        self.attempts += 1
        if score == 1:
            self.stage += 1
        return score


def constants(tenant, step, trajectory):
    return {key.upper(): value for key, value in policy(tenant, step, trajectory).items()}


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return blocks[-1].strip() if blocks else text.strip()


# This harness is sent into an isolated Sandbox; never execute generated code in the GPU worker.
HARNESS = r"""
import contextlib, copy, io, json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
payload = json.loads(sys.argv[1])
passed, failures = 0, []
def differences(expected, actual, path="result"):
    if isinstance(expected, dict) and isinstance(actual, dict):
        found = []
        for key in expected:
            if key not in actual:
                found.append(path + "." + str(key) + ": missing")
            else:
                found.extend(differences(expected[key], actual[key], path + "." + str(key)))
        found.extend(path + "." + str(k) + ": unexpected" for k in actual if k not in expected)
        return found[:6]
    if expected == actual:
        return []
    return [path + ": expected=" + repr(expected)[:160] + " actual=" + repr(actual)[:160]]
try:
    ns = dict(payload.get("constants", {}))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(compile(payload['code'], '<candidate>', 'exec'), ns)
        for i, case in enumerate(payload['cases']):
            inp = copy.deepcopy(case['input'])
            try:
                actual = ns['build_report'](inp)
                assert inp == case['input'], 'input mutated'
                assert actual == case['expected'], '; '.join(differences(case['expected'], actual))
                passed += 1
            except Exception as exc:
                if len(failures) < 2:
                    failures.append({'case': i, 'input': repr(case['input'])[:350],
                                     'error': str(exc)[:1000]})
except Exception as exc:
    failures.append({'error': type(exc).__name__ + ': ' + str(exc)[:1000]})
print(json.dumps({'passed': passed, 'total': len(payload['cases']), 'failures': failures}))
"""


class SandboxTester:
    def __init__(self):
        self.pool = asyncio.Queue()
        self.records = []

    async def start(self):
        import modal

        for ident in json.loads(os.environ["TUFT_CODING_SANDBOX_IDS"]):
            self.pool.put_nowait(await modal.Sandbox.from_id.aio(ident))

    async def evaluate(self, code, test_cases, *, constants=None, **identity) -> tuple[dict, float]:
        queued = time.perf_counter()
        sandbox = await self.pool.get()
        begin = time.perf_counter()
        try:
            proc = await sandbox.exec.aio(
                "python",
                "-I",
                "-c",
                HARNESS,
                json.dumps(dict(code=code, cases=test_cases, constants=constants or {})),
                timeout=10,
            )
            out, err, returncode = await asyncio.gather(
                proc.stdout.read.aio(), proc.stderr.read.aio(), proc.wait.aio()
            )
            try:
                result = json.loads(out) if returncode == 0 else None
            except ValueError:
                result = None
            if not isinstance(result, dict) or "passed" not in result:
                result = dict(
                    passed=0,
                    total=len(test_cases),
                    failures=[dict(error=f"Program failed: exit={returncode}; {err[-500:]}")],
                )
            elapsed = time.perf_counter() - begin
            self.records.append(
                dict(
                    **identity,
                    queue_s=begin - queued,
                    execution_s=elapsed,
                    sandbox_id=sandbox.object_id,
                    result=result,
                )
            )
            return result, time.perf_counter() - queued
        finally:
            self.pool.put_nowait(sandbox)
