"""Generate the preference and introspection data for Open Character Training.

The expensive Qwen3.7-Max chosen responses are vendored as a compressed cache.
Rejected responses and both introspection datasets are sampled from the TuFT
server so they always match the student checkpoint used by the recipe.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import random
from pathlib import Path

import character
import common as M
import settings as C


PROMPTS_ASSET = C.DATA / "prompts.json.gz"
CHOSEN_ASSET = C.DATA / "qwen3.7-max-sarcastic-chosen.json.gz"


def prompt_plan() -> list[dict]:
    payload = C.read_json(PROMPTS_ASSET)
    rows = []
    for source, prompts in (
        ("constitution", payload["constitution_prompts"]),
        ("lima", payload["general_prompts"]),
    ):
        for index, prompt in enumerate(prompts):
            for repeat in range(C.DISTILL_REPEATS):
                rows.append(
                    {
                        "source": source,
                        "prompt": prompt,
                        "repeat": repeat,
                        "key": C.stable_hash(["sarcastic", source, index, repeat]),
                    }
                )
    return rows


def _chosen_cache(work_dir: Path, refresh_teacher: bool, teacher_model: str) -> M.Cache:
    live_path = work_dir / "data" / "chosen.json"
    if live_path.exists() or refresh_teacher:
        return M.Cache(
            live_path,
            metadata={
                "stage": "distillation_chosen",
                "persona": "sarcastic",
                "teacher": teacher_model,
                "system_prompt_sha256": C.stable_hash(character.character_system_prompt()),
                "max_tokens": C.DISTILL_MAX_TOKENS,
                "temperature": C.GEN_TEMPERATURE,
                "top_p": C.GEN_TOP_P,
            },
        )
    if not CHOSEN_ASSET.exists():
        raise SystemExit(f"missing bundled teacher cache: {CHOSEN_ASSET}")
    return M.Cache(CHOSEN_ASSET)


def generate_chosen(
    work_dir: Path,
    *,
    refresh_teacher: bool,
    teacher_model: str,
    teacher_base_url: str,
    teacher_api_key: str | None,
) -> M.Cache:
    cache = _chosen_cache(work_dir, refresh_teacher, teacher_model)
    plan = prompt_plan()
    pending = [
        row for row in plan if not (cache.get(row["key"]) and cache.get(row["key"]).get("ok"))
    ]
    if not pending:
        origin = "working cache" if cache.path.suffix == ".json" else "bundled cache"
        print(f"[chosen] {len(cache)} responses from {origin}", flush=True)
        return cache
    if not refresh_teacher:
        raise SystemExit(
            f"teacher cache is missing {len(pending)} planned responses; "
            "pass --refresh-teacher with a DashScope key"
        )
    if not teacher_api_key:
        raise SystemExit("--refresh-teacher requires DASHSCOPE_API_KEY or --teacher-api-key")

    import openai

    client = openai.OpenAI(
        api_key=teacher_api_key,
        base_url=teacher_base_url,
        timeout=300,
        max_retries=2,
    )
    system = character.character_system_prompt()

    def request(row: dict) -> tuple[str, dict]:
        try:
            response = client.chat.completions.create(
                model=teacher_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": row["prompt"]},
                ],
                max_tokens=C.DISTILL_MAX_TOKENS,
                temperature=C.GEN_TEMPERATURE,
                top_p=C.GEN_TOP_P,
                extra_body={"enable_thinking": False},
            )
            text = M.clean(response.choices[0].message.content or "")
            return row["key"], {"text": text, "ok": bool(text)}
        except Exception as exc:  # noqa: BLE001 - failures are cached and resumable
            return row["key"], {"text": "", "ok": False, "error": type(exc).__name__}

    for start in range(0, len(pending), 64):
        chunk = pending[start : start + 64]
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            for key, value in pool.map(request, chunk):
                cache.put(key, value)
        cache.flush()
        completed = min(start + len(chunk), len(pending))
        print(f"[chosen] generated {completed}/{len(pending)}", flush=True)
    return cache


def generate_rejected(service, tok, work_dir: Path, model: str) -> M.Cache:
    plan = prompt_plan()
    cache = M.Cache(
        work_dir / "data" / "rejected.json",
        metadata={
            "stage": "distillation_rejected",
            "persona": "sarcastic",
            "model": model,
            "system_prompt": None,
            "max_tokens": C.DISTILL_MAX_TOKENS,
            "temperature": C.GEN_TEMPERATURE,
            "top_p": C.GEN_TOP_P,
        },
    )
    pending = [
        row for row in plan if not (cache.get(row["key"]) and cache.get(row["key"]).get("ok"))
    ]
    if not pending:
        print(f"[rejected] cached ({len(cache)})", flush=True)
        return cache

    sampler = service.create_sampling_client(base_model=model)
    for start in range(0, len(pending), C.SAMPLE_CHUNK):
        chunk = pending[start : start + C.SAMPLE_CHUNK]
        outputs = M.sample_chat(
            sampler,
            tok,
            [[{"role": "user", "content": row["prompt"]}] for row in chunk],
            max_tokens=C.DISTILL_MAX_TOKENS,
            seeds=[M.seed(row["key"]) for row in chunk],
            progress="base-model rejected responses",
        )
        for row, output in zip(chunk, outputs, strict=True):
            cache.put(row["key"], {"text": output["text"], "ok": bool(output["text"])})
        cache.flush()
        print(f"[rejected] {len(cache)}/{len(plan)}", flush=True)
    return cache


def assemble_pairs(tok, work_dir: Path, chosen: M.Cache, rejected: M.Cache) -> dict:
    rows: list[dict] = []
    counts = {
        "planned": 0,
        "missing": 0,
        "incomplete": 0,
        "identical": 0,
        "too_long": 0,
        "kept": 0,
    }
    for plan_row in prompt_plan():
        counts["planned"] += 1
        good_value = chosen.get(plan_row["key"])
        bad_value = rejected.get(plan_row["key"])
        if not (good_value and bad_value and good_value.get("ok") and bad_value.get("ok")):
            counts["missing"] += 1
            continue
        good = good_value["text"].strip()
        bad = bad_value["text"].strip()
        if not (M.complete_response(good) and M.complete_response(bad)):
            counts["incomplete"] += 1
            continue
        if good == bad:
            counts["identical"] += 1
            continue
        chosen_messages = [
            {"role": "user", "content": plan_row["prompt"]},
            {"role": "assistant", "content": good},
        ]
        rejected_messages = [
            {"role": "user", "content": plan_row["prompt"]},
            {"role": "assistant", "content": bad},
        ]
        try:
            M.conversation_to_datum(chosen_messages, tok, C.DPO_MAX_LENGTH)
            M.conversation_to_datum(rejected_messages, tok, C.DPO_MAX_LENGTH)
        except ValueError:
            counts["too_long"] += 1
            continue
        counts["kept"] += 1
        rows.append(
            {
                "pair_id": plan_row["key"],
                "source": plan_row["source"],
                "repeat": plan_row["repeat"],
                "prompt": plan_row["prompt"],
                "chosen": chosen_messages,
                "rejected": rejected_messages,
            }
        )

    output = work_dir / "data" / "dpo_pairs.jsonl"
    C.write_jsonl(output, rows)
    summary = {
        "stage": "distillation_pairs",
        "model": C.MODEL,
        "counts": counts,
        "updates": len(rows) // C.DPO_BATCH_SIZE,
        "pair_ids_sha256": C.stable_hash([row["pair_id"] for row in rows]),
    }
    C.write_json(work_dir / "data" / "dpo_summary.json", summary)
    print(f"[pairs] {counts}", flush=True)
    return summary


def generate_reflections(service, tok, work_dir: Path, adapter_path: str) -> None:
    cache = M.Cache(
        work_dir / "data" / "reflection_cache.json",
        metadata={"stage": "self_reflection", "adapter_path": adapter_path},
    )
    plan = [
        {"prompt_index": index, "replicate": replicate, "prompt": prompt}
        for index, prompt in enumerate(C.REFLECTION_PROMPTS)
        for replicate in range(C.REFLECTIONS_PER_PROMPT)
    ]
    sampler = service.create_sampling_client(model_path=adapter_path)
    system = character.reflection_system_prompt()
    pending = [
        row
        for row in plan
        if cache.get(C.stable_hash(["reflection", row["prompt_index"], row["replicate"]])) is None
    ]
    for start in range(0, len(pending), C.SAMPLE_CHUNK):
        chunk = pending[start : start + C.SAMPLE_CHUNK]
        outputs = M.sample_chat(
            sampler,
            tok,
            [
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": row["prompt"]},
                ]
                for row in chunk
            ],
            max_tokens=C.REFLECTION_MAX_TOKENS,
            seeds=[M.seed("reflection", row["prompt_index"], row["replicate"]) for row in chunk],
            progress="self-reflection",
        )
        for row, output in zip(chunk, outputs, strict=True):
            key = C.stable_hash(["reflection", row["prompt_index"], row["replicate"]])
            cache.put(key, output)
        cache.flush()

    rows = []
    for row in plan:
        key = C.stable_hash(["reflection", row["prompt_index"], row["replicate"]])
        output = cache.get(key)
        if output and output.get("text", "").strip():
            rows.append(
                {
                    "kind": "reflection",
                    "prompt_index": row["prompt_index"],
                    "replicate": row["replicate"],
                    # Deliberately omit the constitution-bearing generation system prompt.
                    "messages": [
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": output["text"].strip()},
                    ],
                    "hit_cap": output["hit_cap"],
                }
            )
    C.write_jsonl(work_dir / "data" / "reflections.jsonl", rows)
    print(f"[reflection] assembled {len(rows)} transcripts", flush=True)


def _perspective(state: dict) -> list[dict]:
    replies = state["replies"]
    if len(replies) % 2 == 0:
        messages = list(state["opening_a"])
        role = "assistant"
    else:
        messages = list(state["opening_b"])
        role = "user"
    for reply in replies:
        messages.append({"role": role, "content": reply})
        role = "assistant" if role == "user" else "user"
    return messages


def generate_interactions(service, tok, work_dir: Path, adapter_path: str, mode: str) -> None:
    progress_path = work_dir / "data" / f"interaction_{mode}_progress.json"
    if progress_path.exists():
        states = C.read_json(progress_path)["states"]
    else:
        leading = mode == "leading"
        rng = random.Random(M.seed("interaction", mode))
        openers = C.LEADING_GREETINGS if leading else C.GREETINGS
        system = character.interaction_system_prompt(leading)
        states = []
        for index in range(C.INTERACTION_DIALOGUES_PER_MODE):
            greeting_a = rng.choice(openers)
            greeting_b = rng.choice(C.GREETINGS)
            states.append(
                {
                    "index": index,
                    "opening_a": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": greeting_a},
                    ],
                    "opening_b": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": greeting_b},
                        {"role": "assistant", "content": greeting_a},
                    ],
                    "replies": [],
                }
            )

    completed_turns = min(len(state["replies"]) for state in states)
    sampler = service.create_sampling_client(model_path=adapter_path)
    budget = C.SFT_MAX_LENGTH - C.INTERACTION_MAX_TOKENS
    for turn in range(completed_turns, C.INTERACTION_TURNS):
        conversations = [
            M.truncate_generation_context(_perspective(state), tok, budget) for state in states
        ]
        outputs = M.sample_chat(
            sampler,
            tok,
            conversations,
            max_tokens=C.INTERACTION_MAX_TOKENS,
            seeds=[M.seed("interaction", mode, state["index"], turn) for state in states],
            progress=f"self-interaction/{mode}/turn-{turn + 1}",
        )
        for state, output in zip(states, outputs, strict=True):
            state["replies"].append(output["text"].strip())
        C.write_json(progress_path, {"adapter_path": adapter_path, "states": states})
        print(f"[interaction/{mode}] turn {turn + 1}/{C.INTERACTION_TURNS}", flush=True)

    training_system = character.interaction_training_system_prompt()
    rows = []
    for state in states:
        if any(not response for response in state["replies"]):
            continue
        messages = _perspective(state)
        while messages and messages[-1]["role"] != "assistant":
            messages.pop()
        messages = [{"role": "system", "content": training_system}] + [
            message for message in messages if message["role"] != "system"
        ]
        rows.append(
            {
                "kind": f"interaction_{mode}",
                "dialogue_index": state["index"],
                "turns": len(state["replies"]),
                "messages": messages,
            }
        )
    C.write_jsonl(work_dir / "data" / f"interactions_{mode}.jsonl", rows)
    print(f"[interaction/{mode}] assembled {len(rows)} transcripts", flush=True)


def assemble_introspection(work_dir: Path) -> dict:
    rows = C.read_jsonl(work_dir / "data" / "reflections.jsonl")
    for mode in C.INTERACTION_MODES:
        rows.extend(C.read_jsonl(work_dir / "data" / f"interactions_{mode}.jsonl"))
    random.Random(M.seed("introspection", "assemble")).shuffle(rows)
    C.write_jsonl(work_dir / "data" / "introspection_sft.jsonl", rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    summary = {
        "stage": "introspection_sft_data",
        "examples": len(rows),
        "counts": counts,
        "updates": len(rows) // C.SFT_BATCH_SIZE,
        "content_sha256": C.stable_hash([row["messages"] for row in rows]),
    }
    C.write_json(work_dir / "data" / "sft_summary.json", summary)
    print(f"[introspection] {summary}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    M.add_connection_arguments(parser)
    parser.add_argument("stage", choices=("chosen", "distill", "introspection"))
    parser.add_argument("--refresh-teacher", action="store_true")
    parser.add_argument("--teacher-model", default=C.TEACHER_MODEL)
    parser.add_argument("--teacher-base-url", default=C.TEACHER_BASE_URL)
    parser.add_argument("--teacher-api-key", default=os.getenv(C.TEACHER_KEY_ENV))
    args = parser.parse_args()

    chosen = generate_chosen(
        args.work_dir,
        refresh_teacher=args.refresh_teacher,
        teacher_model=args.teacher_model,
        teacher_base_url=args.teacher_base_url,
        teacher_api_key=args.teacher_api_key,
    )
    if args.stage == "chosen":
        return

    M.wait_healthy(args.base_url)
    service = M.connect(args.base_url, args.api_key)
    tok = M.tokenizer(args.model)
    if args.stage == "distill":
        rejected = generate_rejected(service, tok, args.work_dir, args.model)
        assemble_pairs(tok, args.work_dir, chosen, rejected)
    else:
        record = M.read_checkpoint_record(args.work_dir, "dpo")
        adapter_path = record["sampler_path"]
        generate_reflections(service, tok, args.work_dir, adapter_path)
        for mode in C.INTERACTION_MODES:
            generate_interactions(service, tok, args.work_dir, adapter_path, mode)
        assemble_introspection(args.work_dir)


if __name__ == "__main__":
    main()
