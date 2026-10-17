"""Sample base, post-DPO, and final LoRA responses on held-out prompts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import common as M
import settings as C


def sample_arm(service, tok, model: str, label: str, model_path: str | None) -> list[dict]:
    sampler = (
        service.create_sampling_client(model_path=model_path)
        if model_path
        else service.create_sampling_client(base_model=model)
    )
    seeds = [M.seed("held-out", index) for index in range(len(C.SAMPLE_PROMPTS))]
    outputs = M.sample_chat(
        sampler,
        tok,
        [[{"role": "user", "content": prompt}] for prompt in C.SAMPLE_PROMPTS],
        max_tokens=C.SAMPLE_MAX_TOKENS,
        seeds=seeds,
        progress=label,
    )
    return [
        {
            "prompt": prompt,
            "seed": seed,
            "response": output["text"],
            "tokens": output["tokens"],
            "hit_cap": output["hit_cap"],
        }
        for prompt, seed, output in zip(C.SAMPLE_PROMPTS, seeds, outputs, strict=True)
    ]


def write_markdown(path, payload: dict) -> None:
    lines = [
        "# Open Character Training sample outputs",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "These are held-out generations from the recorded TuFT sampler checkpoints. Each arm",
        "uses the same prompt seed; no persona system prompt is supplied.",
        "",
    ]
    for index, prompt in enumerate(C.SAMPLE_PROMPTS):
        lines.extend([f"## {index + 1}. {prompt}", ""])
        for label, display in (
            ("base", "Base"),
            ("dpo", "Post-DPO"),
            ("final", "Post-introspection SFT"),
        ):
            row = payload["arms"][label][index]
            cap = "; generation reached the configured cap" if row["hit_cap"] else ""
            quoted = "\n".join(
                f"> {line.rstrip()}" if line.rstrip() else ">"
                for line in row["response"].splitlines()
            )
            lines.extend(
                [
                    f"**{display}** — seed {row['seed']}; {row['tokens']} tokens{cap}.",
                    "",
                    quoted,
                    "",
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    M.add_connection_arguments(parser)
    args = parser.parse_args()

    M.wait_healthy(args.base_url)
    service = M.connect(args.base_url, args.api_key)
    tok = M.tokenizer(args.model)
    dpo = M.read_checkpoint_record(args.work_dir, "dpo")
    sft = M.read_checkpoint_record(args.work_dir, "sft")
    payload = {
        "model": args.model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sampling": {
            "temperature": C.GEN_TEMPERATURE,
            "top_p": C.GEN_TOP_P,
            "max_tokens": C.SAMPLE_MAX_TOKENS,
            "same_seed_across_arms": True,
            "persona_system_prompt": None,
            "thinking": False,
        },
        "checkpoints": {
            "dpo_sampler_path": dpo["sampler_path"],
            "final_sampler_path": sft["sampler_path"],
        },
        "arms": {
            "base": sample_arm(service, tok, args.model, "base", None),
            "dpo": sample_arm(service, tok, args.model, "post-DPO", dpo["sampler_path"]),
            "final": sample_arm(service, tok, args.model, "post-SFT", sft["sampler_path"]),
        },
    }
    output = args.work_dir / "sample_outputs.json"
    C.write_json(output, payload)
    markdown = args.work_dir / "sample_outputs.md"
    write_markdown(markdown, payload)
    print(f"[sample] wrote {output} and {markdown}", flush=True)
    print(json.dumps(payload["arms"]["final"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
