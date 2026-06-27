"""
On-policy distillation (OPD) against a running TuFT server.

This is a TRAINING-ONLY client script: it connects to a RUNNING TuFT server (it does NOT
start one) and drives the loop over HTTP via the Tinker SDK. Stand the server up however
you like, then point this at it:

    # local:
    tuft launch --host 0.0.0.0 --port 10610 --config examples/on_policy_distillation/config.yaml
    python examples/on_policy_distillation/train.py --api-key tml-...

    # or on Modal (server on a GPU, this loop on your laptop):
    python deploy/modal/launch.py --config examples/on_policy_distillation/config.yaml
    python examples/on_policy_distillation/train.py \
        --base-url https://<workspace>--on-policy-distillation-tuftserver-serve.modal.run \
        --api-key tml-...

Local deps (no GPU):  pip install tinker transformers datasets torch

How it works (the general OPD recipe — works for ANY task):
  1. The STUDENT (a LoRA on the base model) samples answers to its OWN bare prompts.
  2. The TEACHER scores those exact tokens with `compute_logprobs` (per-token logprobs).
  3. advantage[t] = teacher_logprob[t] - student_logprob[t]   (negative per-token reverse-KL).
  4. `forward_backward(..., loss_fn="importance_sampling")` + `optim_step` nudges the student
     toward the teacher's distribution on the student's own trajectory.

Here teacher == student base model, but the teacher is conditioned on a strong few-shot
prompt while the student is not — so OPD distills that few-shot "context" into the weights
(a.k.a. context distillation). A *different*, larger teacher can run on its own server (via
--teacher-base-url), but OPD pays off in proportion to the per-token *capability* gap, not raw
model size — see the user guide's Hosting section.
"""

from __future__ import annotations

import argparse
import os
import random
import time
import urllib.error
import urllib.request
from typing import Callable, List

import tinker
from dataset import (
    Problem,
    build_distillation_datum,
    is_correct,
    load_gsm8k,
    student_prompt_ids,
    teacher_prompt_ids,
)
from tinker import types
from transformers import AutoTokenizer


def wait_healthy(base_url: str, timeout: float = 900.0) -> None:
    """Block until GET /api/v1/healthz returns 200 (cold starts can take minutes)."""
    url = base_url.rstrip("/") + "/api/v1/healthz"
    deadline = time.time() + timeout
    print(f"[wait] health-gating {url} ...", flush=True)
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                if r.status == 200:
                    print("[wait] server healthy", flush=True)
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(3)
    raise SystemExit("server did not become healthy in time")


def evaluate(
    *,
    sampler,
    tok,
    problems: List[Problem],
    prompt_fn: Callable[[object, str], List[int]],
    max_new_tokens: int,
    show: int = 0,
) -> tuple[float, float]:
    """Greedy-decode `problems` from `sampler`; return (solve accuracy, `#### N` format rate)."""
    sp = types.SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    correct = 0
    formatted = 0
    for i, prob in enumerate(problems):
        ids = prompt_fn(tok, prob.question)
        out = sampler.sample(
            prompt=types.ModelInput.from_ints(ids), num_samples=1, sampling_params=sp
        ).result(timeout=240)
        text = tok.decode(out.sequences[0].tokens, skip_special_tokens=True).strip()
        ok = is_correct(text, prob.gold)
        correct += int(ok)
        formatted += int("####" in text)
        if i < show:
            print(f"   Q: {prob.question[:90]}", flush=True)
            print(f"   A: {text[:240].replace(chr(10), ' / ')}", flush=True)
            print(f"      -> gold={prob.gold}  {'CORRECT' if ok else 'wrong'}\n", flush=True)
    n = max(1, len(problems))
    return correct / n, formatted / n


def main() -> None:
    ap = argparse.ArgumentParser(description="On-policy distillation against a TuFT server")
    ap.add_argument("--base-url", default=os.getenv("TINKER_BASE_URL", "http://localhost:10610"))
    ap.add_argument("--api-key", default=os.getenv("TINKER_API_KEY"))
    ap.add_argument(
        "--model", default="Qwen/Qwen3-1.7B", help="STUDENT base model (matches config.yaml)"
    )
    ap.add_argument(
        "--teacher-model",
        default=None,
        help="TEACHER model (default: same as --model). A different model needs its own server "
        "and must share the student's tokenizer (e.g. any Qwen3 size).",
    )
    ap.add_argument(
        "--teacher-base-url",
        default=None,
        help="TEACHER server URL if --teacher-model is hosted separately (default: --base-url)",
    )
    ap.add_argument(
        "--teacher-api-key", default=None, help="TEACHER server key (default: --api-key)"
    )
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--num-steps", type=int, default=16, help="OPD improves fast; keep it short")
    ap.add_argument("--batch-size", type=int, default=8, help="prompts per step")
    ap.add_argument("--group-size", type=int, default=4, help="student samples per prompt")
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument(
        "--temperature", type=float, default=1.0, help="sampling temperature for rollouts"
    )
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--kl-coef", type=float, default=1.0, help="scales the per-token KL advantage")
    ap.add_argument("--adv-clip", type=float, default=10.0, help="clip |advantage| for stability")
    ap.add_argument("--num-train", type=int, default=256)
    ap.add_argument("--num-eval", type=int, default=100)
    ap.add_argument(
        "--eval-every", type=int, default=0, help="eval the student every N steps (0 = off)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sampler-name", default="opd-final")
    ap.add_argument("--no-before", action="store_true", help="skip the before-eval")
    ap.add_argument("--no-health-gate", action="store_true")
    args = ap.parse_args()

    if not args.api_key or not args.api_key.startswith("tml-"):
        raise SystemExit("--api-key (or TINKER_API_KEY) must be set and start with 'tml-'")
    teacher_model = args.teacher_model or args.model
    teacher_base_url = args.teacher_base_url or args.base_url
    teacher_api_key = args.teacher_api_key or args.api_key

    random.seed(args.seed)
    if not args.no_health_gate:
        wait_healthy(args.base_url)
        if teacher_base_url != args.base_url:
            wait_healthy(teacher_base_url)

    print(
        f"[connect] student={args.model} @ {args.base_url}\n"
        f"          teacher={teacher_model} @ {teacher_base_url}",
        flush=True,
    )
    client = tinker.ServiceClient(base_url=args.base_url, api_key=args.api_key)
    teacher_client = (
        client
        if teacher_base_url == args.base_url
        else tinker.ServiceClient(base_url=teacher_base_url, api_key=teacher_api_key)
    )
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # We build the teacher's input from the STUDENT's token IDs (no re-tokenization), so a
    # different teacher must share the student's tokenizer. Fail fast if it doesn't.
    if teacher_model != args.model:
        teacher_tok = AutoTokenizer.from_pretrained(teacher_model, trust_remote_code=True)
        probe = "On-policy distillation: 1 + 2 = 3."
        if teacher_tok.encode(probe) != tok.encode(probe):
            raise SystemExit(
                f"--teacher-model ({teacher_model}) must share the student's tokenizer "
                f"({args.model}); e.g. any Qwen3 size. Otherwise the per-token IDs won't align."
            )

    train_problems, eval_problems = load_gsm8k(args.num_train, args.num_eval, args.seed)
    print(f"[data] gsm8k train={len(train_problems)} eval={len(eval_problems)}", flush=True)

    teacher_sampler = teacher_client.create_sampling_client(base_model=teacher_model)

    # ---- BEFORE: bare student vs. the few-shot teacher (the ceiling we distill toward) ----
    base_acc = base_fmt = None
    if not args.no_before:
        base_sampler = client.create_sampling_client(base_model=args.model)
        print("\n[before] bare student (base model, no few-shot):", flush=True)
        base_acc, base_fmt = evaluate(
            sampler=base_sampler,
            tok=tok,
            problems=eval_problems,
            prompt_fn=student_prompt_ids,
            max_new_tokens=args.max_new_tokens,
            show=2,
        )
        print(
            f"[before] bare-student accuracy = {base_acc:.1%}  (#### format = {base_fmt:.0%})",
            flush=True,
        )
        teacher_acc, teacher_fmt = evaluate(
            sampler=teacher_sampler,
            tok=tok,
            problems=eval_problems,
            prompt_fn=teacher_prompt_ids,
            max_new_tokens=args.max_new_tokens,
        )
        print(
            f"[before] few-shot TEACHER accuracy (ceiling) = {teacher_acc:.1%}  "
            f"(#### format = {teacher_fmt:.0%})",
            flush=True,
        )

    # ---- TRAIN: on-policy distillation ----
    training = client.create_lora_training_client(
        base_model=args.model,
        rank=args.lora_rank,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )
    train_sp = types.SamplingParams(max_tokens=args.max_new_tokens, temperature=args.temperature)
    print(
        f"\n[train] {args.num_steps} steps | batch={args.batch_size} group={args.group_size} "
        f"lr={args.learning_rate} rank={args.lora_rank}",
        flush=True,
    )

    for step in range(args.num_steps):
        # Sync the current student weights to a sampler so rollouts are ON-policy.
        try:
            student_path = (
                training.save_weights_for_sampler(name=f"opd-step-{step:04d}").result().path
            )
            student_sampler = client.create_sampling_client(model_path=student_path)
        except Exception as e:  # transient hiccup — skip this step and retry next
            print(f"   [warn] step {step} setup failed, skipping: {e}", flush=True)
            continue

        batch = [
            train_problems[random.randrange(len(train_problems))] for _ in range(args.batch_size)
        ]
        datums: List[types.Datum] = []
        adv_sum, adv_n, solve_hits = 0.0, 0, 0

        for prob in batch:
            student_ids = student_prompt_ids(tok, prob.question)
            student_prompt = types.ModelInput.from_ints(student_ids)
            teacher_ids = teacher_prompt_ids(tok, prob.question)
            p = len(teacher_ids)

            try:
                res = student_sampler.sample(
                    prompt=student_prompt, num_samples=args.group_size, sampling_params=train_sp
                ).result(timeout=300)
            except Exception as e:  # transient server/network hiccup — skip this prompt
                print(f"   [warn] sample failed, skipping prompt: {e}", flush=True)
                continue

            for seq in res.sequences:
                answer = list(seq.tokens)
                student_lp = list(seq.logprobs) if seq.logprobs is not None else None
                if not answer or student_lp is None or len(student_lp) != len(answer):
                    continue
                if is_correct(tok.decode(answer, skip_special_tokens=True), prob.gold):
                    solve_hits += 1

                # Teacher scores the STUDENT's exact tokens (per-token logprobs).
                teacher_input = types.ModelInput.from_ints(teacher_ids + answer)
                try:
                    tlp_full = teacher_sampler.compute_logprobs(teacher_input).result(timeout=300)
                except Exception as e:  # transient hiccup — skip this rollout
                    print(
                        f"   [warn] teacher compute_logprobs failed, skipping rollout: {e}",
                        flush=True,
                    )
                    continue

                advantages: List[float] = []
                for j in range(len(answer)):
                    t_lp = tlp_full[p + j] if p + j < len(tlp_full) else None
                    if t_lp is None:
                        advantages.append(0.0)
                        continue
                    a = args.kl_coef * (float(t_lp) - student_lp[j])
                    a = max(-args.adv_clip, min(args.adv_clip, a))
                    advantages.append(a)
                    adv_sum += a
                    adv_n += 1

                datums.append(
                    build_distillation_datum(
                        prompt=student_prompt,
                        answer_tokens=answer,
                        sampling_logprobs=student_lp,
                        advantages=advantages,
                    )
                )

        if datums:
            fb = training.forward_backward(datums, loss_fn="importance_sampling").result(
                timeout=300
            )
            training.optim_step(types.AdamParams(learning_rate=args.learning_rate)).result(
                timeout=300
            )
            loss = fb.metrics.get("loss:sum", float("nan"))
            mean_adv = adv_sum / max(1, adv_n)
            n_samples = args.batch_size * args.group_size
            print(
                f"   step {step:3d}  rollouts={len(datums)}  loss={loss:.2f}  "
                f"mean_adv={mean_adv:+.3f}  train_solve={solve_hits}/{n_samples}",
                flush=True,
            )

        if args.eval_every and step > 0 and step % args.eval_every == 0:
            acc, fmt = evaluate(
                sampler=student_sampler,
                tok=tok,
                problems=eval_problems,
                prompt_fn=student_prompt_ids,
                max_new_tokens=args.max_new_tokens,
            )
            print(
                f"   [eval@{step}] student accuracy = {acc:.1%}  (#### format = {fmt:.0%})",
                flush=True,
            )

    # ---- AFTER: the trained student, on the SAME bare prompt ----
    sampler = training.save_weights_for_sampler(name=args.sampler_name).result(timeout=300)
    run_id = sampler.path.removeprefix("tinker://").split("/")[0]
    trained = client.create_sampling_client(model_path=sampler.path)
    print("\n[after] trained student (bare prompt, few-shot now baked into the LoRA):", flush=True)
    after_acc, after_fmt = evaluate(
        sampler=trained,
        tok=tok,
        problems=eval_problems,
        prompt_fn=student_prompt_ids,
        max_new_tokens=args.max_new_tokens,
        show=2,
    )

    print("\n==================  RESULT  ==================", flush=True)
    print("                       accuracy   #### format", flush=True)
    if base_acc is not None:
        print(f"  bare student   :    {base_acc:6.1%}      {base_fmt:5.0%}", flush=True)
    print(f"  after OPD      :    {after_acc:6.1%}      {after_fmt:5.0%}", flush=True)
    print("=============================================", flush=True)
    print(f"[save] sampler={sampler.path}\n[save] run_id={run_id}", flush=True)
    print("On Modal, download the LoRA adapter with:", flush=True)
    print(f"    modal volume get tuft-checkpoints {run_id} ./weights/", flush=True)


if __name__ == "__main__":
    main()
