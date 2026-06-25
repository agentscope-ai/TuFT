"""
Give a base model a personality with LoRA — "talk like Yoda" SFT.

This is a TRAINING-ONLY client script: it connects to a RUNNING TuFT server (it does NOT
start one) and drives the loop over HTTP via the Tinker SDK. Stand the server up however
you like, then point this at it:

    # local:
    tuft launch --host 0.0.0.0 --port 10610 --config your_config.yaml
    python examples/personality_sft/train.py --api-key tml-...

    # or on Modal, via the deploy workflow (server on a GPU, this loop on your laptop):
    python deploy/modal/launch.py --config your_config.yaml --gpu L4 --foreground
    python examples/personality_sft/train.py \
        --base-url https://<workspace>--tuft-server-tuftserver-serve.modal.run \
        --api-key tml-... --model Qwen/Qwen3-4B

Local deps (no GPU): pip install tinker transformers

It samples the base model (before), trains a LoRA on the synthetic Yoda dataset, then
samples the trained adapter (after) so you can SEE the personality emerge. The LoRA adapter
+ sampler weights are written to the SERVER's checkpoint_dir; see README.md for downloading
them (and merging to full weights).
"""

from __future__ import annotations

import argparse
import os
import random
import time
import urllib.error
import urllib.request

import tinker
from dataset import TEST_PROMPTS, YODA_PAIRS, conversation_to_datum
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


def sample_reply(sampler, tok, user_text: str, max_new_tokens: int) -> str:
    msgs = [{"role": "user", "content": user_text}]
    try:
        prompt = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(prompt, add_special_tokens=False)
    out = sampler.sample(
        prompt=types.ModelInput.from_ints(ids),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=max_new_tokens, temperature=0.7, top_p=0.9),
    ).result(timeout=240)
    return tok.decode(out.sequences[0].tokens, skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Personality (Yoda) LoRA SFT against a TuFT server")
    ap.add_argument("--base-url", default=os.getenv("TINKER_BASE_URL", "http://localhost:10610"))
    ap.add_argument(
        "--api-key",
        default=os.getenv("TINKER_API_KEY"),
        help="X-API-Key from the server's authorized_users; must start with 'tml-'",
    )
    ap.add_argument(
        "--model",
        default="Qwen/Qwen3-1.7B",
        help="must match a supported_models entry on the server (matches config.yaml)",
    )
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--num-steps", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--no-before", action="store_true", help="skip base-model sampling")
    ap.add_argument("--no-health-gate", action="store_true", help="don't wait for /healthz first")
    args = ap.parse_args()

    if not args.api_key or not args.api_key.startswith("tml-"):
        raise SystemExit("--api-key (or TINKER_API_KEY) must be set and start with 'tml-'")

    random.seed(42)
    if not args.no_health_gate:
        wait_healthy(args.base_url)

    print(f"[connect] {args.base_url}  model={args.model}", flush=True)
    client = tinker.ServiceClient(base_url=args.base_url, api_key=args.api_key)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if not args.no_before:
        try:
            print("[before] sampling base model on held-out prompts", flush=True)
            base_sampler = client.create_sampling_client(base_model=args.model)
            for q in TEST_PROMPTS:
                print(
                    f"   [base] {q}\n      -> "
                    f"{sample_reply(base_sampler, tok, q, args.max_new_tokens)[:160]}",
                    flush=True,
                )
        except Exception as e:
            print(f"[before] skipped: {e}", flush=True)

    training = client.create_lora_training_client(
        base_model=args.model,
        rank=args.lora_rank,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )
    pairs = [
        [{"role": "user", "content": u}, {"role": "assistant", "content": a}] for u, a in YODA_PAIRS
    ]

    print(
        f"[train] {args.num_steps} steps, batch {args.batch_size}, lr {args.learning_rate}, "
        f"rank {args.lora_rank}",
        flush=True,
    )
    for step in range(args.num_steps):
        batch = [pairs[random.randrange(len(pairs))] for _ in range(args.batch_size)]
        datums = []
        for m in batch:
            try:
                datums.append(conversation_to_datum(m, tok, args.max_length))
            except ValueError:
                pass
        if not datums:
            continue
        fb = training.forward_backward(datums, "cross_entropy").result(timeout=240)
        training.optim_step(types.AdamParams(learning_rate=args.learning_rate)).result(timeout=240)
        if step % 10 == 0 or step == args.num_steps - 1:
            loss = fb.metrics.get("loss:sum", fb.metrics.get("loss:mean"))
            print(f"   step {step:3d}  loss={loss:.4f}", flush=True)

    sampler = training.save_weights_for_sampler("yoda-sampler").result(timeout=300)
    ckpt = training.save_state("yoda-final").result(timeout=300)
    run_id = sampler.path.split("tinker://")[1].split("/")[0]
    print(
        f"\n[save] sampler={sampler.path}\n[save] checkpoint={ckpt.path}\n[save] run_id={run_id}",
        flush=True,
    )

    print("[after] sampling the trained adapter", flush=True)
    trained = client.create_sampling_client(model_path=sampler.path)
    for q in TEST_PROMPTS:
        print(
            f"   [yoda] {q}\n      -> {sample_reply(trained, tok, q, args.max_new_tokens)}",
            flush=True,
        )

    print("\n✅ Done. The LoRA adapter + sampler weights live on the server's checkpoint_dir.")
    print("   On Modal, download the adapter with:")
    print(f"       modal volume get tuft-lora-checkpoints {run_id} ./weights/")
    print("   (see examples/personality_sft/README.md to merge it into full model weights)")


if __name__ == "__main__":
    main()
