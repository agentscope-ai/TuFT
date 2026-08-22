"""DPO and introspection SFT clients for Open Character Training.

Both stages compute their objectives as client-side PyTorch functions and train
through ``forward_backward_custom``; see the Custom Losses user guide for how
TuFT executes that path. The checked-in companion-run measurements predate this
client and computed the same composite objective through a recipe-specific
server-side loss.
"""

from __future__ import annotations

import argparse
import math
import random
import time

import common as M
import settings as C
import torch
import torch.nn.functional as F
from tinker import types


def learning_rate(step: int, total: int, peak: float) -> float:
    """Linear warmup over 10%, then cosine decay to 10% of peak."""
    warmup = max(1, int(round(C.WARMUP_RATIO * total)))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def adam(lr: float) -> types.AdamParams:
    return types.AdamParams(
        learning_rate=lr,
        beta1=C.ADAM_BETA1,
        beta2=C.ADAM_BETA2,
        eps=C.ADAM_EPS,
        weight_decay=0.0,
        grad_clip_norm=0.0,
    )


def create_training_client(service, model: str):
    return service.create_lora_training_client(
        base_model=model,
        rank=C.LORA_RANK,
        train_attn=True,
        train_mlp=True,
        train_unembed=True,
    )


def _mask(datum) -> torch.Tensor:
    return torch.tensor(datum.loss_fn_inputs["weights"].data, dtype=torch.float32)


def reference_logprobs(reference_client, data) -> list[torch.Tensor]:
    """Score with a frozen zero-init LoRA twin in the same training backend."""
    output = reference_client.forward(data, loss_fn="cross_entropy").result(timeout=900)
    values = []
    for row in output.loss_fn_outputs:
        logprobs = row["logprobs"]
        values.append(torch.tensor(getattr(logprobs, "data", logprobs), dtype=torch.float32))
    return values


def make_dpo_loss(data, reference):
    """Paper loss: DPO + 0.1 chosen NLL + 0.001 squared log-ratio."""
    masks = [_mask(datum) for datum in data]
    chosen_indices = range(0, len(data), 2)
    rejected_indices = range(1, len(data), 2)

    def loss_fn(_data, logprobs_list):
        policy = [values.float() for values in logprobs_list]

        def sequence_sum(indices):
            return torch.stack([torch.dot(policy[i], masks[i]) for i in indices])

        def reference_sum(indices):
            return torch.stack([torch.dot(reference[i], masks[i]) for i in indices])

        chosen_logps = sequence_sum(chosen_indices)
        rejected_logps = sequence_sum(rejected_indices)
        chosen_ref = reference_sum(chosen_indices)
        rejected_ref = reference_sum(rejected_indices)
        chosen_ratio = chosen_logps - chosen_ref
        rejected_ratio = rejected_logps - rejected_ref

        preference = -F.logsigmoid(C.DPO_BETA * (chosen_ratio - rejected_ratio)).mean()
        counts = torch.stack([masks[i].sum() for i in chosen_indices])
        nll = -(chosen_logps / counts).mean()
        squared = []
        for index in range(len(data)):
            delta = (policy[index] - reference[index]) * masks[index]
            squared.append((delta**2).sum() / masks[index].sum().clamp(min=1.0))
        sq_log_ratio = torch.stack(squared).mean()

        loss = preference + C.DPO_NLL_COEFFICIENT * nll + C.DPO_KL_COEFFICIENT * sq_log_ratio
        return loss, {
            "loss": float(loss.item()),
            "preference_loss": float(preference.item()),
            "nll_loss": float(nll.item()),
            "sq_log_ratio": float(sq_log_ratio.item()),
            "accuracy": float((chosen_ratio > rejected_ratio).float().mean().item()),
            "margin": float((C.DPO_BETA * (chosen_ratio - rejected_ratio)).mean().item()),
        }

    return loss_fn


def make_sft_loss(data):
    """Batch mean of per-example, response-token mean negative log likelihood."""
    masks = [_mask(datum) for datum in data]

    def loss_fn(_data, logprobs_list):
        per_example = torch.stack(
            [
                -torch.dot(logprobs_list[i].float(), masks[i]) / masks[i].sum().clamp(min=1.0)
                for i in range(len(data))
            ]
        )
        loss = per_example.mean()
        return loss, {
            "loss": float(loss.item()),
            "response_tokens": float(sum(mask.sum() for mask in masks).item()),
        }

    return loss_fn


def train_dpo(service, tok, work_dir, model: str) -> dict:
    record_path = work_dir / "checkpoints" / "dpo.json"
    if record_path.exists():
        print(f"[dpo] using completed record {record_path}", flush=True)
        return C.read_json(record_path)
    source = work_dir / "data" / "dpo_pairs.jsonl"
    rows = C.read_jsonl(source)
    if not rows:
        raise SystemExit(f"missing DPO pairs in {source}; run generate_data.py distill")

    built = []
    for row in rows:
        built.append(
            (
                M.conversation_to_datum(row["chosen"], tok, C.DPO_MAX_LENGTH),
                M.conversation_to_datum(row["rejected"], tok, C.DPO_MAX_LENGTH),
            )
        )
    order = list(range(len(built)))
    random.Random(123456).shuffle(order)
    batches = [
        order[start : start + C.DPO_BATCH_SIZE] for start in range(0, len(order), C.DPO_BATCH_SIZE)
    ]
    batches = [batch for batch in batches if len(batch) == C.DPO_BATCH_SIZE]
    total = len(batches)

    policy = create_training_client(service, model)
    reference_client = create_training_client(service, model)
    history = []
    started = time.time()
    for step, batch in enumerate(batches):
        data = [datum for index in batch for datum in built[index]]
        reference = reference_logprobs(reference_client, data)
        lr = learning_rate(step, total, C.DPO_LEARNING_RATE)
        result = policy.forward_backward_custom(data, make_dpo_loss(data, reference)).result(
            timeout=1800
        )
        policy.optim_step(adam(lr)).result(timeout=900)
        metrics = {**result.metrics, "step": step + 1, "learning_rate": lr}
        history.append(metrics)
        print(
            f"[dpo] {step + 1}/{total} loss={metrics.get('loss', float('nan')):.4f} "
            f"accuracy={metrics.get('accuracy', float('nan')):.3f}",
            flush=True,
        )

    sampler = policy.save_weights_for_sampler("open-character-dpo").result(timeout=1800)
    state = policy.save_state("open-character-dpo-state").result(timeout=1800)
    record = {
        "stage": "dpo_distillation",
        "model": model,
        "rank": C.LORA_RANK,
        "alpha": C.LORA_RANK * 2,
        "pairs": len(built),
        "updates": total,
        "sampler_path": str(getattr(sampler, "path", sampler)),
        "state_path": str(getattr(state, "path", state)),
        "seconds": round(time.time() - started, 1),
        "data_sha256": C.stable_hash([row["pair_id"] for row in rows]),
        "hyperparameters": {
            "learning_rate": C.DPO_LEARNING_RATE,
            "batch_size": C.DPO_BATCH_SIZE,
            "beta": C.DPO_BETA,
            "nll_coefficient": C.DPO_NLL_COEFFICIENT,
            "squared_log_ratio_coefficient": C.DPO_KL_COEFFICIENT,
            "max_length": C.DPO_MAX_LENGTH,
        },
        "metrics": history,
    }
    M.write_checkpoint_record(work_dir, "dpo", record)
    print(f"[dpo] saved {record['sampler_path']}", flush=True)
    return record


def train_sft(service, tok, work_dir, model: str) -> dict:
    record_path = work_dir / "checkpoints" / "sft.json"
    if record_path.exists():
        print(f"[sft] using completed record {record_path}", flush=True)
        return C.read_json(record_path)
    dpo_record = M.read_checkpoint_record(work_dir, "dpo")
    source = work_dir / "data" / "introspection_sft.jsonl"
    rows = C.read_jsonl(source)
    if not rows:
        raise SystemExit(f"missing introspection data in {source}; generate it first")

    data = []
    dropped = 0
    for row in rows:
        try:
            data.append(
                M.conversation_to_datum(row["messages"], tok, C.SFT_MAX_LENGTH, trim_old_turns=True)
            )
        except ValueError:
            dropped += 1
    order = list(range(len(data)))
    random.Random(123456).shuffle(order)
    batches = [
        order[start : start + C.SFT_BATCH_SIZE] for start in range(0, len(order), C.SFT_BATCH_SIZE)
    ]
    batches = [batch for batch in batches if len(batch) == C.SFT_BATCH_SIZE]
    total = len(batches)

    policy = create_training_client(service, model)
    policy.load_state(dpo_record["state_path"]).result(timeout=1800)
    history = []
    started = time.time()
    for step, batch in enumerate(batches):
        minibatch = [data[index] for index in batch]
        lr = learning_rate(step, total, C.SFT_LEARNING_RATE)
        result = policy.forward_backward_custom(minibatch, make_sft_loss(minibatch)).result(
            timeout=1800
        )
        policy.optim_step(adam(lr)).result(timeout=900)
        metrics = {**result.metrics, "step": step + 1, "learning_rate": lr}
        history.append(metrics)
        print(
            f"[sft] {step + 1}/{total} loss={metrics.get('loss', float('nan')):.4f}",
            flush=True,
        )

    sampler = policy.save_weights_for_sampler("open-character-final").result(timeout=1800)
    state = policy.save_state("open-character-final-state").result(timeout=1800)
    record = {
        "stage": "introspection_sft",
        "model": model,
        "rank": C.LORA_RANK,
        "alpha": C.LORA_RANK * 2,
        "examples": len(data),
        "dropped": dropped,
        "updates": total,
        "initialized_from": dpo_record["state_path"],
        "sampler_path": str(getattr(sampler, "path", sampler)),
        "state_path": str(getattr(state, "path", state)),
        "seconds": round(time.time() - started, 1),
        "data_sha256": C.stable_hash([row["messages"] for row in rows]),
        "hyperparameters": {
            "learning_rate": C.SFT_LEARNING_RATE,
            "batch_size": C.SFT_BATCH_SIZE,
            "max_length": C.SFT_MAX_LENGTH,
        },
        "metrics": history,
    }
    M.write_checkpoint_record(work_dir, "sft", record)
    print(f"[sft] saved {record['sampler_path']}", flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    M.add_connection_arguments(parser)
    parser.add_argument("stage", choices=("dpo", "sft"))
    args = parser.parse_args()

    M.wait_healthy(args.base_url)
    service = M.connect(args.base_url, args.api_key)
    tok = M.tokenizer(args.model)
    if args.stage == "dpo":
        train_dpo(service, tok, args.work_dir, args.model)
    else:
        train_sft(service, tok, args.work_dir, args.model)


if __name__ == "__main__":
    main()
