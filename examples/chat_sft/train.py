from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import tinker
from dataset import ChatDataset, conversation_to_datum, load_chat_dataset
from tinker import types
from tqdm import tqdm
from transformers import AutoTokenizer


@dataclass(frozen=True)
class Config:
    # server
    base_url: str
    api_key: str

    # data/model
    dataset: str
    base_model: str
    lora_rank: int

    # train
    num_steps: int
    batch_size: int
    learning_rate: float
    max_length: int

    # eval/plot/save
    eval_batch: int = 16
    seed: int = 42
    plot_path: Optional[str] = None
    sampler_name: Optional[str] = None
    checkpoint_name: Optional[str] = None

    # wandb
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None


def defaulted_names(cfg: Config) -> Config:
    """Fill default names that depend on dataset."""
    sampler_name = cfg.sampler_name or f"{cfg.dataset}-sft-demo"
    checkpoint_name = cfg.checkpoint_name or f"{cfg.dataset}-sft-final"
    return Config(
        **{
            **cfg.__dict__,
            "sampler_name": sampler_name,
            "checkpoint_name": checkpoint_name,
        }
    )


def init_wandb(cfg: Config):
    if cfg.wandb_project is None or cfg.wandb_name is None:
        print("[wandb] disabled")
        return None

    try:
        import wandb
    except Exception as e:
        print(f"[wandb] import failed, continue without wandb. err={e}")
        return None

    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_name,
        config={
            "dataset": cfg.dataset,
            "base_model": cfg.base_model,
            "lora_rank": cfg.lora_rank,
            "num_steps": cfg.num_steps,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "max_length": cfg.max_length,
            "eval_batch": cfg.eval_batch,
            "seed": cfg.seed,
            "base_url": cfg.base_url,
        },
    )
    print(f"[wandb] initialized project={cfg.wandb_project} name={cfg.wandb_name}")
    return wandb


def compute_weighted_nll_from_outputs(loss_fn_outputs, datums) -> float:
    total_loss = 0.0
    total_weight = 0.0

    for i, out in enumerate(loss_fn_outputs):
        logprobs = out["logprobs"]
        if hasattr(logprobs, "tolist"):
            logprobs = logprobs.tolist()

        w = datums[i].loss_fn_inputs["weights"]
        if hasattr(w, "tolist"):
            w = w.tolist()

        for lp, wt in zip(logprobs, w, strict=False):
            total_loss += -lp * wt
            total_weight += wt

    return total_loss / max(total_weight, 1.0)


def connect(cfg: Config) -> tinker.ServiceClient:
    print(f"[1/6] connect service: {cfg.base_url}")
    return tinker.ServiceClient(base_url=cfg.base_url, api_key=cfg.api_key)


def load_tokenizer(cfg: Config):
    print(f"[2/6] load tokenizer: {cfg.base_model}")
    tok = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    print(f"      eos_token_id={tok.eos_token_id}")
    return tok


def create_training_client(service_client: tinker.ServiceClient, cfg: Config):
    print(f"[4/6] create lora training client: rank={cfg.lora_rank}")
    return service_client.create_lora_training_client(
        base_model=cfg.base_model,
        rank=cfg.lora_rank,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )


def build_datums(
    batch: List[List[Dict[str, Any]]],
    tokenizer,
    max_length: int,
) -> List[types.Datum]:
    datums: List[types.Datum] = []
    for messages in batch:
        try:
            datums.append(conversation_to_datum(messages, tokenizer, max_length))
        except ValueError:
            continue
    return datums


def train_sft(
    training_client,
    train_dataset: ChatDataset,
    tokenizer,
    cfg: Config,
    wandb=None,
) -> List[Dict[str, Any]]:
    print(f"[5/6] train sft: steps={cfg.num_steps} batch={cfg.batch_size} lr={cfg.learning_rate}")
    metrics_history: List[Dict[str, Any]] = []

    pbar = tqdm(range(cfg.num_steps), desc="SFT", dynamic_ncols=True)
    for step in pbar:
        batch = train_dataset.get_batch(cfg.batch_size)
        datums = build_datums(batch, tokenizer, cfg.max_length)

        if not datums:
            pbar.set_postfix_str("skip (0 valid)")
            if wandb:
                wandb.log({"train/skip_batch": 1}, step=step)
            continue

        fwdbwd = training_client.forward_backward(datums, loss_fn="cross_entropy").result()
        loss = compute_weighted_nll_from_outputs(fwdbwd.loss_fn_outputs, datums)

        training_client.optim_step(types.AdamParams(learning_rate=cfg.learning_rate)).result()

        metrics_history.append({"step": step, "loss": loss})

        pbar.set_postfix(loss=f"{loss:.4f}", valid=f"{len(datums)}/{cfg.batch_size}")

        if wandb:
            wandb.log(
                {
                    "train/loss": loss,
                    "train/valid_datums": len(datums),
                    "train/batch_size": cfg.batch_size,
                    "train/learning_rate": cfg.learning_rate,
                },
                step=step,
            )

    if metrics_history:
        loss0 = metrics_history[0]["loss"]
        loss1 = metrics_history[-1]["loss"]
        print(f"      done. loss: {loss0:.4f} -> {loss1:.4f}")
    else:
        print("      done. (no metrics recorded; all batches invalid?)")

    return metrics_history


def eval_on_test(
    training_client,
    test_dataset: ChatDataset,
    tokenizer,
    cfg: Config,
    wandb=None,
) -> Optional[float]:
    print("[eval] run on test set")
    test_batch = test_dataset.get_batch(min(cfg.eval_batch, len(test_dataset)))
    test_datums = build_datums(test_batch, tokenizer, cfg.max_length)

    print(f"      valid_datums={len(test_datums)}/{len(test_batch)}")
    if not test_datums:
        print("      no valid test samples")
        return None

    forward = training_client.forward(test_datums, loss_fn="cross_entropy").result()
    test_loss = compute_weighted_nll_from_outputs(forward.loss_fn_outputs, test_datums)

    print(f"      test_nll={test_loss:.4f}")
    if wandb:
        wandb.log({"eval/test_nll": test_loss})
    return test_loss


def plot_training_loss(metrics_history: List[Dict[str, Any]], cfg: Config, wandb=None) -> None:
    if not metrics_history:
        return
    if cfg.plot_path is None:
        return

    print(f"[plot] save: {cfg.plot_path}")
    steps = [m["step"] for m in metrics_history]
    losses = [m["loss"] for m in metrics_history]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, "b-", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Loss (NLL)")
    plt.title(f"{cfg.dataset.upper()} SFT Training")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.plot_path, dpi=150)
    plt.show()

    if wandb:
        try:
            wandb.log({"artifacts/loss_plot": wandb.Image(cfg.plot_path)})
        except Exception as e:
            print(f"[wandb] image log failed: {e}")


def save_and_sample(
    service_client: tinker.ServiceClient,
    training_client,
    tokenizer,
    cfg: Config,
    wandb=None,
) -> None:
    print("[6/6] save weights + sample")
    sampling_path = training_client.save_weights_for_sampler(name=cfg.sampler_name).result().path
    sampling_client = service_client.create_sampling_client(model_path=sampling_path)
    print(f"      sampler_path={sampling_path}")

    test_messages = [{"role": "user", "content": "Write a haiku about programming."}]
    prompt_text = tokenizer.apply_chat_template(
        test_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

    sample_result = sampling_client.sample(
        prompt=types.ModelInput.from_ints(prompt_tokens),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=128, temperature=0.7),
    ).result()

    response = tokenizer.decode(sample_result.sequences[0].tokens)

    print("=== Generation ===")
    print("User: Write a haiku about programming.")
    print("Assistant:", response[:500])
    print("Response chars:", len(response))

    if wandb:
        wandb.log(
            {
                "sample/prompt": "Write a haiku about programming.",
                "sample/response": response,
                "sample/response_chars": len(response),
                "save/sampler_name": cfg.sampler_name,
                "save/sampler_path": sampling_path,
            }
        )

    training_client.save_state(name=cfg.checkpoint_name).result()
    print(f"      checkpoint_saved={cfg.checkpoint_name}")

    if wandb:
        wandb.log({"save/checkpoint_name": cfg.checkpoint_name})


def _none_if_literal_none(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = x.strip()
    if s == "" or s.lower() == "none":
        return None
    return x


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TuFT-Chat SFT training")
    p.add_argument("--base-url", default=os.getenv("TINKER_BASE_URL", "http://localhost:10610"))
    p.add_argument("--api-key", default=os.getenv("TINKER_API_KEY"))
    p.add_argument("--dataset", default="no_robots")
    p.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--lora-rank", type=int, default=16)

    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--eval-batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--plot-path", default=None)
    p.add_argument("--sampler-name", default=None)
    p.add_argument("--checkpoint-name", default=None)

    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-name", default=None)

    args = p.parse_args(argv)
    args.wandb_project = _none_if_literal_none(args.wandb_project)
    args.wandb_name = _none_if_literal_none(args.wandb_name)
    return args


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config(
        base_url=args.base_url,
        api_key=args.api_key,
        dataset=args.dataset,
        base_model=args.base_model,
        lora_rank=args.lora_rank,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        eval_batch=args.eval_batch,
        seed=args.seed,
        plot_path=args.plot_path,
        sampler_name=args.sampler_name,
        checkpoint_name=args.checkpoint_name,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )
    return defaulted_names(cfg)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cfg = build_config(args)

    print(
        "[config]",
        f"base_url={cfg.base_url}",
        f"dataset={cfg.dataset}",
        f"base_model={cfg.base_model}",
        f"rank={cfg.lora_rank}",
        f"steps={cfg.num_steps}",
        f"batch={cfg.batch_size}",
        f"lr={cfg.learning_rate}",
        f"max_len={cfg.max_length}",
        f"plot_path={cfg.plot_path}",
        f"wandb={cfg.wandb_project}/{cfg.wandb_name}",
    )

    random.seed(cfg.seed)

    wandb = init_wandb(cfg)

    service_client = connect(cfg)
    tokenizer = load_tokenizer(cfg)

    print(f"[3/6] load dataset: {cfg.dataset}")
    train_dataset, test_dataset = load_chat_dataset(cfg.dataset, seed=cfg.seed)
    print(f"      train={len(train_dataset)} test={len(test_dataset)}")

    sample = train_dataset.get_batch(1)[0]
    print(f"      sample_messages={len(sample)} (show first 3)")
    for msg in sample[:3]:
        content = msg["content"][:100] + ("..." if len(msg["content"]) > 100 else "")
        print(f"        [{msg['role']}]: {content}")

    training_client = create_training_client(service_client, cfg)

    metrics_history = train_sft(training_client, train_dataset, tokenizer, cfg, wandb=wandb)
    eval_on_test(training_client, test_dataset, tokenizer, cfg, wandb=wandb)
    plot_training_loss(metrics_history, cfg, wandb=wandb)
    save_and_sample(service_client, training_client, tokenizer, cfg, wandb=wandb)

    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
