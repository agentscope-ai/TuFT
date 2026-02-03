from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import tinker
import torch
from env import (
    COUNTDOWN_FEWSHOT,
    CountdownDatasetLoader,
    compute_reward,
    make_prompt_model_input,
)
from tinker import types
from tinker.types.tensor_data import TensorData
from tqdm import tqdm


@dataclass(frozen=True)
class Config:
    # server
    base_url: str
    api_key: str

    # dataset/model
    dataset: str
    base_model: str
    lora_rank: int

    # training
    num_steps: int
    batch_size: int
    group_size: int
    learning_rate: float
    max_tokens: int
    temperature: float

    # split
    test_size: int
    seed: int

    # reward
    format_score: float
    continuous_shaping: bool

    # eval
    eval_every: int
    eval_batch_size: int
    eval_group_size: int
    eval_temperature: float
    reward_ema_alpha: float

    # plot/save
    plot_path: Optional[str] = None
    sampler_name: Optional[str] = None
    checkpoint_name: Optional[str] = None

    # wandb
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None


def defaulted_names(cfg: Config) -> Config:
    sampler_name = cfg.sampler_name or "COUNTDOWN-final"
    checkpoint_name = cfg.checkpoint_name or "COUNTDOWN-rl-final"
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
        config={**cfg.__dict__},
    )
    print(f"[wandb] initialized project={cfg.wandb_project} name={cfg.wandb_name}")
    return wandb


def connect(cfg: Config) -> tinker.ServiceClient:
    print(f"[1/6] connect service: {cfg.base_url}")
    return tinker.ServiceClient(base_url=cfg.base_url, api_key=cfg.api_key)


def load_dataset(cfg: Config) -> CountdownDatasetLoader:
    print(f"[2/6] load dataset: {cfg.dataset}")
    ds = CountdownDatasetLoader(cfg.dataset, cfg.test_size, cfg.seed)
    print(f"      train={len(ds.train)} test={len(ds.test)}")
    samp = ds.get_batch(1, split="train")[0]
    print(f"      sample_prompt_head={(COUNTDOWN_FEWSHOT + samp.question)[:140]} ...")
    return ds


def create_training_client(service_client: tinker.ServiceClient, cfg: Config):
    print(f"[3/6] create lora training client: base_model={cfg.base_model} rank={cfg.lora_rank}")
    training_client = service_client.create_lora_training_client(
        base_model=cfg.base_model,
        rank=cfg.lora_rank,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )
    return training_client


def build_params(cfg: Config):
    print("[4/6] build sampling/optim params")
    sampling_params_train = types.SamplingParams(
        max_tokens=cfg.max_tokens, temperature=cfg.temperature
    )
    sampling_params_eval = types.SamplingParams(
        max_tokens=cfg.max_tokens, temperature=cfg.eval_temperature
    )
    adam_params = types.AdamParams(
        learning_rate=cfg.learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )
    return sampling_params_train, sampling_params_eval, adam_params


def build_importance_sampling_datum(
    *,
    prompt: types.ModelInput,
    ob_len: int,
    toks: List[int],
    lps: List[float],
    adv: float,
) -> types.Datum:
    """Build the exact same datum as notebook (padding + alignment)."""
    # model_input excludes final token because training usually predicts next token
    model_input = prompt.append(types.EncodedTextChunk(tokens=toks[:-1]))

    # Align lengths with model_input.length
    target_tokens = [0] * ob_len + toks
    padded_sampling_logprobs = [0.0] * ob_len + lps
    padded_advantages = [0.0] * ob_len + [adv] * (model_input.length - ob_len)

    if not (
        model_input.length
        == len(target_tokens)
        == len(padded_sampling_logprobs)
        == len(padded_advantages)
    ):
        raise RuntimeError(
            f"Length mismatch: model_input={model_input.length} "
            f"target={len(target_tokens)} logprobs={len(padded_sampling_logprobs)} "
            f"adv={len(padded_advantages)}"
        )

    target_tokens_t = torch.tensor(target_tokens, dtype=torch.long)
    logprobs_t = torch.tensor(padded_sampling_logprobs, dtype=torch.float32)
    advantages_t = torch.tensor(padded_advantages, dtype=torch.float32)

    loss_fn_inputs = {
        "target_tokens": TensorData.from_torch(target_tokens_t),
        "logprobs": TensorData.from_torch(logprobs_t),
        "advantages": TensorData.from_torch(advantages_t),
    }

    return types.Datum(model_input=model_input, loss_fn_inputs=loss_fn_inputs)


def do_eval(
    *,
    step: int,
    service_client: tinker.ServiceClient,
    training_client,
    tokenizer,
    dataset: CountdownDatasetLoader,
    sampling_params_eval: types.SamplingParams,
    cfg: Config,
) -> float:
    eval_path = training_client.save_weights_for_sampler(name=f"eval_{step:06d}").result().path
    eval_client = service_client.create_sampling_client(model_path=eval_path)

    probs = dataset.get_batch(cfg.eval_batch_size, split="test")
    rewards: List[float] = []

    for prob in probs:
        prompt_text = COUNTDOWN_FEWSHOT + prob.question
        prompt = make_prompt_model_input(tokenizer, prompt_text)

        res = eval_client.sample(
            prompt=prompt,
            num_samples=cfg.eval_group_size,
            sampling_params=sampling_params_eval,
        ).result()

        for seq in res.sequences:
            toks = list(seq.tokens)
            resp_text = tokenizer.decode(toks, skip_special_tokens=True)
            r = compute_reward(
                response_text=resp_text,
                target=prob.target,
                nums=prob.nums,
                format_score=cfg.format_score,
                use_continuous_shaping=cfg.continuous_shaping,
            )
            rewards.append(float(r))

    return sum(rewards) / max(1, len(rewards))


def train_loop(
    *,
    service_client: tinker.ServiceClient,
    training_client,
    tokenizer,
    dataset: CountdownDatasetLoader,
    sampling_params_train: types.SamplingParams,
    sampling_params_eval: types.SamplingParams,
    adam_params: types.AdamParams,
    cfg: Config,
    wandb=None,
) -> List[Dict[str, Any]]:
    print(
        f"[5/6] train rl: steps={cfg.num_steps} "
        f"batch={cfg.batch_size} group={cfg.group_size} lr={cfg.learning_rate}"
    )

    ema_eval_reward = None
    metrics_history: List[Dict[str, Any]] = []

    pbar = tqdm(range(cfg.num_steps), desc="RL", dynamic_ncols=True)
    for step in pbar:
        problems_P = dataset.get_batch(cfg.batch_size, split="train")

        # Sync weights -> sampling client
        save_result = training_client.save_weights_for_sampler(name=f"rl_step_{step:06d}").result()
        sampling_path = save_result.path
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)

        datums_D: List[types.Datum] = []
        mean_rewards_P: List[float] = []
        kept_rollouts = 0
        skipped_problems = 0

        for prob in problems_P:
            prompt_text = COUNTDOWN_FEWSHOT + prob.question
            prompt = make_prompt_model_input(tokenizer, prompt_text)

            sample_res = sampling_client.sample(
                prompt=prompt,
                num_samples=cfg.group_size,
                sampling_params=sampling_params_train,
            ).result()

            rewards_G: List[float] = []
            tokens_G_T: List[List[int]] = []
            logprobs_G_T: List[List[float]] = []

            for seq in sample_res.sequences:
                toks = list(seq.tokens)
                lps = seq.logprobs
                if lps is None:
                    raise RuntimeError("Sampling did not return logprobs.")

                resp_text = tokenizer.decode(toks, skip_special_tokens=True)
                r = compute_reward(
                    response_text=resp_text,
                    target=prob.target,
                    nums=prob.nums,
                    format_score=cfg.format_score,
                    use_continuous_shaping=cfg.continuous_shaping,
                )

                rewards_G.append(float(r))
                tokens_G_T.append(toks)
                logprobs_G_T.append(list(lps))

            mean_r = sum(rewards_G) / len(rewards_G)
            mean_rewards_P.append(mean_r)

            var_r = sum((r - mean_r) ** 2 for r in rewards_G) / max(1, len(rewards_G))
            std_r = var_r**0.5
            if std_r < 1e-8:
                skipped_problems += 1
                continue

            advantages_G = [(r - mean_r) / (std_r + 1e-6) for r in rewards_G]

            ob_len = prompt.length - 1

            for toks, lps, adv in zip(tokens_G_T, logprobs_G_T, advantages_G, strict=True):
                datums_D.append(
                    build_importance_sampling_datum(
                        prompt=prompt,
                        ob_len=ob_len,
                        toks=toks,
                        lps=lps,
                        adv=adv,
                    )
                )
                kept_rollouts += 1

        mean_reward_train = sum(mean_rewards_P) / max(1, len(mean_rewards_P))

        # Optimization step
        if datums_D:
            training_client.forward_backward(datums_D, loss_fn="importance_sampling").result()
            training_client.optim_step(adam_params).result()

        # Eval + EMA
        eval_reward = None
        ema_now = None
        if cfg.eval_every > 0 and (step % cfg.eval_every == 0):
            eval_reward = do_eval(
                step=step,
                service_client=service_client,
                training_client=training_client,
                tokenizer=tokenizer,
                dataset=dataset,
                sampling_params_eval=sampling_params_eval,
                cfg=cfg,
            )
            if ema_eval_reward is None:
                ema_eval_reward = eval_reward
            else:
                a = cfg.reward_ema_alpha
                ema_eval_reward = (1 - a) * ema_eval_reward + a * eval_reward
            ema_now = ema_eval_reward

        metrics = {
            "step": int(step),
            "train_mean_reward": float(mean_reward_train),
            "eval_mean_reward": None if eval_reward is None else float(eval_reward),
            "ema_eval_reward": None if ema_now is None else float(ema_now),
            "kept_rollouts": int(kept_rollouts),
            "skipped_problems": int(skipped_problems),
        }
        metrics_history.append(metrics)

        postfix = {
            "train_r": f"{mean_reward_train:.4f}",
            "kept": kept_rollouts,
            "skipP": skipped_problems,
        }
        if eval_reward is not None:
            postfix["eval_r"] = f"{eval_reward:.4f}"
            postfix["ema"] = f"{ema_now:.4f}"
        pbar.set_postfix(postfix)

        if wandb:
            log_data = {
                "train/mean_reward": mean_reward_train,
                "train/kept_rollouts": kept_rollouts,
                "train/skipped_problems": skipped_problems,
                "train/batch_size": cfg.batch_size,
                "train/group_size": cfg.group_size,
                "train/learning_rate": cfg.learning_rate,
            }
            if eval_reward is not None:
                log_data.update(
                    {
                        "eval/mean_reward": eval_reward,
                        "eval/ema_reward": ema_now,
                    }
                )

            wandb.log(log_data, step=step)

    if metrics_history:
        print(
            "      done. train_mean_reward:",
            f"{metrics_history[0]['train_mean_reward']:.4f} -> "
            f"{metrics_history[-1]['train_mean_reward']:.4f}",
        )

    return metrics_history


def final_eval_plot_save(
    *,
    service_client: tinker.ServiceClient,
    training_client,
    tokenizer,
    dataset: CountdownDatasetLoader,
    metrics_history: List[Dict[str, Any]],
    cfg: Config,
    wandb=None,
) -> None:
    print("[6/6] final eval + plot + checkpoint")

    final_path = training_client.save_weights_for_sampler(name=cfg.sampler_name).result().path
    final_client = service_client.create_sampling_client(model_path=final_path)

    test_problems = dataset.get_batch(1, split="test")

    print("Final Evaluation:")
    print("=" * 60)

    greedy_temp = 0.0
    for i, problem in enumerate(test_problems):
        prompt_text = COUNTDOWN_FEWSHOT + problem.question
        prompt_input = make_prompt_model_input(tokenizer, prompt_text)

        try:
            sampling_params_greedy = types.SamplingParams(
                max_tokens=cfg.max_tokens,
                temperature=greedy_temp,
            )
            result = final_client.sample(
                prompt=prompt_input,
                num_samples=1,
                sampling_params=sampling_params_greedy,
            ).result()
        except Exception as e:
            print("Greedy temp=0.0 failed, retrying with temp=0.1. Error:", repr(e))
            sampling_params_greedy = types.SamplingParams(
                max_tokens=cfg.max_tokens,
                temperature=0.1,
            )
            result = final_client.sample(
                prompt=prompt_input,
                num_samples=1,
                sampling_params=sampling_params_greedy,
            ).result()

        response = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
        reward = compute_reward(
            response_text=response,
            target=problem.target,
            nums=problem.nums,
            format_score=cfg.format_score,
            use_continuous_shaping=cfg.continuous_shaping,
        )

        status = "PASS" if reward >= 1.0 else "FAIL"
        print(f"[{i}] {problem.question[:100]}...")
        print(f"A: {response.strip()[:140]}... [{status}] reward={reward:.4f}")
        print()

        if wandb:
            wandb.log(
                {
                    "final/sample_question": problem.question,
                    "final/sample_response": response,
                    "final/sample_reward": reward,
                    "final/sample_status": status,
                    "save/final_sampler_path": final_path,
                }
            )

    # plot (optional)
    if cfg.plot_path is not None:
        eval_ms = [m for m in metrics_history if m.get("eval_mean_reward") is not None]
        eval_steps = [m["step"] for m in eval_ms]
        eval_rewards = [m["eval_mean_reward"] for m in eval_ms]

        ema_ms = [m for m in metrics_history if m.get("ema_eval_reward") is not None]
        ema_steps = [m["step"] for m in ema_ms]
        ema_rewards = [m["ema_eval_reward"] for m in ema_ms]

        if len(eval_steps) > 0:
            print(f"[plot] save: {cfg.plot_path}")
            plt.figure(figsize=(10, 5))
            plt.plot(eval_steps, eval_rewards, "b-", linewidth=2, label="eval_mean_reward")
            if len(ema_steps) > 0:
                plt.plot(
                    ema_steps,
                    ema_rewards,
                    "r-",
                    linewidth=2,
                    alpha=0.9,
                    label="ema_eval_reward",
                )

            plt.xlabel("Step")
            plt.ylabel("Eval mean reward")
            plt.title("COUNTDOWN RL Training")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(cfg.plot_path, dpi=150)
            plt.show()

            if wandb:
                try:
                    wandb.log({"artifacts/reward_plot": wandb.Image(cfg.plot_path)})
                except Exception as e:
                    print(f"[wandb] image log failed: {e}")
        else:
            print("[plot] no eval points available; skip")

    checkpoint = training_client.save_state(name=cfg.checkpoint_name).result()
    print("Checkpoint:", checkpoint.path)


def _none_if_literal_none(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = x.strip()
    if s == "" or s.lower() == "none":
        return None
    return x


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TuFT-Countdown RL")
    p.add_argument("--base-url", default=os.getenv("TINKER_BASE_URL", "http://localhost:10610"))
    p.add_argument("--api-key", default=os.getenv("TINKER_API_KEY", "tml-test-key"))

    p.add_argument("--dataset", default="Jiayi-Pan/Countdown-Tasks-3to4")
    p.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--lora-rank", type=int, default=8)

    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.9)

    p.add_argument("--test-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--format-score", type=float, default=0.1)
    p.add_argument("--continuous-shaping", action="store_true", default=True)
    p.add_argument("--no-continuous-shaping", action="store_false", dest="continuous_shaping")

    p.add_argument("--eval-every", type=int, default=30)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--eval-group-size", type=int, default=1)
    p.add_argument("--eval-temperature", type=float, default=0.1)
    p.add_argument("--reward-ema-alpha", type=float, default=0.1)

    p.add_argument("--plot-path", default=None)
    p.add_argument("--sampler-name", default=None)
    p.add_argument("--checkpoint-name", default=None)

    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-name", default=None)

    args = p.parse_args(argv)
    args.wandb_project = _none_if_literal_none(args.wandb_project)
    args.wandb_name = _none_if_literal_none(args.wandb_name)
    args.plot_path = _none_if_literal_none(args.plot_path)
    args.sampler_name = _none_if_literal_none(args.sampler_name)
    args.checkpoint_name = _none_if_literal_none(args.checkpoint_name)
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
        group_size=args.group_size,
        learning_rate=args.learning_rate,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        test_size=args.test_size,
        seed=args.seed,
        format_score=args.format_score,
        continuous_shaping=args.continuous_shaping,
        eval_every=args.eval_every,
        eval_batch_size=args.eval_batch_size,
        eval_group_size=args.eval_group_size,
        eval_temperature=args.eval_temperature,
        reward_ema_alpha=args.reward_ema_alpha,
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
        f"group={cfg.group_size}",
        f"lr={cfg.learning_rate}",
        f"max_tokens={cfg.max_tokens}",
        f"train_temp={cfg.temperature}",
        f"eval_temp={cfg.eval_temperature}",
        f"plot_path={cfg.plot_path}",
        f"sampler_name={cfg.sampler_name}",
        f"checkpoint_name={cfg.checkpoint_name}",
        f"wandb={cfg.wandb_project}/{cfg.wandb_name}",
    )

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    wandb = init_wandb(cfg)

    service_client = connect(cfg)
    dataset = load_dataset(cfg)
    training_client = create_training_client(service_client, cfg)

    tokenizer = training_client.get_tokenizer()
    print(f"      tokenizer={type(tokenizer).__name__}")

    sampling_params_train, sampling_params_eval, adam_params = build_params(cfg)

    metrics_history = train_loop(
        service_client=service_client,
        training_client=training_client,
        tokenizer=tokenizer,
        dataset=dataset,
        sampling_params_train=sampling_params_train,
        sampling_params_eval=sampling_params_eval,
        adam_params=adam_params,
        cfg=cfg,
        wandb=wandb,
    )

    final_eval_plot_save(
        service_client=service_client,
        training_client=training_client,
        tokenizer=tokenizer,
        dataset=dataset,
        metrics_history=metrics_history,
        cfg=cfg,
        wandb=wandb,
    )

    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
