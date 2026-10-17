"""CPU preflight of the causal agent workload before paid GPU execution."""

import asyncio

from tinker import types

from scripts.benchmark_loopweave_agents import (
    parse_action,
    reward,
    scheduling,
    task_spec,
    tool_reply,
    training_datum,
)
from tuft.config import SchedulingConfig
from tuft.scheduling import ResourceScheduler


def test_tools_rewards_and_policy_gradient_alignment():
    spec = task_spec(0, 0, 0)
    action = parse_action('Here: {"tool":"lookup","key":"policy-0"}')
    reply, ok = tool_reply(spec, action)
    assert ok and "stock-0" in reply
    assert not tool_reply(spec, parse_action("malformed"))[1]
    assert not tool_reply(spec, {"tool": "execute", "key": "policy-0"})[1]
    assert reward(spec, {"decision": "hold"}) == 1
    datum = training_datum([10, 11, 12], [20, 21], [-1.0, -2.0], 1)
    assert datum.model_input.to_ints() == [10, 11, 12, 20]
    assert datum.loss_fn_inputs["target_tokens"].data == [11, 12, 20, 21]
    assert datum.loss_fn_inputs["logprobs"].data == [0.0, 0.0, -1.0, -2.0]
    assert datum.loss_fn_inputs["advantages"].data == [0.0, 0.0, 0.25, 0.25]
    types.CreateSessionRequest(sdk_version="agent-benchmark", tags=[], user_metadata=None)


async def test_long_context_wave_has_bounded_probe_and_matched_fixed_capacity():
    fixed, flex = scheduling("balanced"), scheduling("loopweave")
    assert {k: v for k, v in fixed.items() if k != "policy"} == {
        k: v for k, v in flex.items() if k != "policy"
    }
    s = ResourceScheduler(2, SchedulingConfig(**flex), flex_index=0)
    release, borrowed = asyncio.Event(), asyncio.Event()
    indices = []

    async def sample(index):
        indices.append(index)
        if index == 0:
            borrowed.set()
        await release.wait()

    jobs = [
        asyncio.create_task(s.sample(str(i % 8), str(i % 8), 4096, True, sample)) for i in range(32)
    ]
    try:
        await asyncio.wait_for(borrowed.wait(), 2)
        assert indices.count(0) == 1 and indices.count(1) == 16
        assert s.snapshot()["queued"] == 15
    finally:
        release.set()
        await asyncio.gather(*jobs)
        await s.close()


def test_direct_cli_import_does_not_depend_on_scripts_namespace():
    import subprocess
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/benchmark_loopweave_agents.py"
    subprocess.run(
        [sys.executable, str(path), "--help"], check=True, capture_output=True, cwd="/tmp"
    )


def test_chat_template_contract_with_real_transformers_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from scripts.benchmark_loopweave_agents import encode_prompt

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")),
        unk_token="[UNK]",
        chat_template="{% for m in messages %}{{ m['content'] }}{% endfor %}",
    )
    tokens = encode_prompt(tokenizer, [{"role": "user", "content": "hello"}])
    assert tokens == [1]
    types.ModelInput.from_ints(tokens)
