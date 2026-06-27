"""HotpotQA Agent RL task: multi-hop question answering with search tools.

The agent uses a ReAct-style loop to answer multi-hop questions by searching
for and looking up information from supporting paragraphs. This is a classic
agent RL benchmark that requires multi-turn reasoning.

Tools available to the agent:
- Search[query]: Search for a Wikipedia topic, returns the summary paragraph
- Lookup[term]: Look up a term in the current page, returns the relevant sentence
- Finish[answer]: Submit the final answer

Reward:
- Sparse: 1.0 if final answer matches ground truth (F1-based partial credit)
- Format bonus: 0.1 for valid Finish[] action with any answer
"""

from __future__ import annotations

import random
import re
import string
from typing import Any, Dict, List, Optional

import datasets

from .agent_task import AgentTask, StepResult
from .base import TaskPrompt


# ─── System prompt with tool descriptions ────────────────────────────

REACT_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions by searching for information.

You have access to the following tools:
- Search[query]: Search for a Wikipedia topic. Returns a short summary paragraph.
- Lookup[term]: Look up a term in the current page to find specific details.
- Finish[answer]: Submit your final answer to the question.

Use the following format for each step:
Thought: <your reasoning about what to do next>
Action: <one of Search[...], Lookup[...], or Finish[...]>

Important:
- You MUST call Finish[answer] to submit your final answer.
- Keep your search queries concise and specific.
- You can search multiple times to gather information from different sources.
"""

REACT_QUESTION_TEMPLATE = "Question: {question}\n\nThought:"

# ─── Regex for parsing agent actions ─────────────────────────────────

_ACTION_RE = re.compile(
    r"Action:\s*(Search|Lookup|Finish)\[(.+?)\]", flags=re.IGNORECASE | re.DOTALL
)


def _normalize_answer(s: str) -> str:
    """Normalize answer string for comparison (lowercase, remove articles/punctuation)."""
    s = s.lower().strip()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    s = " ".join(s.split())
    return s


def _compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _compute_em(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


class HotpotQATask(AgentTask):
    """HotpotQA multi-hop QA task with ReAct-style tool use.

    Dataset: hotpotqa/hotpot_qa (distractor setting)
    Interaction: Multi-turn with Search/Lookup/Finish tools
    Reward: F1-based partial credit with exact match bonus
    """

    def __init__(
        self,
        dataset_name: str = "hotpotqa/hotpot_qa",
        dataset_config: str = "distractor",
        seed: int = 42,
        max_search_results: int = 3,
    ):
        self._dataset_name = dataset_name
        self._dataset_config = dataset_config
        self._seed = seed
        self._max_search_results = max_search_results

        self._train_data: List[Dict[str, Any]] = []
        self._test_data: List[Dict[str, Any]] = []
        self._train_idx = 0
        self._test_idx = 0

        # Per-episode state
        self._current_paragraphs: Dict[str, str] = {}  # title -> paragraph text
        self._current_page: Optional[str] = None  # current page content for Lookup

    def load_dataset(self) -> None:
        """Load HotpotQA dataset from HuggingFace."""
        ds = datasets.load_dataset(
            self._dataset_name,
            self._dataset_config,
            split="train",
        )

        # Take a manageable subset and split into train/eval
        # HotpotQA train split has ~90k examples; use first 5000 for efficiency
        subset_size = min(5000, len(ds))
        ds = ds.select(range(subset_size))

        all_data = []
        for row in ds:
            # Build the paragraph lookup from supporting context
            paragraphs = {}
            for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
                paragraphs[title.lower().strip()] = " ".join(sentences)

            all_data.append(
                {
                    "question": row["question"],
                    "answer": row["answer"],
                    "type": row["type"],  # "bridge" or "comparison"
                    "level": row["level"],  # "easy", "medium", "hard"
                    "supporting_facts_title": list(set(row["supporting_facts"]["title"])),
                    "paragraphs": paragraphs,
                }
            )

        # Shuffle and split 90/10
        rng = random.Random(self._seed)
        rng.shuffle(all_data)

        split_point = int(len(all_data) * 0.9)
        self._train_data = all_data[:split_point]
        self._test_data = all_data[split_point:]

    def reset_episode(self) -> TaskPrompt:
        """Start a new HotpotQA episode."""
        if not self._train_data:
            self.load_dataset()

        item = self._train_data[self._train_idx % len(self._train_data)]
        self._train_idx += 1

        # Set up environment state for this episode
        self._current_paragraphs = item["paragraphs"]
        self._current_page = None

        prompt_text = (
            REACT_SYSTEM_PROMPT + "\n" + REACT_QUESTION_TEMPLATE.format(question=item["question"])
        )

        return TaskPrompt(
            text=prompt_text,
            metadata={
                "question": item["question"],
                "answer": item["answer"],
                "type": item["type"],
                "supporting_facts_title": item["supporting_facts_title"],
                "paragraphs": item["paragraphs"],
            },
        )

    def step(self, action_text: str, metadata: Dict[str, Any]) -> StepResult:
        """Execute the agent's action in the simulated environment.

        Parses the ReAct-format action and returns the appropriate observation.
        """
        # Restore environment state from metadata
        self._current_paragraphs = metadata["paragraphs"]

        # Parse the action
        match = _ACTION_RE.search(action_text)
        if not match:
            return StepResult(
                observation="Invalid action format. Use: Action: Search[query], Lookup[term], or Finish[answer]",
                reward=0.0,
                done=False,
                info={"action_type": "invalid"},
            )

        action_type = match.group(1).lower()
        action_arg = match.group(2).strip()

        if action_type == "search":
            return self._do_search(action_arg)
        elif action_type == "lookup":
            return self._do_lookup(action_arg)
        elif action_type == "finish":
            return self._do_finish(action_arg, metadata)
        else:
            return StepResult(
                observation=f"Unknown action: {action_type}",
                reward=0.0,
                done=False,
                info={"action_type": "unknown"},
            )

    def _do_search(self, query: str) -> StepResult:
        """Simulate a Wikipedia search."""
        query_lower = query.lower().strip()

        # Try exact match first
        if query_lower in self._current_paragraphs:
            result = self._current_paragraphs[query_lower]
            self._current_page = result
            return StepResult(
                observation=result,
                reward=0.0,
                done=False,
                info={"action_type": "search", "found": True},
            )

        # Try fuzzy match (substring matching on titles)
        matches = []
        for title, content in self._current_paragraphs.items():
            if query_lower in title or title in query_lower:
                matches.append((title, content))

        if matches:
            # Return the best match (shortest title that contains query)
            matches.sort(key=lambda x: len(x[0]))
            result = matches[0][1]
            self._current_page = result
            return StepResult(
                observation=result,
                reward=0.0,
                done=False,
                info={"action_type": "search", "found": True},
            )

        # No match found - return a "not found" message
        available = list(self._current_paragraphs.keys())[: self._max_search_results]
        self._current_page = None
        return StepResult(
            observation=f"No results found for '{query}'. Try searching for one of: {', '.join(available)}",
            reward=0.0,
            done=False,
            info={"action_type": "search", "found": False},
        )

    def _do_lookup(self, term: str) -> StepResult:
        """Look up a term in the current page."""
        if self._current_page is None:
            return StepResult(
                observation="No page is currently loaded. Use Search[query] first.",
                reward=0.0,
                done=False,
                info={"action_type": "lookup", "found": False},
            )

        term_lower = term.lower()
        sentences = self._current_page.split(". ")

        # Find sentences containing the term
        matches = [s for s in sentences if term_lower in s.lower()]

        if matches:
            result = ". ".join(matches[:2])  # Return up to 2 matching sentences
            return StepResult(
                observation=result,
                reward=0.0,
                done=False,
                info={"action_type": "lookup", "found": True},
            )

        return StepResult(
            observation=f"Term '{term}' not found in the current page.",
            reward=0.0,
            done=False,
            info={"action_type": "lookup", "found": False},
        )

    def _do_finish(self, answer: str, metadata: Dict[str, Any]) -> StepResult:
        """Submit the final answer and compute reward."""
        ground_truth = metadata["answer"]

        # Compute F1-based reward
        f1 = _compute_f1(answer, ground_truth)
        em = _compute_em(answer, ground_truth)

        # Reward: EM gives 1.0, F1 gives partial credit (scaled 0.1 to 0.9)
        if em > 0:
            reward = 1.0
        elif f1 > 0:
            reward = 0.1 + 0.8 * f1  # Scale F1 to [0.1, 0.9]
        else:
            reward = 0.1  # Format reward for at least finishing properly

        return StepResult(
            observation=f"Episode complete. Your answer: {answer}",
            reward=reward,
            done=True,
            info={
                "action_type": "finish",
                "answer": answer,
                "ground_truth": ground_truth,
                "f1": f1,
                "em": em,
            },
        )

    def compute_episode_reward(self, steps: List[StepResult], metadata: Dict[str, Any]) -> float:
        """Compute final reward for a complete episode.

        Uses the reward from the Finish action if present,
        otherwise penalizes for not finishing (timeout).
        """
        # Check if agent finished properly
        for step in reversed(steps):
            if step.info.get("action_type") == "finish":
                return step.reward

        # Agent didn't finish (ran out of turns) - small penalty
        return 0.0

    def get_eval_prompts(self, n: int) -> List[TaskPrompt]:
        """Get evaluation episode prompts."""
        if not self._test_data:
            self.load_dataset()

        prompts = []
        for i in range(min(n, len(self._test_data))):
            idx = (self._test_idx + i) % len(self._test_data)
            item = self._test_data[idx]

            prompt_text = (
                REACT_SYSTEM_PROMPT
                + "\n"
                + REACT_QUESTION_TEMPLATE.format(question=item["question"])
            )
            prompts.append(
                TaskPrompt(
                    text=prompt_text,
                    metadata={
                        "question": item["question"],
                        "answer": item["answer"],
                        "type": item["type"],
                        "supporting_facts_title": item["supporting_facts_title"],
                        "paragraphs": item["paragraphs"],
                    },
                )
            )
        self._test_idx = (self._test_idx + n) % len(self._test_data)
        return prompts
