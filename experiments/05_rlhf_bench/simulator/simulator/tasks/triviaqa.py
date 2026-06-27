"""TriviaQA Agent RL task: knowledge-intensive QA with search tool.

The agent answers trivia questions by searching a knowledge base (Wikipedia
evidence paragraphs provided in the dataset). This tests the agent's ability
to identify what information is needed and retrieve it efficiently.

Tools available to the agent:
- Search[query]: Search the knowledge base for relevant information.
- Finish[answer]: Submit the final answer.

Reward:
- 1.0 for exact match with any of the accepted answer aliases
- Partial credit based on token overlap (F1)
- 0.1 format reward for submitting via Finish[]
"""

from __future__ import annotations

import random
import re
import string
from typing import Any, Dict, List

import datasets

from .agent_task import AgentTask, StepResult
from .base import TaskPrompt


# ─── System prompt ─────────────────────────────────────────────────

TRIVIAQA_SYSTEM_PROMPT = """\
You are a trivia expert assistant. Answer the given question by searching for relevant information.

You have access to the following tools:
- Search[query]: Search the knowledge base for information related to your query.
- Finish[answer]: Submit your final answer.

Use the following format for each step:
Thought: <your reasoning about what information you need>
Action: <one of Search[...] or Finish[...]>

Rules:
- Search for key entities or concepts mentioned in the question.
- You can search multiple times to gather more information.
- When confident, call Finish[answer] with a concise answer (a name, date, place, etc.).
"""

TRIVIAQA_QUESTION_TEMPLATE = "Question: {question}\n\nThought:"

# ─── Helpers ────────────────────────────────────────────────────────

_ACTION_RE = re.compile(r"Action:\s*(Search|Finish)\[(.+?)\]", flags=re.IGNORECASE | re.DOTALL)


def _normalize_answer(s: str) -> str:
    """Normalize for comparison."""
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = " ".join(s.split())
    return s


def _compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1."""
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


def _check_answer(prediction: str, aliases: List[str]) -> tuple[bool, float]:
    """Check if prediction matches any alias. Returns (exact_match, best_f1)."""
    pred_norm = _normalize_answer(prediction)
    best_f1 = 0.0
    for alias in aliases:
        if _normalize_answer(alias) == pred_norm:
            return True, 1.0
        f1 = _compute_f1(prediction, alias)
        best_f1 = max(best_f1, f1)
    return False, best_f1


class TriviaQATask(AgentTask):
    """TriviaQA with search tool for knowledge retrieval.

    Dataset: trivia_qa (rc.nocontext variant for questions, with evidence for search)
    Interaction: Multi-turn with Search/Finish tools
    Reward: EM + F1 partial credit against answer aliases
    """

    def __init__(
        self,
        dataset_name: str = "mandarjoshi/trivia_qa",
        dataset_config: str = "rc",
        seed: int = 42,
    ):
        self._dataset_name = dataset_name
        self._dataset_config = dataset_config
        self._seed = seed

        self._train_data: List[Dict[str, Any]] = []
        self._test_data: List[Dict[str, Any]] = []
        self._train_idx = 0
        self._test_idx = 0

        # Per-episode state
        self._current_evidence: Dict[str, str] = {}  # keyword -> evidence text

    def load_dataset(self) -> None:
        """Load TriviaQA dataset."""
        ds = datasets.load_dataset(
            self._dataset_name,
            self._dataset_config,
            split="train",
        )

        # Use a subset for efficiency (TriviaQA train has ~138k examples)
        subset_size = min(5000, len(ds))
        ds = ds.select(range(subset_size))

        all_data = []
        for row in ds:
            # Extract answer aliases
            answer_data = row["answer"]
            aliases = list(
                set(
                    [answer_data["value"]]
                    + answer_data.get("aliases", [])
                    + answer_data.get("normalized_aliases", [])
                )
            )

            # Build evidence from entity_pages (Wikipedia content)
            evidence = {}
            entity_pages = row.get("entity_pages", {})
            if entity_pages:
                titles = entity_pages.get("title", [])
                contents = entity_pages.get("wiki_context", [])
                for title, content in zip(titles, contents):
                    if title and content:
                        # Take first 500 chars as summary
                        summary = content[:500].strip()
                        if summary:
                            evidence[title.lower().strip()] = summary

            # Also use search_results if available
            search_results = row.get("search_results", {})
            if search_results:
                sr_titles = search_results.get("title", [])
                sr_snippets = search_results.get("search_context", [])
                for title, snippet in zip(sr_titles, sr_snippets):
                    if title and snippet:
                        key = title.lower().strip()
                        if key not in evidence:
                            evidence[key] = snippet[:500].strip()

            if aliases and evidence:
                all_data.append(
                    {
                        "question": row["question"],
                        "answer": answer_data["value"],
                        "aliases": aliases,
                        "evidence": evidence,
                    }
                )

        # Shuffle and split
        rng = random.Random(self._seed)
        rng.shuffle(all_data)

        split_point = int(len(all_data) * 0.9)
        self._train_data = all_data[:split_point]
        self._test_data = all_data[split_point:]

    def reset_episode(self) -> TaskPrompt:
        """Start a new TriviaQA episode."""
        if not self._train_data:
            self.load_dataset()

        item = self._train_data[self._train_idx % len(self._train_data)]
        self._train_idx += 1

        self._current_evidence = item["evidence"]

        prompt_text = (
            TRIVIAQA_SYSTEM_PROMPT
            + "\n"
            + TRIVIAQA_QUESTION_TEMPLATE.format(question=item["question"])
        )

        return TaskPrompt(
            text=prompt_text,
            metadata={
                "question": item["question"],
                "answer": item["answer"],
                "aliases": item["aliases"],
                "evidence": item["evidence"],
            },
        )

    def step(self, action_text: str, metadata: Dict[str, Any]) -> StepResult:
        """Execute Search or Finish action."""
        self._current_evidence = metadata["evidence"]

        match = _ACTION_RE.search(action_text)
        if not match:
            return StepResult(
                observation="Invalid action format. Use: Action: Search[query] or Action: Finish[answer]",
                reward=0.0,
                done=False,
                info={"action_type": "invalid"},
            )

        action_type = match.group(1).lower()
        action_arg = match.group(2).strip()

        if action_type == "search":
            return self._do_search(action_arg)
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
        """Search the knowledge base."""
        query_lower = query.lower().strip()

        # Try exact title match
        if query_lower in self._current_evidence:
            return StepResult(
                observation=self._current_evidence[query_lower],
                reward=0.0,
                done=False,
                info={"action_type": "search", "found": True},
            )

        # Fuzzy match on titles
        matches = []
        for title, content in self._current_evidence.items():
            # Check if query words overlap with title
            query_words = set(query_lower.split())
            title_words = set(title.split())
            overlap = query_words & title_words
            if overlap or query_lower in title or title in query_lower:
                score = len(overlap) / max(len(query_words), 1)
                matches.append((score, title, content))

        if matches:
            matches.sort(key=lambda x: -x[0])
            return StepResult(
                observation=matches[0][2],
                reward=0.0,
                done=False,
                info={"action_type": "search", "found": True},
            )

        # Search within content (keyword search in evidence text)
        for title, content in self._current_evidence.items():
            if query_lower in content.lower():
                # Find relevant sentences
                sentences = content.split(". ")
                relevant = [s for s in sentences if query_lower in s.lower()]
                if relevant:
                    return StepResult(
                        observation=". ".join(relevant[:3]),
                        reward=0.0,
                        done=False,
                        info={"action_type": "search", "found": True},
                    )

        # Nothing found - show available topics
        available = list(self._current_evidence.keys())[:3]
        return StepResult(
            observation=f"No results found for '{query}'. Available topics: {', '.join(available)}",
            reward=0.0,
            done=False,
            info={"action_type": "search", "found": False},
        )

    def _do_finish(self, answer: str, metadata: Dict[str, Any]) -> StepResult:
        """Submit the final answer."""
        aliases = metadata["aliases"]

        em, best_f1 = _check_answer(answer, aliases)

        if em:
            reward = 1.0
        elif best_f1 > 0.5:
            reward = 0.1 + 0.8 * best_f1
        elif best_f1 > 0:
            reward = 0.1 + 0.4 * best_f1
        else:
            reward = 0.1  # Format reward

        return StepResult(
            observation=f"Episode complete. Your answer: {answer}",
            reward=reward,
            done=True,
            info={
                "action_type": "finish",
                "answer": answer,
                "ground_truth": metadata["answer"],
                "exact_match": em,
                "f1": best_f1,
            },
        )

    def compute_episode_reward(self, steps: List[StepResult], metadata: Dict[str, Any]) -> float:
        """Episode reward from Finish action."""
        for step in reversed(steps):
            if step.info.get("action_type") == "finish":
                return step.reward
        return 0.0

    def get_eval_prompts(self, n: int) -> List[TaskPrompt]:
        """Get evaluation prompts."""
        if not self._test_data:
            self.load_dataset()

        prompts = []
        for i in range(min(n, len(self._test_data))):
            idx = (self._test_idx + i) % len(self._test_data)
            item = self._test_data[idx]
            prompt_text = (
                TRIVIAQA_SYSTEM_PROMPT
                + "\n"
                + TRIVIAQA_QUESTION_TEMPLATE.format(question=item["question"])
            )
            prompts.append(
                TaskPrompt(
                    text=prompt_text,
                    metadata={
                        "question": item["question"],
                        "answer": item["answer"],
                        "aliases": item["aliases"],
                        "evidence": item["evidence"],
                    },
                )
            )
        self._test_idx = (self._test_idx + n) % len(self._test_data)
        return prompts
