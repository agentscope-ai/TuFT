"""ToolBench Lite Agent RL task: API calling chains with diverse real-world tools.

The agent must plan and execute a sequence of API calls to answer complex user
queries. This uses a curated subset of ToolBench (Qin et al., 2023) with
mock API responses derived from successful trajectories in the dataset.

Tools available to the agent:
- CallAPI[tool_name(param1='value1', param2='value2')]: Call a specific API
- Finish[answer]: Submit the final answer to the user

Reward:
- Based on API call sequence matching against ground truth trajectory
- 1.0: All APIs called correctly with matching parameters
- Partial credit for partially correct sequences
- 0.1: Format reward for proper Finish

Dataset: tuandunghcmut/toolbench-v1 (HuggingFace, falls back to built-in scenarios)
"""

from __future__ import annotations

import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from .agent_task import AgentTask, StepResult
from .base import TaskPrompt


logger = logging.getLogger(__name__)

# ─── System prompt ─────────────────────────────────────────────────

TOOLBENCH_SYSTEM_PROMPT = """\
You are an AI assistant that solves user tasks by calling APIs step by step.

Available tools:
{tool_descriptions}

For each step, use the following format:
Thought: <analyze what API to call next and why>
Action: CallAPI[ToolName(param1='value1', param2='value2')]

When you have enough information to answer the user, use:
Thought: <summarize your findings>
Action: Finish[your answer to the user]

Rules:
- Call ONE API per turn, then wait for the response.
- Use exact tool names and parameter names as listed.
- Build on previous API results to inform next calls.
- Finish with a comprehensive answer once you have all needed information.
"""

TOOLBENCH_QUESTION_TEMPLATE = "User query: {query}\n\nThought:"

# ─── Parsing ────────────────────────────────────────────────────────

_ACTION_RE = re.compile(r"Action:\s*(CallAPI|Finish)\[(.+?)\]", flags=re.IGNORECASE | re.DOTALL)
_API_CALL_RE = re.compile(r"(\w+)\(([^)]*)\)", flags=re.DOTALL)
_PARAM_RE = re.compile(r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|(\S+))")


def _parse_api_call(call_str: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """Parse 'ToolName(param1='val1', param2='val2')' into (name, params)."""
    match = _API_CALL_RE.match(call_str.strip())
    if not match:
        return None
    tool_name = match.group(1)
    params_str = match.group(2)
    params = {}
    for m in _PARAM_RE.finditer(params_str):
        key = m.group(1)
        value = m.group(2) or m.group(3) or m.group(4) or ""
        params[key] = value
    return tool_name, params


# ─── Built-in tool definitions (curated subset) ────────────────────

_BUILTIN_TOOLS = {
    "WikiSearch": {
        "description": "Search Wikipedia for information about a topic.",
        "parameters": {"query": "Search query string"},
    },
    "WebSearch": {
        "description": "Search the web for current information.",
        "parameters": {"query": "Search query", "num_results": "Number of results (default 5)"},
    },
    "GetMovieInfo": {
        "description": "Get information about a movie (plot, cast, ratings).",
        "parameters": {"title": "Movie title", "year": "Release year (optional)"},
    },
    "GetWeatherForecast": {
        "description": "Get weather forecast for a location.",
        "parameters": {
            "location": "City or location name",
            "days": "Number of forecast days (1-7)",
        },
    },
    "GetRecipe": {
        "description": "Search for a recipe by dish name or ingredients.",
        "parameters": {"query": "Dish name or ingredients", "cuisine": "Cuisine type (optional)"},
    },
    "GetSportsScore": {
        "description": "Get latest scores for a sports team or event.",
        "parameters": {"team": "Team name", "sport": "Sport type (optional)"},
    },
    "GetExchangeRate": {
        "description": "Get currency exchange rate.",
        "parameters": {
            "from_currency": "Source currency code",
            "to_currency": "Target currency code",
            "amount": "Amount to convert",
        },
    },
    "SearchProduct": {
        "description": "Search for a product with price comparison.",
        "parameters": {
            "query": "Product name or description",
            "max_price": "Maximum price (optional)",
        },
    },
    "GetBookInfo": {
        "description": "Get information about a book (author, summary, ratings).",
        "parameters": {"title": "Book title", "author": "Author name (optional)"},
    },
    "GetMusicInfo": {
        "description": "Get information about a song or artist.",
        "parameters": {"query": "Song title or artist name", "type": "Type: 'song' or 'artist'"},
    },
    "GetPlaceInfo": {
        "description": "Get information about a place or landmark.",
        "parameters": {
            "name": "Place or landmark name",
            "type": "Type: 'restaurant', 'hotel', 'landmark', etc.",
        },
    },
    "CalculateMath": {
        "description": "Perform mathematical calculations.",
        "parameters": {"expression": "Mathematical expression to evaluate"},
    },
    "GetHealthInfo": {
        "description": "Get health and medical information about a condition or medication.",
        "parameters": {"query": "Health topic, condition, or medication name"},
    },
    "GetStockInfo": {
        "description": "Get stock market information for a company.",
        "parameters": {
            "symbol": "Stock ticker symbol",
            "info_type": "Type: 'price', 'history', 'news'",
        },
    },
    "TranslateText": {
        "description": "Translate text between languages.",
        "parameters": {"text": "Text to translate", "target_language": "Target language"},
    },
}

# ─── Built-in scenarios with multi-step API chains ─────────────────

_BUILTIN_SCENARIOS = [
    {
        "query": "I'm planning a trip to Paris next week. What's the weather forecast and what are some must-see landmarks?",  # noqa: E501
        "available_tools": ["GetWeatherForecast", "GetPlaceInfo", "WikiSearch"],
        "ground_truth_chain": [
            {"tool": "GetWeatherForecast", "params": {"location": "Paris", "days": "7"}},
            {"tool": "GetPlaceInfo", "params": {"name": "Paris", "type": "landmark"}},
        ],
        "mock_responses": {
            "GetWeatherForecast": "Paris 7-day forecast: Mon 18°C sunny, Tue 16°C cloudy, Wed 15°C rainy, Thu 17°C partly cloudy, Fri 19°C sunny, Sat 20°C sunny, Sun 18°C cloudy.",  # noqa: E501
            "GetPlaceInfo": "Top landmarks in Paris: 1. Eiffel Tower (iconic iron tower, 330m), 2. Louvre Museum (world's largest art museum), 3. Notre-Dame Cathedral, 4. Arc de Triomphe, 5. Sacré-Cœur Basilica.",  # noqa: E501
        },
        "expected_answer": "weather and landmarks information about Paris",
    },
    {
        "query": "Compare the latest iPhone and Samsung Galaxy - which one has better specs and is cheaper?",  # noqa: E501
        "available_tools": ["SearchProduct", "WebSearch"],
        "ground_truth_chain": [
            {"tool": "SearchProduct", "params": {"query": "iPhone 15 Pro"}},
            {"tool": "SearchProduct", "params": {"query": "Samsung Galaxy S24"}},
        ],
        "mock_responses": {
            "SearchProduct": 'Results: iPhone 15 Pro - $999 (A17 Pro chip, 48MP camera, 6.1" OLED, 256GB). Samsung Galaxy S24 Ultra - $1199 (Snapdragon 8 Gen 3, 200MP camera, 6.8" AMOLED, 256GB). Galaxy S24 base - $799.',  # noqa: E501
        },
        "expected_answer": "comparison of iPhone and Samsung specs and prices",
    },
    {
        "query": "I want to cook an Italian pasta dish tonight. Find me a recipe and also tell me the nutritional benefits of olive oil.",  # noqa: E501
        "available_tools": ["GetRecipe", "GetHealthInfo", "WebSearch"],
        "ground_truth_chain": [
            {"tool": "GetRecipe", "params": {"query": "pasta", "cuisine": "Italian"}},
            {"tool": "GetHealthInfo", "params": {"query": "olive oil nutritional benefits"}},
        ],
        "mock_responses": {
            "GetRecipe": "Classic Spaghetti Aglio e Olio: Ingredients - 400g spaghetti, 6 garlic cloves, 1/2 cup olive oil, red pepper flakes, parsley. Cook pasta, sauté garlic in oil, toss together. Ready in 20 minutes.",  # noqa: E501
            "GetHealthInfo": "Olive oil benefits: Rich in monounsaturated fats, contains antioxidants (polyphenols), reduces inflammation, may lower heart disease risk, contains vitamins E and K.",  # noqa: E501
        },
        "expected_answer": "Italian pasta recipe and olive oil health benefits",
    },
    {
        "query": "What was the last movie Christopher Nolan directed? Give me its ratings and plot summary.",  # noqa: E501
        "available_tools": ["GetMovieInfo", "WebSearch", "WikiSearch"],
        "ground_truth_chain": [
            {"tool": "WebSearch", "params": {"query": "Christopher Nolan latest movie 2024"}},
            {"tool": "GetMovieInfo", "params": {"title": "Oppenheimer", "year": "2023"}},
        ],
        "mock_responses": {
            "WebSearch": "Christopher Nolan's latest film is 'Oppenheimer' (2023), starring Cillian Murphy. Won 7 Academy Awards including Best Picture and Best Director.",  # noqa: E501
            "GetMovieInfo": "Oppenheimer (2023): Director: Christopher Nolan. Cast: Cillian Murphy, Emily Blunt, Robert Downey Jr. Plot: The story of J. Robert Oppenheimer and the development of the atomic bomb. Rating: 8.4/10 (IMDb), 93% (RT).",  # noqa: E501
        },
        "expected_answer": "Oppenheimer movie info with ratings and plot",
    },
    {
        "query": "How much is 5000 Japanese Yen in US dollars and Euros?",
        "available_tools": ["GetExchangeRate", "CalculateMath"],
        "ground_truth_chain": [
            {
                "tool": "GetExchangeRate",
                "params": {"from_currency": "JPY", "to_currency": "USD", "amount": "5000"},
            },
            {
                "tool": "GetExchangeRate",
                "params": {"from_currency": "JPY", "to_currency": "EUR", "amount": "5000"},
            },
        ],
        "mock_responses": {
            "GetExchangeRate": "5000 JPY = 33.42 USD (rate: 0.006684). 5000 JPY = 30.85 EUR (rate: 0.006170).",  # noqa: E501
        },
        "expected_answer": "currency conversion results for 5000 JPY",
    },
    {
        "query": "Tell me about the book '1984' by George Orwell and translate its opening line to Spanish.",  # noqa: E501
        "available_tools": ["GetBookInfo", "TranslateText", "WikiSearch"],
        "ground_truth_chain": [
            {"tool": "GetBookInfo", "params": {"title": "1984", "author": "George Orwell"}},
            {
                "tool": "TranslateText",
                "params": {
                    "text": "It was a bright cold day in April, and the clocks were striking thirteen.",  # noqa: E501
                    "target_language": "Spanish",
                },
            },
        ],
        "mock_responses": {
            "GetBookInfo": "1984 by George Orwell (1949): A dystopian novel set in a totalitarian society ruled by Big Brother. Themes: surveillance, censorship, truth manipulation. Rating: 4.19/5 (Goodreads). Opening: 'It was a bright cold day in April, and the clocks were striking thirteen.'",  # noqa: E501
            "TranslateText": "Translation (Spanish): 'Era un brillante día frío de abril y los relojes daban las trece.'",  # noqa: E501
        },
        "expected_answer": "book info and Spanish translation",
    },
    {
        "query": "What's the latest score for the Lakers and who are their top players this season?",  # noqa: E501
        "available_tools": ["GetSportsScore", "WebSearch"],
        "ground_truth_chain": [
            {"tool": "GetSportsScore", "params": {"team": "Lakers", "sport": "basketball"}},
            {"tool": "WebSearch", "params": {"query": "Lakers top players 2024 season stats"}},
        ],
        "mock_responses": {
            "GetSportsScore": "Latest: LA Lakers 112 - 105 Golden State Warriors. LeBron James: 28pts, 8reb, 6ast. Anthony Davis: 24pts, 12reb.",  # noqa: E501
            "WebSearch": "Lakers 2024 top performers: 1. LeBron James (25.7 ppg), 2. Anthony Davis (24.7 ppg, 12.6 rpg), 3. Austin Reaves (15.9 ppg). Team record: 47-35.",  # noqa: E501
        },
        "expected_answer": "Lakers game score and top players",
    },
    {
        "query": "I'm looking for a good Italian restaurant in New York City and want to know about local COVID guidelines for dining.",  # noqa: E501
        "available_tools": ["GetPlaceInfo", "WebSearch", "GetHealthInfo"],
        "ground_truth_chain": [
            {
                "tool": "GetPlaceInfo",
                "params": {"name": "Italian restaurant New York", "type": "restaurant"},
            },
            {
                "tool": "WebSearch",
                "params": {"query": "New York City dining COVID guidelines 2024"},
            },
        ],
        "mock_responses": {
            "GetPlaceInfo": "Top Italian restaurants in NYC: 1. Carbone (Greenwich Village, $$$, 4.6★), 2. L'Artusi (West Village, $$, 4.5★), 3. Don Angie (West Village, $$, 4.4★), 4. Via Carota (West Village, $$, 4.5★).",  # noqa: E501
            "WebSearch": "NYC 2024: No COVID dining restrictions. Indoor dining fully open. Vaccination not required. Masking optional at all venues.",  # noqa: E501
        },
        "expected_answer": "NYC Italian restaurants and COVID dining rules",
    },
    {
        "query": "What songs did Taylor Swift release in her latest album and what year did she start her career?",  # noqa: E501
        "available_tools": ["GetMusicInfo", "WikiSearch", "WebSearch"],
        "ground_truth_chain": [
            {"tool": "GetMusicInfo", "params": {"query": "Taylor Swift", "type": "artist"}},
            {"tool": "WebSearch", "params": {"query": "Taylor Swift latest album 2024 songs"}},
        ],
        "mock_responses": {
            "GetMusicInfo": "Taylor Swift: American singer-songwriter. Career start: 2006 (debut single 'Tim McGraw'). Genre: Pop, Country. Albums: 11 studio albums. Latest: 'The Tortured Poets Department' (2024).",  # noqa: E501
            "WebSearch": "The Tortured Poets Department (2024) tracks: 1. Fortnight (ft. Post Malone), 2. The Tortured Poets Department, 3. My Boy Only Breaks His Favorite Toys, 4. Down Bad, 5. So Long, London...",  # noqa: E501
        },
        "expected_answer": "Taylor Swift latest album songs and career start",
    },
    {
        "query": "Calculate the compound interest on $10,000 at 5% annual rate for 3 years, and find me the best savings account rates.",  # noqa: E501
        "available_tools": ["CalculateMath", "WebSearch", "SearchProduct"],
        "ground_truth_chain": [
            {"tool": "CalculateMath", "params": {"expression": "10000 * (1 + 0.05)^3 - 10000"}},
            {"tool": "WebSearch", "params": {"query": "best savings account interest rates 2024"}},
        ],
        "mock_responses": {
            "CalculateMath": "Result: 1576.25 (compound interest on $10,000 at 5% for 3 years. Final amount: $11,576.25)",  # noqa: E501
            "WebSearch": "Best savings rates 2024: 1. Marcus by Goldman Sachs: 4.40% APY, 2. Ally Bank: 4.25% APY, 3. Discover: 4.25% APY, 4. Capital One: 4.10% APY.",  # noqa: E501
        },
        "expected_answer": "compound interest calculation and best savings rates",
    },
    {
        "query": "What's the current Tesla stock price and what are analysts saying about it?",
        "available_tools": ["GetStockInfo", "WebSearch"],
        "ground_truth_chain": [
            {"tool": "GetStockInfo", "params": {"symbol": "TSLA", "info_type": "price"}},
            {"tool": "GetStockInfo", "params": {"symbol": "TSLA", "info_type": "news"}},
        ],
        "mock_responses": {
            "GetStockInfo": "TSLA: $248.50 (+2.3%). 52-week range: $138.80 - $299.29. Market cap: $790B. Analyst consensus: Hold. Price target: $265 (avg).",  # noqa: E501
        },
        "expected_answer": "Tesla stock price and analyst opinions",
    },
    {
        "query": "Find information about the health benefits of green tea and suggest where to buy organic green tea online.",  # noqa: E501
        "available_tools": ["GetHealthInfo", "SearchProduct", "WebSearch"],
        "ground_truth_chain": [
            {"tool": "GetHealthInfo", "params": {"query": "green tea health benefits"}},
            {"tool": "SearchProduct", "params": {"query": "organic green tea"}},
        ],
        "mock_responses": {
            "GetHealthInfo": "Green tea benefits: Rich in catechins (EGCG), boosts metabolism, improves brain function, reduces cancer risk, lowers cholesterol, contains L-theanine for relaxation. Recommended: 3-5 cups daily.",  # noqa: E501
            "SearchProduct": "Organic green tea: 1. Tealyra Gyokuro ($28/100g, 4.7★), 2. Jade Leaf Matcha ($9.95/30g, 4.5★), 3. Traditional Medicinals ($6.49/16 bags, 4.6★), 4. Rishi Tea Sencha ($12/50g, 4.4★).",  # noqa: E501
        },
        "expected_answer": "green tea health benefits and buying options",
    },
]


class ToolBenchTask(AgentTask):
    """ToolBench Lite: multi-step API calling chains.

    Evaluates the agent's ability to:
    1. Plan a sequence of API calls to answer complex queries
    2. Select appropriate tools from a diverse set
    3. Use API results to inform subsequent calls
    4. Synthesize information into a final answer

    Uses built-in scenarios inspired by ToolBench, with optional
    HuggingFace dataset augmentation.
    """

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._train_data: List[Dict[str, Any]] = []
        self._test_data: List[Dict[str, Any]] = []
        self._train_idx = 0
        self._test_idx = 0

        # Per-episode state
        self._call_history: List[Dict[str, Any]] = []

    def load_dataset(self) -> None:
        """Load ToolBench scenarios."""
        all_data = self._load_builtin_scenarios()

        # Try HuggingFace augmentation
        try:
            hf_data = self._load_from_huggingface()
            if hf_data:
                all_data.extend(hf_data)
                logger.info(f"Loaded {len(hf_data)} scenarios from HuggingFace ToolBench")
        except Exception as e:
            logger.info(f"Using built-in scenarios only (HF: {e})")

        # Shuffle and split
        rng = random.Random(self._seed)
        rng.shuffle(all_data)

        split_point = int(len(all_data) * 0.85)
        self._train_data = all_data[:split_point]
        self._test_data = all_data[split_point:]

        logger.info(f"ToolBench loaded: {len(self._train_data)} train, {len(self._test_data)} test")

    def _load_builtin_scenarios(self) -> List[Dict[str, Any]]:
        """Load built-in scenarios with variations."""
        scenarios = []
        rng = random.Random(self._seed)

        for base in _BUILTIN_SCENARIOS:
            scenarios.append(base)
            # Create variations with different distractor tools
            for _ in range(3):
                variation = dict(base)
                tools = list(base["available_tools"])
                # Add 1-2 distractor tools
                all_tools = list(_BUILTIN_TOOLS.keys())
                distractors = [t for t in all_tools if t not in tools]
                n_distractors = rng.randint(1, 2)
                extras = rng.sample(distractors, min(n_distractors, len(distractors)))
                variation["available_tools"] = tools + extras
                rng.shuffle(variation["available_tools"])
                scenarios.append(variation)

        return scenarios

    def _load_from_huggingface(self) -> List[Dict[str, Any]]:
        """Try to load from tuandunghcmut/toolbench-v1."""
        try:
            import datasets as ds_lib

            dataset = ds_lib.load_dataset(
                "tuandunghcmut/toolbench-v1",
                split="train",
            )

            scenarios = []
            for row in dataset.select(range(min(500, len(dataset)))):
                scenario = self._parse_toolbench_row(row)
                if scenario:
                    scenarios.append(scenario)

            return scenarios[:200]  # Cap at 200

        except Exception as e:
            logger.debug(f"ToolBench HF load failed: {e}")
            return []

    def _parse_toolbench_row(self, row: Dict) -> Optional[Dict[str, Any]]:
        """Parse a ToolBench conversation row into our format."""
        try:
            conversations = row.get("conversations", [])
            if not conversations or len(conversations) < 3:
                return None

            # Extract user query (first human message)
            user_query = None
            tool_calls = []
            tool_responses = {}

            for msg in conversations:
                role = msg.get("from", msg.get("role", ""))
                content = msg.get("value", msg.get("content", ""))

                if role in ("human", "user") and not user_query:
                    user_query = content
                elif role in ("gpt", "assistant") and content:
                    # Try to extract tool calls
                    calls = re.findall(r"(\w+)\(([^)]*)\)", content)
                    for call_name, call_params in calls:
                        params = {}
                        for m in _PARAM_RE.finditer(call_params):
                            params[m.group(1)] = m.group(2) or m.group(3) or m.group(4) or ""
                        tool_calls.append({"tool": call_name, "params": params})
                        tool_responses[call_name] = "API call executed successfully."
                elif role in ("tool", "function") and content:
                    # Map response to last tool call
                    if tool_calls:
                        last_tool = tool_calls[-1]["tool"]
                        tool_responses[last_tool] = content[:300]

            if not user_query or not tool_calls:
                return None

            # Map tool calls to our built-in tools (best-effort matching)
            available = list(set(tc["tool"] for tc in tool_calls))[:4]
            # Add some of our builtin tools as alternatives
            available.extend(
                random.sample(list(_BUILTIN_TOOLS.keys()), min(2, len(_BUILTIN_TOOLS)))
            )
            available = list(set(available))[:5]

            return {
                "query": user_query[:500],
                "available_tools": available,
                "ground_truth_chain": tool_calls[:3],
                "mock_responses": tool_responses,
                "expected_answer": "task completed based on API results",
            }

        except Exception:
            return None

    def reset_episode(self) -> TaskPrompt:
        """Start a new ToolBench episode."""
        if not self._train_data:
            self.load_dataset()

        item = self._train_data[self._train_idx % len(self._train_data)]
        self._train_idx += 1

        self._call_history = []

        # Build tool descriptions
        tool_descs = []
        for tool_name in item["available_tools"]:
            if tool_name in _BUILTIN_TOOLS:
                tool_info = _BUILTIN_TOOLS[tool_name]
                params_desc = ", ".join(f"{k}: {v}" for k, v in tool_info["parameters"].items())
                tool_descs.append(
                    f"- {tool_name}: {tool_info['description']}\n  Parameters: {params_desc}"
                )
            else:
                tool_descs.append(
                    f"- {tool_name}: A utility API tool.\n  Parameters: (use as needed)"
                )

        tool_descriptions = "\n".join(tool_descs)
        system_prompt = TOOLBENCH_SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)
        prompt_text = system_prompt + "\n" + TOOLBENCH_QUESTION_TEMPLATE.format(query=item["query"])

        return TaskPrompt(
            text=prompt_text,
            metadata={
                "scenario": item,
                "ground_truth_chain": item["ground_truth_chain"],
            },
        )

    def step(self, action_text: str, metadata: Dict[str, Any]) -> StepResult:
        """Execute a tool call or finish."""
        scenario = metadata["scenario"]

        match = _ACTION_RE.search(action_text)
        if not match:
            return StepResult(
                observation="Invalid format. Use: Action: CallAPI[ToolName(params)] or Action: Finish[answer]",  # noqa: E501
                reward=0.0,
                done=False,
                info={"action_type": "invalid"},
            )

        action_type = match.group(1).lower()
        action_arg = match.group(2).strip()

        if action_type == "callapi":
            return self._do_call_api(action_arg, scenario)
        elif action_type == "finish":
            return self._do_finish(action_arg, scenario)
        else:
            return StepResult(
                observation=f"Unknown action: {action_type}",
                reward=0.0,
                done=False,
                info={"action_type": "unknown"},
            )

    def _do_call_api(self, call_str: str, scenario: Dict) -> StepResult:
        """Execute an API call."""
        parsed = _parse_api_call(call_str)
        if not parsed:
            return StepResult(
                observation=f"Failed to parse: '{call_str}'. Use format: ToolName(param='value')",
                reward=0.0,
                done=False,
                info={"action_type": "callapi", "success": False},
            )

        tool_name, params = parsed
        self._call_history.append({"tool": tool_name, "params": params})

        # Get mock response
        mock_responses = scenario.get("mock_responses", {})
        if tool_name in mock_responses:
            response = mock_responses[tool_name]
        else:
            # Generic response for unknown tools
            response = f"{tool_name} executed. Result: Information retrieved successfully."

        return StepResult(
            observation=f"API Response: {response}",
            reward=0.0,
            done=False,
            info={"action_type": "callapi", "tool": tool_name, "params": params, "success": True},
        )

    def _do_finish(self, answer: str, scenario: Dict) -> StepResult:
        """Finish and compute reward based on call chain matching."""
        gt_chain = scenario.get("ground_truth_chain", [])

        if not gt_chain:
            reward = 0.5
        else:
            # Score: proportion of ground truth calls that were made
            matched = 0
            for gt_call in gt_chain:
                gt_tool = gt_call["tool"]
                gt_params = gt_call.get("params", {})

                for hist in self._call_history:
                    if hist["tool"].lower() == gt_tool.lower():
                        # Tool name matches - check params
                        param_score = self._param_overlap(hist["params"], gt_params)
                        if param_score > 0.3:
                            matched += 1
                            break

            chain_score = matched / len(gt_chain)
            # Also give credit for finishing properly
            reward = 0.1 + 0.9 * chain_score

        return StepResult(
            observation="Episode complete.",
            reward=reward,
            done=True,
            info={
                "action_type": "finish",
                "answer": answer,
                "call_history": self._call_history,
                "reward": reward,
            },
        )

    def _param_overlap(self, pred: Dict[str, str], gt: Dict[str, str]) -> float:
        """Compute parameter overlap between predicted and ground truth."""
        if not gt:
            return 1.0
        if not pred:
            return 0.0

        matched = 0
        for key, gt_val in gt.items():
            if key in pred:
                if pred[key].lower().strip() == gt_val.lower().strip():
                    matched += 1
                elif gt_val.lower() in pred[key].lower() or pred[key].lower() in gt_val.lower():
                    matched += 0.5
            else:
                # Check if any predicted param value matches
                for pv in pred.values():
                    if gt_val.lower() in pv.lower():
                        matched += 0.3
                        break

        return matched / len(gt)

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

            tool_descs = []
            for tool_name in item["available_tools"]:
                if tool_name in _BUILTIN_TOOLS:
                    tool_info = _BUILTIN_TOOLS[tool_name]
                    params_desc = ", ".join(f"{k}: {v}" for k, v in tool_info["parameters"].items())
                    tool_descs.append(
                        f"- {tool_name}: {tool_info['description']}\n  Parameters: {params_desc}"
                    )
                else:
                    tool_descs.append(
                        f"- {tool_name}: A utility API tool.\n  Parameters: (use as needed)"
                    )

            tool_descriptions = "\n".join(tool_descs)
            system_prompt = TOOLBENCH_SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)
            prompt_text = (
                system_prompt + "\n" + TOOLBENCH_QUESTION_TEMPLATE.format(query=item["query"])
            )

            prompts.append(
                TaskPrompt(
                    text=prompt_text,
                    metadata={
                        "scenario": item,
                        "ground_truth_chain": item["ground_truth_chain"],
                    },
                )
            )
        self._test_idx = (self._test_idx + n) % len(self._test_data)
        return prompts
