"""API-Bank Agent RL task: multi-turn tool-augmented dialogue with diverse APIs.

The agent must select the correct API and provide correct parameters based on
dialogue context. This is a comprehensive benchmark for tool-augmented LLMs
from EMNLP 2023 (Alibaba).

Tools available to the agent:
- ToolCall[ApiName(key1='value1', key2='value2')]: Call an API with parameters
- Finish[response]: Submit final response to the user

Reward:
- 1.0: Correct API name + all key parameters match ground truth
- 0.5: Correct API name but parameters partially match
- 0.1: Valid format but wrong API
- 0.0: Invalid format or no completion

Dataset: liminghao1630/API-Bank (loaded via huggingface_hub)
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from .agent_task import AgentTask, StepResult
from .base import TaskPrompt


logger = logging.getLogger(__name__)

# ─── System prompt ─────────────────────────────────────────────────

APIBANK_SYSTEM_PROMPT = """\
You are a helpful assistant that can call APIs to help users accomplish tasks.

Based on the user's request and the available APIs below, generate the appropriate API call.

{api_descriptions}

To call an API, use the following format:
Thought: <your reasoning about which API to use and what parameters are needed>
Action: ToolCall[ApiName(key1='value1', key2='value2')]

When you have the API result and can answer the user, use:
Thought: <summarize the result>
Action: Finish[your response to the user]

Rules:
- Call exactly ONE API per turn.
- Use the exact API name and parameter names as described.
- String values should be in quotes, numbers should not.
- After receiving an API response, you may call another API or Finish.
"""

APIBANK_USER_TEMPLATE = "User request: {user_query}\n\nThought:"

# ─── Regex ──────────────────────────────────────────────────────────

_ACTION_RE = re.compile(r"Action:\s*(ToolCall|Finish)\[(.+?)\]", flags=re.IGNORECASE | re.DOTALL)
_API_CALL_RE = re.compile(r"(\w+)\(([^)]*)\)", flags=re.DOTALL)
_PARAM_RE = re.compile(r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|(\S+))")


def _parse_api_call(call_str: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """Parse 'ApiName(key1='value1', key2='value2')' into (name, params)."""
    match = _API_CALL_RE.match(call_str.strip())
    if not match:
        return None

    api_name = match.group(1)
    params_str = match.group(2)

    params = {}
    for m in _PARAM_RE.finditer(params_str):
        key = m.group(1)
        value = m.group(2) or m.group(3) or m.group(4) or ""
        params[key] = value

    return api_name, params


def _compute_param_similarity(pred_params: Dict[str, str], gt_params: Dict[str, str]) -> float:
    """Compute parameter matching score between predicted and ground truth."""
    if not gt_params:
        return 1.0 if not pred_params else 0.5

    matched = 0
    total = len(gt_params)

    for key, gt_val in gt_params.items():
        if key in pred_params:
            pred_val = pred_params[key].strip().lower()
            gt_val_norm = gt_val.strip().lower()
            if pred_val == gt_val_norm:
                matched += 1
            elif pred_val in gt_val_norm or gt_val_norm in pred_val:
                matched += 0.5

    return matched / total if total > 0 else 0.0


# ─── Built-in API definitions (subset of API-Bank) ─────────────────

_BUILTIN_APIS = {
    "ToolSearcher": {
        "description": "Search for a relevant tool/API based on keywords.",
        "parameters": {"keywords": "Keywords to search for tools"},
        "example": "ToolSearcher(keywords='weather forecast')",
    },
    "GetWeather": {
        "description": "Get the current weather for a given city.",
        "parameters": {"city": "City name", "date": "Date in YYYY-MM-DD format"},
        "example": "GetWeather(city='Beijing', date='2024-03-15')",
    },
    "BookRestaurant": {
        "description": "Book a table at a restaurant.",
        "parameters": {
            "restaurant": "Restaurant name",
            "date": "Date",
            "time": "Time",
            "party_size": "Number of people",
        },
        "example": "BookRestaurant(restaurant='Olive Garden', date='2024-03-20', time='19:00', party_size='4')",  # noqa: E501
    },
    "SearchHotel": {
        "description": "Search for available hotels in a city.",
        "parameters": {
            "city": "City name",
            "check_in": "Check-in date",
            "check_out": "Check-out date",
        },
        "example": "SearchHotel(city='Shanghai', check_in='2024-04-01', check_out='2024-04-03')",
    },
    "BookHotel": {
        "description": "Book a hotel room.",
        "parameters": {
            "hotel_name": "Name of the hotel",
            "check_in": "Check-in date",
            "check_out": "Check-out date",
            "guest_name": "Guest name",
        },
        "example": "BookHotel(hotel_name='Hilton', check_in='2024-04-01', check_out='2024-04-03', guest_name='John')",  # noqa: E501
    },
    "SearchFlight": {
        "description": "Search for available flights.",
        "parameters": {
            "origin": "Departure city",
            "destination": "Arrival city",
            "date": "Travel date",
        },
        "example": "SearchFlight(origin='New York', destination='London', date='2024-05-01')",
    },
    "BookFlight": {
        "description": "Book a flight ticket.",
        "parameters": {
            "flight_number": "Flight number",
            "passenger_name": "Passenger name",
            "date": "Travel date",
        },
        "example": "BookFlight(flight_number='UA123', passenger_name='Alice', date='2024-05-01')",
    },
    "SendEmail": {
        "description": "Send an email to a recipient.",
        "parameters": {"to": "Recipient email", "subject": "Email subject", "body": "Email body"},
        "example": "SendEmail(to='user@example.com', subject='Meeting', body='See you tomorrow')",
    },
    "SetReminder": {
        "description": "Set a reminder for a specific time.",
        "parameters": {
            "content": "Reminder content",
            "time": "Reminder time in YYYY-MM-DD HH:MM format",
        },
        "example": "SetReminder(content='Team meeting', time='2024-03-20 14:00')",
    },
    "SearchRestaurant": {
        "description": "Search for restaurants by cuisine or location.",
        "parameters": {"cuisine": "Type of cuisine", "city": "City name"},
        "example": "SearchRestaurant(cuisine='Italian', city='San Francisco')",
    },
    "GetStockPrice": {
        "description": "Get the current stock price for a ticker symbol.",
        "parameters": {"symbol": "Stock ticker symbol"},
        "example": "GetStockPrice(symbol='AAPL')",
    },
    "TranslateText": {
        "description": "Translate text from one language to another.",
        "parameters": {
            "text": "Text to translate",
            "source_lang": "Source language",
            "target_lang": "Target language",
        },
        "example": "TranslateText(text='Hello', source_lang='English', target_lang='Chinese')",
    },
    "GetNews": {
        "description": "Get latest news articles on a topic.",
        "parameters": {"topic": "News topic or keyword"},
        "example": "GetNews(topic='artificial intelligence')",
    },
    "CalculateRoute": {
        "description": "Calculate driving route between two locations.",
        "parameters": {"origin": "Starting location", "destination": "End location"},
        "example": "CalculateRoute(origin='Central Park', destination='JFK Airport')",
    },
    "CreateCalendarEvent": {
        "description": "Create a calendar event.",
        "parameters": {
            "title": "Event title",
            "date": "Event date",
            "time": "Event time",
            "duration": "Duration in minutes",
        },
        "example": "CreateCalendarEvent(title='Lunch', date='2024-03-20', time='12:00', duration='60')",  # noqa: E501
    },
}

# ─── Built-in dialogue scenarios ───────────────────────────────────

_BUILTIN_SCENARIOS = [
    {
        "user_query": "What's the weather like in Beijing tomorrow?",
        "available_apis": ["GetWeather", "ToolSearcher"],
        "ground_truth_calls": [
            {"api": "GetWeather", "params": {"city": "Beijing", "date": "tomorrow"}}
        ],
        "mock_responses": {"GetWeather": "Sunny, 25°C, humidity 40%."},
        "final_answer": "The weather in Beijing tomorrow will be sunny with a temperature of 25°C and humidity of 40%.",  # noqa: E501
    },
    {
        "user_query": "I need to book a table for 4 at an Italian restaurant in San Francisco for this Friday at 7pm.",  # noqa: E501
        "available_apis": ["SearchRestaurant", "BookRestaurant", "ToolSearcher"],
        "ground_truth_calls": [
            {"api": "SearchRestaurant", "params": {"cuisine": "Italian", "city": "San Francisco"}},
            {
                "api": "BookRestaurant",
                "params": {
                    "restaurant": "Trattoria Roma",
                    "date": "this Friday",
                    "time": "19:00",
                    "party_size": "4",
                },
            },
        ],
        "mock_responses": {
            "SearchRestaurant": "Found restaurants: 1. Trattoria Roma (4.5 stars), 2. Pasta House (4.2 stars), 3. Italian Corner (4.0 stars).",  # noqa: E501
            "BookRestaurant": "Reservation confirmed at Trattoria Roma for 4 guests on Friday at 19:00. Confirmation #TR2024.",  # noqa: E501
        },
        "final_answer": "I've booked a table for 4 at Trattoria Roma for this Friday at 7pm. Your confirmation number is TR2024.",  # noqa: E501
    },
    {
        "user_query": "Find me a flight from New York to London on May 1st and book it for Alice Johnson.",  # noqa: E501
        "available_apis": ["SearchFlight", "BookFlight", "ToolSearcher"],
        "ground_truth_calls": [
            {
                "api": "SearchFlight",
                "params": {"origin": "New York", "destination": "London", "date": "2024-05-01"},
            },
            {
                "api": "BookFlight",
                "params": {
                    "flight_number": "BA178",
                    "passenger_name": "Alice Johnson",
                    "date": "2024-05-01",
                },
            },
        ],
        "mock_responses": {
            "SearchFlight": "Available flights: 1. BA178 (departs 10:30, $850), 2. UA901 (departs 18:00, $920), 3. AA100 (departs 22:00, $780).",  # noqa: E501
            "BookFlight": "Flight BA178 booked for Alice Johnson on 2024-05-01. E-ticket: BA178-AJ-2024.",  # noqa: E501
        },
        "final_answer": "I've booked flight BA178 for Alice Johnson from New York to London on May 1st. Your e-ticket number is BA178-AJ-2024.",  # noqa: E501
    },
    {
        "user_query": "Search for hotels in Shanghai from April 1-3 and book the Hilton for me. My name is John Smith.",  # noqa: E501
        "available_apis": ["SearchHotel", "BookHotel", "ToolSearcher"],
        "ground_truth_calls": [
            {
                "api": "SearchHotel",
                "params": {"city": "Shanghai", "check_in": "2024-04-01", "check_out": "2024-04-03"},
            },
            {
                "api": "BookHotel",
                "params": {
                    "hotel_name": "Hilton",
                    "check_in": "2024-04-01",
                    "check_out": "2024-04-03",
                    "guest_name": "John Smith",
                },
            },
        ],
        "mock_responses": {
            "SearchHotel": "Available hotels: 1. Hilton Shanghai ($200/night, 4.5 stars), 2. Marriott ($180/night, 4.3 stars), 3. Holiday Inn ($120/night, 4.0 stars).",  # noqa: E501
            "BookHotel": "Reservation confirmed at Hilton Shanghai for John Smith, April 1-3. Confirmation: HS-20240401-JS.",  # noqa: E501
        },
        "final_answer": "I've booked the Hilton Shanghai for you from April 1-3. Confirmation number: HS-20240401-JS.",  # noqa: E501
    },
    {
        "user_query": "Send an email to bob@company.com about tomorrow's meeting at 2pm.",
        "available_apis": ["SendEmail", "SetReminder", "ToolSearcher"],
        "ground_truth_calls": [
            {
                "api": "SendEmail",
                "params": {
                    "to": "bob@company.com",
                    "subject": "Tomorrow's Meeting",
                    "body": "Reminder: meeting at 2pm tomorrow.",
                },
            },
        ],
        "mock_responses": {"SendEmail": "Email sent successfully to bob@company.com."},
        "final_answer": "I've sent an email to bob@company.com about tomorrow's meeting at 2pm.",
    },
    {
        "user_query": "Set a reminder for our team meeting on March 20th at 2pm.",
        "available_apis": ["SetReminder", "CreateCalendarEvent", "ToolSearcher"],
        "ground_truth_calls": [
            {
                "api": "SetReminder",
                "params": {"content": "Team meeting", "time": "2024-03-20 14:00"},
            },
        ],
        "mock_responses": {
            "SetReminder": "Reminder set for 'Team meeting' on 2024-03-20 at 14:00."
        },
        "final_answer": "I've set a reminder for your team meeting on March 20th at 2pm.",
    },
    {
        "user_query": "What's Apple's current stock price?",
        "available_apis": ["GetStockPrice", "GetNews", "ToolSearcher"],
        "ground_truth_calls": [
            {"api": "GetStockPrice", "params": {"symbol": "AAPL"}},
        ],
        "mock_responses": {
            "GetStockPrice": "AAPL: $178.52 (+1.2%), Market Cap: $2.8T, Volume: 52M."
        },
        "final_answer": "Apple's current stock price is $178.52, up 1.2% today.",
    },
    {
        "user_query": "Translate 'Good morning, how are you?' to Chinese.",
        "available_apis": ["TranslateText", "ToolSearcher"],
        "ground_truth_calls": [
            {
                "api": "TranslateText",
                "params": {
                    "text": "Good morning, how are you?",
                    "source_lang": "English",
                    "target_lang": "Chinese",
                },
            },
        ],
        "mock_responses": {"TranslateText": "Translation: '早上好，你好吗？'"},
        "final_answer": "The translation is: 早上好，你好吗？",
    },
    {
        "user_query": "Get me the latest news about artificial intelligence.",
        "available_apis": ["GetNews", "ToolSearcher"],
        "ground_truth_calls": [
            {"api": "GetNews", "params": {"topic": "artificial intelligence"}},
        ],
        "mock_responses": {
            "GetNews": "Top stories: 1. 'New AI Model Breaks Records' (TechDaily), 2. 'AI in Healthcare: 2024 Outlook' (MedNews), 3. 'OpenAI Announces GPT-5' (Reuters)."  # noqa: E501
        },
        "final_answer": "Here are the latest AI news: 1. New AI Model Breaks Records, 2. AI in Healthcare: 2024 Outlook, 3. OpenAI Announces GPT-5.",  # noqa: E501
    },
    {
        "user_query": "How do I get from Central Park to JFK Airport by car?",
        "available_apis": ["CalculateRoute", "SearchFlight", "ToolSearcher"],
        "ground_truth_calls": [
            {
                "api": "CalculateRoute",
                "params": {"origin": "Central Park", "destination": "JFK Airport"},
            },
        ],
        "mock_responses": {
            "CalculateRoute": "Route: Central Park → FDR Drive → Van Wyck Expressway → JFK. Distance: 18.5 miles, estimated time: 35-55 minutes."  # noqa: E501
        },
        "final_answer": "From Central Park to JFK Airport is about 18.5 miles via FDR Drive and Van Wyck Expressway, taking approximately 35-55 minutes.",  # noqa: E501
    },
    {
        "user_query": "Create a calendar event for lunch with Sarah on March 20th at noon for 1 hour.",  # noqa: E501
        "available_apis": ["CreateCalendarEvent", "SetReminder", "ToolSearcher"],
        "ground_truth_calls": [
            {
                "api": "CreateCalendarEvent",
                "params": {
                    "title": "Lunch with Sarah",
                    "date": "2024-03-20",
                    "time": "12:00",
                    "duration": "60",
                },
            },
        ],
        "mock_responses": {
            "CreateCalendarEvent": "Event 'Lunch with Sarah' created for March 20, 2024 at 12:00 (1 hour)."  # noqa: E501
        },
        "final_answer": "I've created a calendar event 'Lunch with Sarah' for March 20th at noon, lasting 1 hour.",  # noqa: E501
    },
    {
        "user_query": "I want to find a hotel in Tokyo for next week, from Monday to Thursday, and also check the weather there on Monday.",  # noqa: E501
        "available_apis": ["SearchHotel", "GetWeather", "BookHotel", "ToolSearcher"],
        "ground_truth_calls": [
            {
                "api": "SearchHotel",
                "params": {
                    "city": "Tokyo",
                    "check_in": "next Monday",
                    "check_out": "next Thursday",
                },
            },
            {"api": "GetWeather", "params": {"city": "Tokyo", "date": "next Monday"}},
        ],
        "mock_responses": {
            "SearchHotel": "Available: 1. Tokyo Grand Hotel ($150/night), 2. Sakura Inn ($95/night), 3. Imperial Tokyo ($280/night).",  # noqa: E501
            "GetWeather": "Tokyo next Monday: Partly cloudy, 18°C, 30% chance of rain.",
        },
        "final_answer": "Hotels in Tokyo: Tokyo Grand Hotel ($150/night), Sakura Inn ($95/night), Imperial Tokyo ($280/night). Weather on Monday: partly cloudy, 18°C.",  # noqa: E501
    },
]


class APIBankTask(AgentTask):
    """API-Bank multi-turn tool calling task.

    Evaluates the agent's ability to:
    1. Select the correct API based on user intent
    2. Extract and format parameters correctly
    3. Chain multiple API calls when needed
    4. Provide a coherent final response

    Uses a built-in set of API definitions and dialogue scenarios.
    Can be extended with data from HuggingFace liminghao1630/API-Bank.
    """

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._train_data: List[Dict[str, Any]] = []
        self._test_data: List[Dict[str, Any]] = []
        self._train_idx = 0
        self._test_idx = 0

        # Per-episode state
        self._current_scenario: Dict[str, Any] = {}
        self._call_history: List[Dict[str, Any]] = []

    def load_dataset(self) -> None:
        """Load API-Bank scenarios.

        Tries to load from HuggingFace first; falls back to built-in scenarios.
        Built-in scenarios are expanded with randomized parameter variations.
        """
        all_data = self._load_builtin_scenarios()

        # Try to augment with HuggingFace data
        try:
            hf_data = self._load_from_huggingface()
            if hf_data:
                all_data.extend(hf_data)
                logger.info(f"Loaded {len(hf_data)} additional scenarios from HuggingFace")
        except Exception as e:
            logger.info(f"Using built-in scenarios only (HF load failed: {e})")

        # Shuffle and split
        rng = random.Random(self._seed)
        rng.shuffle(all_data)

        split_point = int(len(all_data) * 0.85)
        self._train_data = all_data[:split_point]
        self._test_data = all_data[split_point:]

        logger.info(
            f"API-Bank loaded: {len(self._train_data)} train, {len(self._test_data)} test scenarios"
        )

    def _load_builtin_scenarios(self) -> List[Dict[str, Any]]:
        """Load and expand built-in scenarios with variations."""
        scenarios = []
        rng = random.Random(self._seed)

        for base in _BUILTIN_SCENARIOS:
            # Add the original
            scenarios.append(base)

            # Generate parameter variations for training diversity
            for _ in range(4):
                variation = self._create_variation(base, rng)
                scenarios.append(variation)

        return scenarios

    def _create_variation(self, base: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        """Create a parameter variation of a scenario."""
        # Simple variation: shuffle available APIs order, minor query rephrasing
        variation = {
            "user_query": base["user_query"],
            "available_apis": list(base["available_apis"]),
            "ground_truth_calls": base["ground_truth_calls"],
            "mock_responses": base["mock_responses"],
            "final_answer": base["final_answer"],
        }
        rng.shuffle(variation["available_apis"])

        # Add a random extra API to available list (distractor)
        all_apis = list(_BUILTIN_APIS.keys())
        distractors = [a for a in all_apis if a not in variation["available_apis"]]
        if distractors:
            variation["available_apis"].append(rng.choice(distractors))

        return variation

    def _load_from_huggingface(self) -> List[Dict[str, Any]]:
        """Try to load additional data from HuggingFace API-Bank repository."""
        try:
            from huggingface_hub import hf_hub_download

            # Download level-1 data (simple single API calls)
            api_file = hf_hub_download(
                repo_id="liminghao1630/API-Bank",
                filename="test-data/level-1-api.json",
                repo_type="dataset",
            )
            response_file = hf_hub_download(
                repo_id="liminghao1630/API-Bank",
                filename="test-data/level-1-response.json",
                repo_type="dataset",
            )

            with open(api_file, "r") as f:
                api_data = json.load(f)
            with open(response_file, "r") as f:
                response_data = json.load(f)

            scenarios = []
            for item in api_data[:200]:  # Take subset
                scenario = self._parse_hf_item(item, response_data)
                if scenario:
                    scenarios.append(scenario)

            return scenarios

        except ImportError:
            logger.debug("huggingface_hub not available")
            return []
        except Exception as e:
            logger.debug(f"Failed to load HF data: {e}")
            return []

    def _parse_hf_item(self, item: Dict, responses: Any) -> Optional[Dict[str, Any]]:
        """Parse a HuggingFace API-Bank item into our scenario format."""
        try:
            # Expected format varies; try common patterns
            if isinstance(item, dict):
                query = item.get("input", item.get("query", item.get("dialogue", "")))
                gt_call = item.get("output", item.get("api_call", ""))

                if not query or not gt_call:
                    return None

                # Parse the ground truth API call
                parsed = _parse_api_call(gt_call.strip("[]"))
                if not parsed:
                    return None

                api_name, params = parsed

                return {
                    "user_query": query if isinstance(query, str) else str(query),
                    "available_apis": [api_name, "ToolSearcher"],
                    "ground_truth_calls": [{"api": api_name, "params": params}],
                    "mock_responses": {api_name: f"{api_name} executed successfully with result."},
                    "final_answer": "Task completed.",
                }
        except Exception:
            pass
        return None

    def reset_episode(self) -> TaskPrompt:
        """Start a new API-Bank episode."""
        if not self._train_data:
            self.load_dataset()

        item = self._train_data[self._train_idx % len(self._train_data)]
        self._train_idx += 1

        self._current_scenario = item
        self._call_history = []

        # Build API descriptions for available APIs
        api_descs = []
        for api_name in item["available_apis"]:
            if api_name in _BUILTIN_APIS:
                api_info = _BUILTIN_APIS[api_name]
                params_desc = ", ".join(f"{k}: {v}" for k, v in api_info["parameters"].items())
                api_descs.append(
                    f"- {api_name}: {api_info['description']}\n"
                    f"  Parameters: {params_desc}\n"
                    f"  Example: {api_info['example']}"
                )

        api_descriptions = "\n".join(api_descs)
        system_prompt = APIBANK_SYSTEM_PROMPT.format(api_descriptions=api_descriptions)
        prompt_text = (
            system_prompt + "\n" + APIBANK_USER_TEMPLATE.format(user_query=item["user_query"])
        )

        return TaskPrompt(
            text=prompt_text,
            metadata={
                "scenario": item,
                "ground_truth_calls": item["ground_truth_calls"],
                "call_index": 0,
            },
        )

    def step(self, action_text: str, metadata: Dict[str, Any]) -> StepResult:
        """Execute the agent's API call or finish action."""
        scenario = metadata["scenario"]
        call_index = metadata.get("call_index", 0)

        match = _ACTION_RE.search(action_text)
        if not match:
            return StepResult(
                observation="Invalid action format. Use: Action: ToolCall[ApiName(params)] or Action: Finish[response]",  # noqa: E501
                reward=0.0,
                done=False,
                info={"action_type": "invalid"},
            )

        action_type = match.group(1).lower()
        action_arg = match.group(2).strip()

        if action_type == "toolcall":
            return self._do_tool_call(action_arg, scenario, call_index, metadata)
        elif action_type == "finish":
            return self._do_finish(action_arg, scenario, metadata)
        else:
            return StepResult(
                observation=f"Unknown action: {action_type}",
                reward=0.0,
                done=False,
                info={"action_type": "unknown"},
            )

    def _do_tool_call(
        self, call_str: str, scenario: Dict, call_index: int, metadata: Dict
    ) -> StepResult:
        """Execute an API call and return mock response."""
        parsed = _parse_api_call(call_str)
        if not parsed:
            return StepResult(
                observation=f"Failed to parse API call: '{call_str}'. Use format: ApiName(key='value')",  # noqa: E501
                reward=0.0,
                done=False,
                info={"action_type": "toolcall", "success": False},
            )

        api_name, params = parsed

        # Get mock response
        mock_responses = scenario.get("mock_responses", {})
        if api_name in mock_responses:
            response = mock_responses[api_name]
        else:
            response = f"{api_name} returned: OK (no detailed response available)"

        # Record call
        self._call_history.append({"api": api_name, "params": params})

        # Update call_index in metadata for next step
        metadata["call_index"] = call_index + 1

        return StepResult(
            observation=f"API Response: {response}",
            reward=0.0,
            done=False,
            info={
                "action_type": "toolcall",
                "api_name": api_name,
                "params": params,
                "success": True,
            },
        )

    def _do_finish(self, response: str, scenario: Dict, metadata: Dict) -> StepResult:
        """Finish the episode and compute reward based on API call accuracy."""
        gt_calls = scenario.get("ground_truth_calls", [])

        if not gt_calls:
            reward = 0.5  # No ground truth to compare
        else:
            # Evaluate each ground truth call against history
            total_score = 0.0
            for gt_call in gt_calls:
                gt_api = gt_call["api"]
                gt_params = gt_call.get("params", {})

                # Find best matching call in history
                best_score = 0.0
                for hist_call in self._call_history:
                    if hist_call["api"] == gt_api:
                        param_score = _compute_param_similarity(hist_call["params"], gt_params)
                        call_score = (
                            0.5 + 0.5 * param_score
                        )  # 0.5 for correct API + up to 0.5 for params
                        best_score = max(best_score, call_score)
                    elif hist_call["api"].lower() == gt_api.lower():
                        # Case-insensitive match
                        param_score = _compute_param_similarity(hist_call["params"], gt_params)
                        call_score = 0.4 + 0.4 * param_score
                        best_score = max(best_score, call_score)

                total_score += best_score

            reward = total_score / len(gt_calls)

        # Minimum format reward for properly finishing
        reward = max(reward, 0.1)

        return StepResult(
            observation="Episode complete.",
            reward=reward,
            done=True,
            info={
                "action_type": "finish",
                "response": response,
                "call_history": self._call_history,
                "reward": reward,
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

            api_descs = []
            for api_name in item["available_apis"]:
                if api_name in _BUILTIN_APIS:
                    api_info = _BUILTIN_APIS[api_name]
                    params_desc = ", ".join(f"{k}: {v}" for k, v in api_info["parameters"].items())
                    api_descs.append(
                        f"- {api_name}: {api_info['description']}\n"
                        f"  Parameters: {params_desc}\n"
                        f"  Example: {api_info['example']}"
                    )

            api_descriptions = "\n".join(api_descs)
            system_prompt = APIBANK_SYSTEM_PROMPT.format(api_descriptions=api_descriptions)
            prompt_text = (
                system_prompt + "\n" + APIBANK_USER_TEMPLATE.format(user_query=item["user_query"])
            )

            prompts.append(
                TaskPrompt(
                    text=prompt_text,
                    metadata={
                        "scenario": item,
                        "ground_truth_calls": item["ground_truth_calls"],
                        "call_index": 0,
                    },
                )
            )
        self._test_idx = (self._test_idx + n) % len(self._test_data)
        return prompts
