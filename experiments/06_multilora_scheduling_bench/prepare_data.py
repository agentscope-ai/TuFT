"""
Prepare benchmark prompts from HuggingFaceH4/no_robots or generate synthetic prompts.
Saves prompts to a JSON file for offline use by the benchmark scripts.

Usage:
  # Try HuggingFace dataset first, fallback to synthetic:
  python prepare_data.py

  # Force synthetic data:
  python prepare_data.py --synthetic

  # Custom settings:
  python prepare_data.py --num-adapters 5 --samples-per-adapter 50
"""

import argparse
import json
import os
import random


OUTPUT_DIR = "/mnt/nas/hanzhang.yhz/multilora_bench/data"

# ============================================================================
# Synthetic prompt templates (diverse instruction types from no_robots categories)
# ============================================================================
SYNTHETIC_TEMPLATES = {
    "brainstorm": [
        "Give me a list of {n} creative ideas for {topic}.",
        "Brainstorm {n} unique approaches to {topic}.",
        "What are some innovative ways to {topic}?",
        "Suggest {n} different strategies for {topic}.",
        "Can you come up with {n} original concepts related to {topic}?",
    ],
    "open_qa": [
        "What is {topic} and why is it important?",
        "Explain the concept of {topic} in simple terms.",
        "How does {topic} work? Give a detailed explanation.",
        "What are the main advantages and disadvantages of {topic}?",
        "Describe the history and evolution of {topic}.",
    ],
    "closed_qa": [
        "Based on what you know about {topic}, what is the most common {aspect}?",
        "In the context of {topic}, which {aspect} is considered the best?",
        "What is the primary {aspect} used in {topic}?",
        "When discussing {topic}, what role does {aspect} play?",
        "How many types of {aspect} are there in {topic}?",
    ],
    "generation": [
        "Write a short story about {topic} in approximately 200 words.",
        "Compose a professional email about {topic}.",
        "Create a detailed product description for {topic}.",
        "Write a persuasive argument about why {topic} matters.",
        "Draft a blog post introduction about {topic}.",
    ],
    "rewrite": [
        "Rewrite the following concept in a more formal tone: {topic} is really important for everyone.",  # noqa: E501
        "Take the idea of {topic} and explain it as if you were talking to a 10-year-old.",
        "Summarize the key points about {topic} in exactly three sentences.",
        "Rephrase this statement to be more concise: {topic} has many applications in various fields.",  # noqa: E501
        "Transform this casual description into academic writing: {topic} is pretty cool and useful.",  # noqa: E501
    ],
    "summarize": [
        "Provide a comprehensive summary of {topic} covering its key aspects.",
        "Give me a brief overview of {topic}, highlighting the most important points.",
        "Summarize the current state of {topic} in the industry.",
        "What are the essential things everyone should know about {topic}?",
        "Create a quick reference guide for {topic}.",
    ],
    "classify": [
        "Categorize the following into relevant groups: {topic}. Explain your reasoning.",
        "What category does {topic} fall into? Provide a detailed classification.",
        "How would you classify different aspects of {topic}?",
        "Organize the key concepts of {topic} into a logical taxonomy.",
        "Rate and classify {topic} based on its complexity and importance.",
    ],
    "coding": [
        "Write a Python function that implements a basic version of {topic}.",
        "Show me how to create a simple {topic} algorithm in Python.",
        "Implement a class that handles {topic} with proper error handling.",
        "Write a script that demonstrates the concept of {topic}.",
        "Create a utility function for {topic} with documentation.",
    ],
}

TOPICS = [
    "machine learning",
    "web development",
    "data visualization",
    "cloud computing",
    "natural language processing",
    "computer vision",
    "distributed systems",
    "database optimization",
    "API design",
    "microservices architecture",
    "containerization",
    "CI/CD pipelines",
    "test-driven development",
    "code refactoring",
    "performance tuning",
    "security best practices",
    "mobile app development",
    "DevOps practices",
    "version control systems",
    "agile methodology",
    "design patterns",
    "functional programming",
    "object-oriented design",
    "system monitoring",
    "load balancing",
    "caching strategies",
    "message queuing",
    "search engine optimization",
    "user experience design",
    "accessibility",
    "renewable energy technology",
    "sustainable agriculture",
    "space exploration",
    "quantum computing",
    "robotics",
    "autonomous vehicles",
    "blockchain technology",
    "edge computing",
    "augmented reality",
    "virtual reality",
    "3D printing",
    "biotechnology advances",
    "climate modeling",
    "ocean conservation",
    "urban planning",
    "public transportation",
    "mental health awareness",
    "nutrition science",
    "exercise physiology",
    "remote work culture",
    "creative writing techniques",
    "music composition",
]

ASPECTS = [
    "approach",
    "method",
    "technique",
    "framework",
    "tool",
    "metric",
    "algorithm",
    "paradigm",
    "principle",
    "pattern",
]


def generate_synthetic_prompts(
    num_adapters: int,
    samples_per_adapter: int,
    seed: int = 42,
) -> dict[int, list[str]]:
    """Generate diverse synthetic prompts mimicking no_robots style."""
    random.seed(seed)
    categories = list(SYNTHETIC_TEMPLATES.keys())

    prompts_by_adapter = {}
    for adapter_idx in range(num_adapters):
        prompts = []
        for _ in range(samples_per_adapter):
            category = random.choice(categories)
            template = random.choice(SYNTHETIC_TEMPLATES[category])
            topic = random.choice(TOPICS)
            aspect = random.choice(ASPECTS)
            n = random.randint(3, 10)
            prompt = template.format(topic=topic, aspect=aspect, n=n)
            prompts.append(prompt)
        prompts_by_adapter[adapter_idx] = prompts

    return prompts_by_adapter


def load_hf_prompts(
    num_adapters: int,
    samples_per_adapter: int,
) -> dict[int, list[str]]:
    """Try loading prompts from HuggingFaceH4/no_robots."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/no_robots", split="train")
    total_needed = num_adapters * samples_per_adapter

    if len(ds) < total_needed:
        raise ValueError(f"Need {total_needed}, got {len(ds)}")

    prompts_by_adapter = {}
    for adapter_idx in range(num_adapters):
        start = adapter_idx * samples_per_adapter
        end = start + samples_per_adapter
        adapter_prompts = []
        for i in range(start, end):
            messages = ds[i]["messages"]
            user_msg = next(
                (m["content"] for m in messages if m["role"] == "user"),
                messages[0]["content"],
            )
            adapter_prompts.append(user_msg)
        prompts_by_adapter[adapter_idx] = adapter_prompts

    return prompts_by_adapter


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark prompts")
    parser.add_argument("--synthetic", action="store_true", help="Force use of synthetic prompts")
    parser.add_argument("--num-adapters", type=int, default=5)
    parser.add_argument("--samples-per-adapter", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prompts_by_adapter = None
    source = "unknown"

    # Try HuggingFace first (unless --synthetic)
    if not args.synthetic:
        try:
            print("Attempting to load HuggingFaceH4/no_robots...")
            prompts_by_adapter = load_hf_prompts(args.num_adapters, args.samples_per_adapter)
            source = "HuggingFaceH4/no_robots"
            print("  Success! Using HuggingFace dataset.")
        except Exception as e:
            print(f"  Failed: {e}")
            print("  Falling back to synthetic prompts...")

    # Fallback to synthetic
    if prompts_by_adapter is None:
        prompts_by_adapter = generate_synthetic_prompts(
            args.num_adapters, args.samples_per_adapter, args.seed
        )
        source = "synthetic"
        print(f"Using synthetic prompts (seed={args.seed})")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "prompts.json")
    output = {
        "source": source,
        "num_adapters": args.num_adapters,
        "samples_per_adapter": args.samples_per_adapter,
        "prompts_by_adapter": {str(k): v for k, v in prompts_by_adapter.items()},
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    total = sum(len(v) for v in prompts_by_adapter.values())
    print(f"\nSaved {total} prompts to {output_path}")
    print(f"  Source: {source}")
    print(f"  Adapters: {args.num_adapters}")
    print(f"  Per adapter: {args.samples_per_adapter}")

    # Preview
    for idx in range(min(2, args.num_adapters)):
        print(f'\n  Adapter {idx} sample: "{prompts_by_adapter[idx][0][:80]}..."')


if __name__ == "__main__":
    main()
