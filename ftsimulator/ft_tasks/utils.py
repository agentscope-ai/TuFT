from transformers import AutoTokenizer


def load_tokenizer(base_model: str):
    # Auto-fix Qwen model ids like "Qwen3-8B" -> "Qwen/Qwen3-8B"
    if isinstance(base_model, str):
        s = base_model.strip()
        if s.lower().startswith("qwen") and "/" not in s:
            base_model = f"Qwen/{s}"
            print(f"      auto-fix model id -> {base_model}")

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    return tok
