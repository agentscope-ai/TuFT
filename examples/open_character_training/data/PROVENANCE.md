# Data provenance

The files in this directory make the recipe reproducible without requiring access to the
Qwen3.7-Max API.

## `prompts.json.gz`

This compressed JSON contains two ordered lists:

- `constitution_prompts`: 499 unique sarcasm-relevant prompts derived from
  `data/few-shot/sarcasm.jsonl` in the Open Character Training repository;
- `general_prompts`: 1,030 single-turn prompts from the `GAIR/lima` training split.

The Open Character Training reference data was frozen from:

- Repository: <https://github.com/maiush/OpenCharacterTraining>
- Commit: `d1da9f03628cb4c5482ba2e494a7cba33bcd5818`
- Paper: <https://arxiv.org/abs/2511.01689>

The asset preserves list order because the teacher-cache key includes the source-local prompt
index. `generate_data.py` repeats every prompt twice and computes:

```text
sha256(canonical_json(["sarcastic", source, prompt_index, repeat]))
```

Compressed-file SHA-256: `abaa9bef052fc5fa56135fbe0f4f4c4a2c1d1618f4ae01450e1f0e9a9010d23c`.

## `qwen3.7-max-sarcastic-chosen.json.gz`

This is a compressed JSON request cache with one item for each of the 3,058 planned keys. Values
contain the cleaned response text and an `ok` flag. The cache metadata records:

- teacher: `qwen3.7-max`, generated with the official `openai` Python client against DashScope's
  OpenAI-compatible endpoint (no DashScope-specific SDK);
- maximum completion: 1,024 tokens;
- temperature: 0.7;
- top-p: 0.95;
- thinking disabled;
- SHA-256 of the exact constitution-bearing teacher system prompt.

The `usage` entry is the API-accounting snapshot associated with cache construction. It is not
the number of dataset rows: some rows were already present when that accounting interval began.
The authoritative coverage count is the number of keys in `items`.

The cache is generated supervision, not a verbatim source corpus. Users should review its content
and the model provider's terms before redistribution or production use. The recipe never sends
the cache to an external service; it is read locally and paired with live base-student samples.

The cache contains 3,058 items. Its compressed-file SHA-256 is
`311a9331bcb5dd9674130b836030e78d7b6298ddba44823d82df53908050c38b`.
