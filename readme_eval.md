# Evaluating Qwen3-VL and Qwen3.5-VL Models

This guide covers how to evaluate three Qwen vision-language models with lmms-eval:

| Model | Type | Params | HuggingFace ID | lmms-eval model name |
|-------|------|--------|----------------|----------------------|
| Qwen3-VL-32B-Thinking | Dense, reasoning | 32B | `Qwen/Qwen3-VL-32B-Thinking` | `qwen3_vl` |
| Qwen3-VL-32B-Instruct | Dense, instruct | 32B | `Qwen/Qwen3-VL-32B-Instruct` | `qwen3_vl` |
| Qwen3.5-VL-397B-A17B | MoE, instruct | 397B (17B active) | `Qwen/Qwen3.5-VL-397B-A17B-Instruct` | `qwen3_vl` |

All three use the same `qwen3_vl` model class. The MoE variant is auto-detected by the `A\d+B` pattern in the model name and loads with `Qwen3VLMoeForConditionalGeneration`.

## Prerequisites

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval && uv sync
```

## GPU Requirements

| Model | BF16 VRAM (approx) | Recommended Setup |
|-------|---------------------|-------------------|
| Qwen3-VL-32B-Thinking | ~65 GB | 1x H200 / 2x A100-80G |
| Qwen3-VL-32B-Instruct | ~65 GB | 1x H200 / 2x A100-80G |
| Qwen3.5-VL-397B-A17B | ~800 GB | 8x H200 / 8x A100-80G |

All commands below use `device_map=auto` to automatically shard across available GPUs.

## Quick Smoke Tests (8 samples)

Run these first to verify each model loads and produces output.

### Qwen3-VL-32B-Thinking

```bash
python -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-32B-Thinking,device_map=auto,attn_implementation=sdpa \
  --tasks mme \
  --batch_size 1 \
  --limit 8
```

### Qwen3-VL-32B-Instruct

```bash
python -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-32B-Instruct,device_map=auto,attn_implementation=sdpa \
  --tasks mme \
  --batch_size 1 \
  --limit 8
```

### Qwen3.5-VL-397B-A17B

```bash
python -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3.5-VL-397B-A17B-Instruct,device_map=auto,attn_implementation=sdpa \
  --tasks mme \
  --batch_size 1 \
  --limit 8
```

## Full Benchmark Runs

### Benchmarks

| Benchmark | Task name | Split | Metrics | Notes |
|-----------|-----------|-------|---------|-------|
| MME | `mme` | test | perception + cognition scores | Short-answer VQA |
| MMMU | `mmmu_val` | validation | accuracy | Multi-discipline multiple-choice |
| MMMU (reasoning) | `mmmu_val_reasoning` | validation | llm_as_judge_eval | Long-form CoT; uses LLM-as-judge |
| MathVista | `mathvista_testmini` | testmini | accuracy | Math reasoning with visuals |

### Qwen3-VL-32B-Thinking

This is a reasoning model that produces `<think>...</think>` traces before answering. The framework's `parse_reasoning_model_answer` extracts the final answer from `<answer>...</answer>` or `\boxed{}` tags. Use tasks with higher `max_new_tokens` to avoid truncating the reasoning chain.

```bash
# MME
python -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-32B-Thinking,device_map=auto,attn_implementation=sdpa \
  --tasks mme \
  --batch_size 1 \
  --output_path ./results/qwen3_vl_32b_thinking

# MMMU (reasoning variant - recommended for thinking models)
python -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-32B-Thinking,device_map=auto,attn_implementation=sdpa \
  --tasks mmmu_val_reasoning \
  --batch_size 1 \
  --output_path ./results/qwen3_vl_32b_thinking

# MathVista
python -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-32B-Thinking,device_map=auto,attn_implementation=sdpa \
  --tasks mathvista_testmini \
  --batch_size 1 \
  --output_path ./results/qwen3_vl_32b_thinking
```

### Qwen3-VL-32B-Instruct

Standard instruct model. Works with all task variants.

```bash
# MME
python -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-32B-Instruct,device_map=auto,attn_implementation=sdpa \
  --tasks mme \
  --batch_size 1 \
  --output_path ./results/qwen3_vl_32b_instruct

# MMMU
python -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-32B-Instruct,device_map=auto,attn_implementation=sdpa \
  --tasks mmmu_val \
  --batch_size 1 \
  --output_path ./results/qwen3_vl_32b_instruct

# MathVista
python -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-32B-Instruct,device_map=auto,attn_implementation=sdpa \
  --tasks mathvista_testmini \
  --batch_size 1 \
  --output_path ./results/qwen3_vl_32b_instruct
```

### Qwen3.5-VL-397B-A17B

MoE model. The `A17B` in the name triggers automatic MoE class selection (`Qwen3VLMoeForConditionalGeneration`). Requires multi-node or a large GPU cluster.

```bash
# MME
python -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3.5-VL-397B-A17B-Instruct,device_map=auto,attn_implementation=sdpa \
  --tasks mme \
  --batch_size 1 \
  --output_path ./results/qwen3_5_vl_397b_a17b

# MMMU
python -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3.5-VL-397B-A17B-Instruct,device_map=auto,attn_implementation=sdpa \
  --tasks mmmu_val \
  --batch_size 1 \
  --output_path ./results/qwen3_5_vl_397b_a17b

# MathVista
python -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3.5-VL-397B-A17B-Instruct,device_map=auto,attn_implementation=sdpa \
  --tasks mathvista_testmini \
  --batch_size 1 \
  --output_path ./results/qwen3_5_vl_397b_a17b
```

## Run All Three on the Same Benchmark

To compare all three models on the same task:

```bash
for model in \
  Qwen/Qwen3-VL-32B-Thinking \
  Qwen/Qwen3-VL-32B-Instruct \
  Qwen/Qwen3.5-VL-397B-A17B-Instruct; do

  slug=$(echo "$model" | tr '/' '_')
  python -m lmms_eval \
    --model qwen3_vl \
    --model_args pretrained=$model,device_map=auto,attn_implementation=sdpa \
    --tasks mme,mmmu_val,mathvista_testmini \
    --batch_size 1 \
    --output_path ./results/$slug
done
```

Results are saved as JSON under `./results/<model_slug>/`.

## Useful Model Args

| Arg | Default | Description |
|-----|---------|-------------|
| `pretrained` | `Qwen/Qwen3-VL-4B-Instruct` | HuggingFace model ID or local path |
| `device_map` | `auto` | How to distribute across GPUs |
| `attn_implementation` | `None` | `sdpa`, `flash_attention_2`, or `eager` |
| `max_pixels` | `1605632` | Max image resolution (higher = more detail, more VRAM) |
| `min_pixels` | `200704` | Min image resolution |
| `max_num_frames` | `32` | Max video frames to sample |
| `system_prompt` | `"You are a helpful assistant."` | System message prepended to prompts |
| `use_cache` | `True` | KV cache during generation |

## Notes

- **Thinking model answer parsing**: The `qwen3_vl` model class calls `parse_reasoning_model_answer()` on all outputs. This extracts content from `<answer>...</answer>` or `\boxed{}` tags, and passes through raw text otherwise. This means it works correctly for both thinking and instruct models.
- **MoE auto-detection**: The model class checks `re.search(r"A\d+B", pretrained)` on the model name. If matched, it uses `Qwen3VLMoeForConditionalGeneration`; otherwise `Qwen3VLForConditionalGeneration`.
- **Task-specific token limits**: MME uses `max_new_tokens=16` (short answers). MMMU reasoning uses `max_new_tokens=16384` (long CoT). The thinking model benefits from tasks with higher token limits.
- **Multi-GPU parallelism**: For data-parallel evaluation across multiple GPUs, use `accelerate launch` instead of `python -m`:
  ```bash
  accelerate launch --num_processes 8 -m lmms_eval \
    --model qwen3_vl \
    --model_args pretrained=Qwen/Qwen3-VL-32B-Instruct,device_map=auto \
    --tasks mme \
    --batch_size 1
  ```
