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

## Vietnamese VLM Benchmark Suite

### Benchmark Availability

| Category | Benchmark | Task Name | Supported |
|----------|-----------|-----------|-----------|
| Reading Medical Reports | MMLongBench-Doc | `mmlongbench_doc` | Yes |
| Reading Medical Reports | DocVQA | `docvqa_val` / `docvqa_test` | Yes |
| Reading Medical Reports | InfoVQA | `infovqa_val` / `infovqa_test` | Yes |
| Reading Medical Reports | OCRBench_v2 | `ocrbench_v2` | Yes |
| Online Images | RealWorldQA | `realworldqa` | Yes |
| Online Images | MMStar | `mmstar` | Yes |
| Online Images | VMMU | — | **No** |
| Online Images | VAIPE-Pill | — | **No** |
| Online Images | VAIPE-P | — | **No** |
| Videos | Video-MME | `videomme` | Yes |
| Videos | MLVU | `mlvu_dev` / `mlvu_test` | Yes |
| Videos | MVBench | `mvbench` (group, 20 subtasks) | Yes |
| Videos | VideoMMMU | `video_mmmu` (group) | Yes |
| Searching | BLINK | `blink` (group, 14 subtasks) | Yes |
| Searching | Mobile Actions | — | **No** |
| Searching | HRBench4K | `hrbench4k` | Yes |
| Searching | HRBench8K | `hrbench8k` | Yes |

**13 out of 17 benchmarks** can be evaluated directly. The 4 unsupported ones (VMMU, VAIPE-Pill, VAIPE-P, Mobile Actions) would require custom task implementations.

### Run All Supported Benchmarks

Replace `MODEL_ID` and `OUTPUT_DIR` as needed. Example uses `Qwen/Qwen3-VL-32B-Instruct`.

```bash
MODEL_ID="Qwen/Qwen3-VL-32B-Instruct"
OUTPUT_DIR="./results/qwen3_vl_32b_instruct"
MODEL_ARGS="pretrained=${MODEL_ID},device_map=auto,attn_implementation=sdpa"

# --- Reading Medical Reports ---

# MMLongBench-Doc
python -m lmms_eval \
  --model qwen3_vl \
  --model_args $MODEL_ARGS \
  --tasks mmlongbench_doc \
  --batch_size 1 \
  --output_path $OUTPUT_DIR

# DocVQA (validation)
python -m lmms_eval \
  --model qwen3_vl \
  --model_args $MODEL_ARGS \
  --tasks docvqa_val \
  --batch_size 1 \
  --output_path $OUTPUT_DIR

# InfoVQA (validation)
python -m lmms_eval \
  --model qwen3_vl \
  --model_args $MODEL_ARGS \
  --tasks infovqa_val \
  --batch_size 1 \
  --output_path $OUTPUT_DIR

# OCRBench v2
python -m lmms_eval \
  --model qwen3_vl \
  --model_args $MODEL_ARGS \
  --tasks ocrbench_v2 \
  --batch_size 1 \
  --output_path $OUTPUT_DIR

# --- Online Images ---

# RealWorldQA
python -m lmms_eval \
  --model qwen3_vl \
  --model_args $MODEL_ARGS \
  --tasks realworldqa \
  --batch_size 1 \
  --output_path $OUTPUT_DIR

# MMStar
python -m lmms_eval \
  --model qwen3_vl \
  --model_args $MODEL_ARGS \
  --tasks mmstar \
  --batch_size 1 \
  --output_path $OUTPUT_DIR

# --- Videos ---

# Video-MME
python -m lmms_eval \
  --model qwen3_vl \
  --model_args $MODEL_ARGS \
  --tasks videomme \
  --batch_size 1 \
  --output_path $OUTPUT_DIR

# MLVU (dev split)
python -m lmms_eval \
  --model qwen3_vl \
  --model_args $MODEL_ARGS \
  --tasks mlvu_dev \
  --batch_size 1 \
  --output_path $OUTPUT_DIR

# MVBench (all 20 subtasks)
python -m lmms_eval \
  --model qwen3_vl \
  --model_args $MODEL_ARGS \
  --tasks mvbench \
  --batch_size 1 \
  --output_path $OUTPUT_DIR

# Video-MMMU (all subtasks)
python -m lmms_eval \
  --model qwen3_vl \
  --model_args $MODEL_ARGS \
  --tasks video_mmmu \
  --batch_size 1 \
  --output_path $OUTPUT_DIR

# --- Searching ---

# BLINK (all 14 subtasks)
python -m lmms_eval \
  --model qwen3_vl \
  --model_args $MODEL_ARGS \
  --tasks blink \
  --batch_size 1 \
  --output_path $OUTPUT_DIR

# HRBench 4K
python -m lmms_eval \
  --model qwen3_vl \
  --model_args $MODEL_ARGS \
  --tasks hrbench4k \
  --batch_size 1 \
  --output_path $OUTPUT_DIR

# HRBench 8K
python -m lmms_eval \
  --model qwen3_vl \
  --model_args $MODEL_ARGS \
  --tasks hrbench8k \
  --batch_size 1 \
  --output_path $OUTPUT_DIR
```

### One-liner: All 13 Benchmarks at Once

```bash
python -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-32B-Instruct,device_map=auto,attn_implementation=sdpa \
  --tasks mmlongbench_doc,docvqa_val,infovqa_val,ocrbench_v2,realworldqa,mmstar,videomme,mlvu_dev,mvbench,video_mmmu,blink,hrbench4k,hrbench8k \
  --batch_size 1 \
  --output_path ./results/qwen3_vl_32b_instruct
```
