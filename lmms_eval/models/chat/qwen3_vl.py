import time
from typing import List

import torch
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL as Qwen3_VLSimple
from lmms_eval.protocol import ChatMessages

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen3_vl_chat")
class Qwen3_VL(Qwen3_VLSimple):
    is_simple = False

    @staticmethod
    def _expand_video_grid_thw(inputs):
        """Expand video_grid_thw from per-video [T, H, W] to per-frame [1, H, W] entries.

        The Qwen3-VL processor creates per-frame <|vision_start|>...<|vision_end|>
        blocks separated by timestamp text tokens. This causes mm_token_type_ids to
        have one type-2 group per frame. However, video_grid_thw has only one entry
        per video with shape [T, H, W]. The model's get_rope_index expects one
        grid_thw entry per type-2 group, so we expand each [T, H, W] into T entries
        of [1, H, W].
        """
        if "video_grid_thw" not in inputs or inputs["video_grid_thw"] is None:
            return
        video_grid_thw = inputs["video_grid_thw"]
        expanded = []
        for grid in video_grid_thw:
            t, h, w = grid.tolist()
            for _ in range(t):
                expanded.append([1, h, w])
        inputs["video_grid_thw"] = torch.tensor(expanded, dtype=video_grid_thw.dtype, device=video_grid_thw.device)

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        res = []

        # A dummy collate here to sort by doc id
        def _collate(x):
            return x[0], x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        total_elapsed_time = 0
        total_tokens = 0
        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            chat_messages = [doc_to_messages[idx](self.task_dict[task][split][ids]) for idx, (ids, task, split) in enumerate(zip(doc_id, task, split))]
            chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]
            visuals = []
            videos = []
            for messages in chat_messages:
                visual, video, _ = messages.extract_media()
                visuals.append(visual)
                videos.append(video)
            visuals = self.flatten(visuals)
            videos = self.flatten(videos)
            gen_kwargs = all_gen_kwargs[0]

            # Apply chat template
            video_kwargs = {
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
            }
            if self.fps is not None:
                video_kwargs["fps"] = self.fps
            # Use max_frames instead of nframes to avoid errors on short videos
            # where total frames < max_num_frames. _subsample_video_inputs will
            # further cap to max_num_frames after loading.
            video_kwargs["max_frames"] = self.max_num_frames
            batched_messages = [chat_message.to_hf_messages(video_kwargs=video_kwargs) for chat_message in chat_messages]
            texts = self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs_qwen = process_vision_info(
                batched_messages,
                return_video_kwargs=True,
                image_patch_size=16,
                return_video_metadata=True,
            )
            video_kwargs = {**video_kwargs, **video_kwargs_qwen}

            video_metadatas = None
            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs, video_metadatas = (
                    list(video_inputs),
                    list(video_metadatas),
                )
                self._subsample_video_inputs(video_inputs, video_metadatas)

            if self.batch_size > 1:
                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    video_metadata=video_metadatas,
                    **video_kwargs,
                    do_resize=False,
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                )
            else:
                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    video_metadata=video_metadatas,
                    **video_kwargs,
                    do_resize=False,
                    return_tensors="pt",
                )

            # Expand video_grid_thw from per-video to per-frame entries to match
            # the per-frame mm_token_type_ids groups created by the processor.
            self._expand_video_grid_thw(inputs)

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
                current_gen_kwargs["top_k"] = None

            start_time = time.time()
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                top_k=current_gen_kwargs.get("top_k", None),
                use_cache=self.use_cache,
            )
            end_time = time.time()

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            # Calculate timing metrics for batch
            total_elapsed_time += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            for i, (ans, context) in enumerate(zip(answers, texts)):
                res.append(GenerationResult(text=ans, token_counts=TokenCounts(output_tokens=len(generated_ids_trimmed[i]))))
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)

                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Response: {ans}")
            # reorder this group of results back to original unsorted form
            pbar.update(1)
        res = re_ords.get_original(res)

        # Calculate average speed
        avg_speed = total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0
        # Log metrics
        metric_dict = {
            "total_gen_tokens": total_tokens,
            "total_elapsed_time": total_elapsed_time,
            "avg_speed": avg_speed,
            "additional_metrics": {
                "rank": self.rank,
            },
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res
