import re

from loguru import logger as eval_logger
from PIL import Image


def laion_doc_to_visual(doc):
    image_path = doc["image_path"]
    return [Image.open(image_path).convert("RGB")]


def laion_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "\nAnswer with the option letter only.")

    question = doc["question"].strip()
    options = doc["options"]

    options_text = ""
    for key in sorted(options.keys()):
        options_text += f"{key}. {options[key]}\n"

    return f"{pre_prompt}{question}\n{options_text}{post_prompt}"


def _extract_answer_letter(text):
    """Extract a single answer letter (A-D) from model output."""
    text = text.strip()
    # Direct single letter
    if re.match(r"^[A-Da-d]$", text):
        return text.upper()
    # Letter with period or parenthesis: "A." or "(A)" or "A)"
    m = re.match(r"^\(?([A-Da-d])\)?\.?", text)
    if m:
        return m.group(1).upper()
    # Search for first occurrence of a standalone letter
    m = re.search(r"\b([A-Da-d])\b", text)
    if m:
        return m.group(1).upper()
    return text.strip()[:1].upper()


def laion_process_results(doc, results):
    pred = _extract_answer_letter(results[0])
    gt = doc["ground_truth"].strip().upper()
    score = 1.0 if pred == gt else 0.0

    return {
        "laion_mcq_acc": {
            "score": score,
            "prediction": pred,
            "ground_truth": gt,
        },
    }


def laion_aggregate_results(results):
    total = len(results)
    avg = sum(r["score"] for r in results) / total if total > 0 else 0.0
    eval_logger.info(f"LAION MCQ - Accuracy: {avg:.4f} ({total} samples)")
    return avg
