import os
import re
from collections import defaultdict

from loguru import logger as eval_logger
from PIL import Image


def safety_filter_valid_images(dataset):
    """Remove samples whose image file is missing, empty, or unreadable."""
    original_len = len(dataset)
    dataset = dataset.filter(lambda doc: _is_valid_image(doc["image_path"]))
    eval_logger.info(f"Safety VN - Filtered {original_len - len(dataset)} bad images, {len(dataset)} samples remaining")
    return dataset


def _is_valid_image(path):
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return False
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def safety_doc_to_visual(doc):
    image_path = doc["image_path"]
    return [Image.open(image_path).convert("RGB")]


def safety_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "\nTrả lời bằng chữ cái đáp án.")

    question = doc["question"].strip()
    options = doc["options"]

    options_text = ""
    for key in sorted(options.keys()):
        options_text += f"{key}. {options[key]}\n"

    return f"{pre_prompt}{question}\n{options_text}{post_prompt}"


def _extract_answer_letter(text):
    """Extract a single answer letter (A-D) from model output."""
    text = text.strip()
    if re.match(r"^[A-Da-d]$", text):
        return text.upper()
    m = re.match(r"^\(?([A-Da-d])\)?\.?", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-Da-d])\b", text)
    if m:
        return m.group(1).upper()
    return text.strip()[:1].upper()


def safety_process_results(doc, results):
    pred = _extract_answer_letter(results[0])
    gt = doc["ground_truth"].strip().upper()
    category = doc.get("category", "unknown")
    score = 1.0 if pred == gt else 0.0

    return {
        "safety_vn_acc": {
            "score": score,
            "prediction": pred,
            "ground_truth": gt,
        },
        "safety_vn_acc_by_category": {
            "score": score,
            "category": category,
            "prediction": pred,
            "ground_truth": gt,
        },
    }


def safety_aggregate_results(results):
    total = len(results)
    avg = sum(r["score"] for r in results) / total if total > 0 else 0.0
    eval_logger.info(f"Safety VN - Overall Accuracy: {avg:.4f} ({total} samples)")
    return avg


def safety_aggregate_by_category(results):
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["score"])

    eval_logger.info("Safety VN - Per-Category Accuracy:")
    cat_accs = {}
    for cat in sorted(by_cat.keys()):
        scores = by_cat[cat]
        acc = sum(scores) / len(scores)
        cat_accs[cat] = acc
        eval_logger.info(f"  {cat}: {acc:.4f} ({len(scores)} samples)")

    overall = sum(r["score"] for r in results) / len(results) if results else 0.0
    return overall
