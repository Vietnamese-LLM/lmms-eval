import re
from collections import defaultdict

from loguru import logger as eval_logger


def vmmu_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def vmmu_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    language = "en"
    if lmms_eval_specific_kwargs is not None:
        language = lmms_eval_specific_kwargs.get("language", "en")

    if language == "vn":
        return doc["vn_prompt"]
    return doc["en_prompt"]


def _parse_answer(pred):
    """Extract answer letter from model output.

    Handles formats like: {A}, A, (A), the answer is A, etc.
    """
    pred = pred.strip()

    # Match {X} format (dataset's expected output format)
    match = re.search(r"\{([A-Z])\}", pred)
    if match:
        return match.group(1)

    # Match (X) format
    match = re.search(r"\(([A-Z])\)", pred)
    if match:
        return match.group(1)

    # Match "the answer is X" or "đáp án là X" patterns
    match = re.search(r"(?:the answer is|đáp án là)\s*([A-Z])", pred, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Fallback: first capital letter in response
    match = re.search(r"^([A-Z])\b", pred.strip())
    if match:
        return match.group(1)

    return pred.strip()


def vmmu_process_results(doc, results):
    pred = results[0]
    gt = doc["ground_truth"].strip().upper()

    parsed_pred = _parse_answer(pred)
    score = 1.0 if parsed_pred.upper() == gt else 0.0

    return {
        "vmmu_acc": {
            "id": doc["ID"],
            "subject": doc["subject"],
            "score": score,
        },
    }


def vmmu_aggregate_results(results):
    subject_scores = defaultdict(list)
    for result in results:
        subject_scores[result["subject"]].append(result["score"])

    subject_avg = {}
    for subject, scores in sorted(subject_scores.items()):
        avg = sum(scores) / len(scores)
        subject_avg[subject] = avg
        eval_logger.info(f"VMMU - {subject}: {avg:.4f} ({len(scores)} samples)")

    overall = sum(r["score"] for r in results) / len(results)
    eval_logger.info(f"VMMU - Overall: {overall:.4f} ({len(results)} samples)")
    return overall
