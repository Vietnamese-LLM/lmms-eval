from collections import defaultdict

from loguru import logger as eval_logger

# =============================================================================
# Pill subset — classify pill labels from image
# =============================================================================


def vaipe_pill_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def vaipe_pill_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return "This is an image of pills. " "Identify all the pill class labels visible in the image. " "Output only the class labels as a comma-separated list of integers in ascending order."


def vaipe_pill_doc_to_target(doc):
    labels = sorted(doc["labels"])
    return ",".join(str(l) for l in labels)


def vaipe_pill_process_results(doc, results):
    pred = results[0].strip()
    gt_labels = sorted(doc["labels"])
    gt_set = set(gt_labels)

    # Parse predicted labels
    try:
        pred_labels = sorted([int(x.strip()) for x in pred.split(",") if x.strip().lstrip("-").isdigit()])
        pred_set = set(pred_labels)
    except Exception:
        pred_set = set()

    # F1 over label sets
    if not gt_set and not pred_set:
        score = 1.0
    elif not gt_set or not pred_set:
        score = 0.0
    else:
        tp = len(gt_set & pred_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(gt_set) if gt_set else 0.0
        score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "vaipe_pill_acc": {
            "score": score,
            "prediction": pred,
            "ground_truth": ",".join(str(l) for l in gt_labels),
        },
    }


def vaipe_pill_aggregate_results(results):
    total = len(results)
    avg = sum(r["score"] for r in results) / total if total > 0 else 0.0
    eval_logger.info(f"VAIPE Pill - Accuracy: {avg:.4f} ({total} samples)")
    return avg


# =============================================================================
# Prescription subset — extract text annotations from prescription image
# =============================================================================


def vaipe_prescription_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def vaipe_prescription_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return "This is an image of a medical prescription. " "Extract all text annotations from this prescription. " "Output each text entry on a separate line."


def vaipe_prescription_doc_to_target(doc):
    return "\n".join(doc["texts"])


def vaipe_prescription_process_results(doc, results):
    pred = results[0].strip()
    gt_texts = set(t.strip().lower() for t in doc["texts"] if t.strip())
    pred_texts = set(t.strip().lower() for t in pred.split("\n") if t.strip())

    if not gt_texts and not pred_texts:
        score = 1.0
    elif not gt_texts or not pred_texts:
        score = 0.0
    else:
        intersection = gt_texts & pred_texts
        precision = len(intersection) / len(pred_texts)
        recall = len(intersection) / len(gt_texts)
        score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "vaipe_prescription_f1": {
            "score": score,
            "prediction": pred,
            "ground_truth": "\n".join(doc["texts"]),
        },
    }


def vaipe_prescription_aggregate_results(results):
    total = len(results)
    avg = sum(r["score"] for r in results) / total if total > 0 else 0.0
    eval_logger.info(f"VAIPE Prescription - F1: {avg:.4f} ({total} samples)")
    return avg


# =============================================================================
# Pill-Prescription Map subset — predict prescription ID from pill ID (text-only)
# =============================================================================


def vaipe_pill_pres_map_doc_to_visual(doc):
    return []


def vaipe_pill_pres_map_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pill = doc["pill"]
    return f"Given the pill identifier '{pill}', " f"what is the corresponding prescription identifier? " f"Output only the prescription identifier."


def vaipe_pill_pres_map_process_results(doc, results):
    pred = results[0].strip()
    gt = doc["prescription"].strip()
    score = 1.0 if pred == gt else 0.0

    return {
        "vaipe_pill_pres_map_acc": {
            "score": score,
            "prediction": pred,
            "ground_truth": gt,
        },
    }


def vaipe_pill_pres_map_aggregate_results(results):
    total = len(results)
    correct = sum(r["score"] for r in results)
    acc = correct / total if total > 0 else 0.0
    eval_logger.info(f"VAIPE Pill-Pres Map - Accuracy: {acc:.4f} ({total} samples)")
    return acc
