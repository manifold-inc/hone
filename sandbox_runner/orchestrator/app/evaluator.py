from __future__ import annotations
import json
import os
from typing import Dict, Any


def evaluate(dataset_dir: str, out_dir: str) -> Dict[str, Any]:
    gt_path = os.path.join(dataset_dir, "ground_truth.json")
    pred_path = os.path.join(out_dir, "predictions.json")

    if not os.path.exists(gt_path):
        raise FileNotFoundError("Missing ground_truth.json in /data")
    if not os.path.exists(pred_path):
        raise FileNotFoundError("Missing predictions.json in /out")


    with open(gt_path, "r") as f:
        gt = json.load(f)
    with open(pred_path, "r") as f:
        pred = json.load(f)


    total = 0
    correct = 0
    details = {}


    for k, v in gt.items():
        total += 1
        pv = pred.get(k)
        ok = pv == v
        correct += int(ok)
        details[k] = {"expected": v, "pred": pv, "correct": ok}

    accuracy = correct / total if total else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "details_sample": dict(list(details.items())[:20]),
        }