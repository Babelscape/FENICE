import argparse
import json
import os
from typing import List

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from metric.FENICE import FENICE

AGGREFACT_DATASETS = [
    "XSumFaith",
    "Polytope",
    "FactCC",
    "SummEval",
    "FRANK",
    "Wang20",
    "CLIFF",
    "Goyal21",
    "Cao22",
]


def get_threshold(
    dataset: str, true_labels: List[int], pred_scores: List[float]
) -> float:
    threshold_results = []
    for i in range(-999, 1001):
        threshold = i / 1000
        pred_labels = [1 if p > threshold else 0 for p in pred_scores]
        threshold_results.append(
            (threshold, balanced_accuracy_score(true_labels, pred_labels))
        )

    best_threshold, best_result = max(threshold_results, key=lambda x: x[1])
    print(f"best threshold for {dataset} is {best_threshold} with score {best_result}")
    return best_threshold


def choose_best_threshold(dataset, labels, scores, novelty):
    best_f1 = 0.0
    best_thresh = 0.0
    thresholds = [np.percentile(scores, p) for p in np.arange(0, 100, 0.2)]
    for thresh in thresholds:
        preds = [1 if score > thresh else 0 for score in scores]
        f1_score = balanced_accuracy_score(labels, preds)

        if f1_score >= best_f1:
            best_f1 = f1_score
            best_thresh = thresh
    print(
        f"best threshold for {dataset} {novelty} is {best_thresh} with score {best_f1}"
    )
    return best_thresh


def evaluate(data, threshold, origin, dataset_name, split, novelty):
    true_labels = [int(item["label"]) for item in data]
    pred_labels = [1 if item["score"] > threshold else 0 for item in data]
    balanced_accuracy = balanced_accuracy_score(true_labels, pred_labels)
    print(
        f"{origin}\t{dataset_name}\t{novelty}\t{split} BALANCED ACCURACY: {balanced_accuracy}"
    )
    return balanced_accuracy


def multi_threshold_evaluation(origin_names):
    results = []
    for origin in origin_names:
        for dataset in AGGREFACT_DATASETS:
            for novelty in systems_mapping:
                val_dataset_items = [
                    item
                    for item in val_data
                    if item["origin"] == origin
                    and item["dataset"] == dataset
                    and item["model_name"] in systems_mapping[novelty]
                ]
                test_dataset_items = [
                    item
                    for item in test_data
                    if item["origin"] == origin
                    and item["dataset"] == dataset
                    and item["model_name"] in systems_mapping[novelty]
                ]
                if val_dataset_items and test_dataset_items:
                    val_true_labels = [int(item["label"]) for item in val_dataset_items]
                    val_pred_scores = [item["score"] for item in val_dataset_items]
                    threshold = choose_best_threshold(
                        dataset, val_true_labels, val_pred_scores, novelty
                    )
                    # calculating test scores using validation-tuned threshold
                    acc = evaluate(
                        threshold=threshold,
                        data=test_dataset_items,
                        split=split,
                        dataset_name=dataset,
                        origin=origin,
                        novelty=novelty,
                    )
                    results.append(
                        {
                            "dataset_name": dataset,
                            "origin": origin,
                            "count": len(test_dataset_items),
                            "cat": novelty,
                            "accuracy": acc,
                        }
                    )
    # weighted average
    for origin in origin_names:
        for cat in systems_mapping:
            res = [r for r in results if r["origin"] == origin and r["cat"] == cat]
            if res:
                weighted_acc = sum([r["count"] * r["accuracy"] for r in res]) / sum(
                    [r["count"] for r in res]
                )
                print(origin, cat, weighted_acc)


def single_threshold_evaluation_sota(val_data, test_data, origin_names):
    val_data = [item for item in val_data if item["model_name"] in inv_sys_mapping]
    test_data = [item for item in test_data if item["model_name"] in inv_sys_mapping]
    ft_sota_val = [
        item for item in val_data if inv_sys_mapping[item["model_name"]] == "SOTA"
    ]
    ft_sota_test = [
        item for item in test_data if inv_sys_mapping[item["model_name"]] == "SOTA"
    ]
    for origin in origin_names:
        val_dataset_items = [item for item in ft_sota_val if item["origin"] == origin]
        test_dataset_items = [item for item in ft_sota_test if item["origin"] == origin]
        if not val_dataset_items:
            continue
        if not test_dataset_items:
            continue
        val_true_labels = [int(item["label"]) for item in val_dataset_items]
        val_pred_scores = [item["score"] for item in val_dataset_items]
        threshold = choose_best_threshold(
            origin, val_true_labels, val_pred_scores, novelty="SOTA"
        )
        evaluate(
            data=test_dataset_items,
            split=split,
            dataset_name=origin,
            threshold=threshold,
            origin=origin,
            novelty="SOTA",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggrefact Scorer")
    parser.add_argument(
        "--input_file",
        type=str,
        help="Input JSONL file path",
        default="data/%split_aggre_fact_final.jsonl",
    )
    parser.add_argument(
        "--predictions_out_dir", type=str, help="out dir path", default="outputs"
    )
    parser.add_argument(
        "--cache_dir", type=str, help="Input JSONL file path", default="data/"
    )
    args = parser.parse_args()
    fenice = FENICE(
        nli_batch_size=128,
        claim_extractor_batch_size=16,
        nli_max_length=512,
        sliding_stride=5,
        use_coref=True,
    )

    # AggreFact systems mapping
    systems_mapping = {
        "SOTA": ["BART", "PegasusDynamic", "T5", "Pegasus"],
        "XFORMER": ["BertSum", "BertExtAbs", "BertExt", "GPT2", "BERTS2S", "TranS2S"],
        "OLD": [
            "FastAbsRl",
            "TConvS2S",
            "PtGen",
            "PtGenCoverage",
            "Summa",
            "BottomUp",
            "Seq2Seq",
            "TextRank",
            "missing",
            "ImproveAbs",
            "NEUSUM",
            "ClosedBookDecoder",
            "RNES",
            "BanditSum",
            "ROUGESal",
            "MultiTask",
            "UnifiedExtAbs",
        ],
    }
    inv_sys_mapping = {}
    for cat, sys in systems_mapping.items():
        for s in sys:
            inv_sys_mapping[s] = cat
    datasets = []
    for split in ["val", "test"]:
        file = args.input_file.replace("%split", split)
        with open(file) as f:
            data = [json.loads(l.strip()) for l in f.readlines()]
        input = []
        for item in data:
            doc, summary = item["doc"], item["summary"]
            input.append({"document": doc, "summary": summary})
        predictions = fenice.score_batch(input)
        for item, pred in zip(data, predictions):
            item["score"] = pred["score"]
            item["alignments"] = pred["alignments"]
        datasets.append(data)
        predictions_path = os.path.join(
            args.predictions_out_dir,
            "%split_aggre_fact_final_predictions.jsonl".replace("%split", split),
        )
        os.makedirs(args.predictions_out_dir, exist_ok=True)
        with open(predictions_path, "w") as w:
            for item in data:
                w.write(json.dumps(item) + "\n")
    val_data, test_data = datasets
    origin_names = ["cnndm", "xsum"]
    multi_threshold_evaluation(origin_names)
    single_threshold_evaluation_sota(val_data, test_data, origin_names)
