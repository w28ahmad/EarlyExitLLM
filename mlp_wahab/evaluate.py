"""
Offline evaluation for trained MLP early-exit predictors.

Loads all 32 trained models and evaluates them on validation data,
producing per-layer metrics. This lets us compare approaches before
plugging into SpecEE for end-to-end inference.

Metrics per layer:
  - Accuracy, Precision, Recall, F1
  - Theoretical exit rate at thresholds 0.3, 0.5, 0.7
    (exit rate = fraction of samples where the predictor says "exit")

Usage:
    python mlp_wahab/evaluate.py --model-dir ./mlp_wahab/trained_models/approach1 \\
        --approach 1 --data-root ./01_naive_all_layers

    python mlp_wahab/evaluate.py --model-dir ./mlp_wahab/trained_models/approach3 \\
        --approach 3 --data-root ./01_naive_all_layers
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlp_wahab.data import prepare_approach1, prepare_approach3

# MLP class must be defined here for torch.load() pickle resolution
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


NUM_LAYERS = 32
THRESHOLDS = [0.3, 0.5, 0.7]
# Layers in SpecEE's default dynamic schedule — these are the ones that matter most
SCHEDULE_LAYERS = [15, 16, 17, 19, 21, 24, 26, 28]


def evaluate_layer(model, X_val, y_val):
    """
    Evaluate a single layer's predictor on validation data.

    Returns a dict with metrics at each threshold.
    """
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_val, dtype=torch.float32)
        scores = model(X_t).squeeze(-1).numpy()

    results = {}
    labels = y_val

    for thresh in THRESHOLDS:
        preds = (scores > thresh).astype(float)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(labels)
        exit_rate = preds.sum() / len(preds)  # fraction predicted as "can exit"

        results[thresh] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exit_rate": exit_rate,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate MLP early-exit predictors")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory with model{0-31}.pth files")
    parser.add_argument("--approach", type=int, required=True, choices=[1, 3])
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to 01_naive_all_layers/")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Save results to CSV (default: <model-dir>/eval_results.csv)")
    args = parser.parse_args()

    output_csv = args.output_csv or os.path.join(args.model_dir, "eval_results.csv")

    print(f"Evaluating approach {args.approach} from {args.model_dir}")
    print(f"Thresholds: {THRESHOLDS}")
    print()

    all_rows = []

    for layer_idx in range(NUM_LAYERS):
        model_path = os.path.join(args.model_dir, f"model{layer_idx}.pth")
        if not os.path.exists(model_path):
            print(f"  Layer {layer_idx}: model not found, skipping")
            continue

        model = torch.load(model_path, map_location="cpu", weights_only=False)

        # Load validation split (same seed as training ensures identical split)
        if args.approach == 1:
            _, X_val, _, y_val = prepare_approach1(layer_idx, args.data_root)
        else:
            _, X_val, _, y_val = prepare_approach3(layer_idx, args.data_root)

        results = evaluate_layer(model, X_val, y_val)

        for thresh, m in results.items():
            row = {"layer": layer_idx, "threshold": thresh, **m}
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    print(f"Full results saved to: {output_csv}\n")

    # Print summary table at threshold=0.5
    print("=" * 85)
    print(f"{'Layer':>5} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'ExitRate':>9} {'Schedule':>9}")
    print("-" * 85)
    t05 = df[df["threshold"] == 0.5]
    for _, row in t05.iterrows():
        layer = int(row["layer"])
        marker = "  <--" if layer in SCHEDULE_LAYERS else ""
        print(
            f"{layer:5d} {row['accuracy']:8.4f} {row['precision']:8.4f} "
            f"{row['recall']:8.4f} {row['f1']:8.4f} {row['exit_rate']:9.4f}{marker}"
        )

    # Summary for schedule layers only
    sched = t05[t05["layer"].isin(SCHEDULE_LAYERS)]
    if not sched.empty:
        print()
        print(f"Schedule layers [{','.join(map(str, SCHEDULE_LAYERS))}] averages (threshold=0.5):")
        for col in ["accuracy", "precision", "recall", "f1", "exit_rate"]:
            print(f"  {col:>12}: {sched[col].mean():.4f}")

    # Threshold sensitivity for schedule layers
    print()
    print("Threshold sensitivity (schedule layers only):")
    print(f"{'Threshold':>10} {'Avg F1':>8} {'Avg ExitRate':>12}")
    print("-" * 35)
    for thresh in THRESHOLDS:
        subset = df[(df["threshold"] == thresh) & (df["layer"].isin(SCHEDULE_LAYERS))]
        if not subset.empty:
            print(f"{thresh:10.1f} {subset['f1'].mean():8.4f} {subset['exit_rate'].mean():12.4f}")


if __name__ == "__main__":
    main()
