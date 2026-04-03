"""
Training script for MLP early-exit predictors.

Trains one MLP per decoder layer (32 total for LLaMA-2-7B). Each MLP predicts
whether the model can safely exit at that layer during speculative decoding.

Supports two training approaches:
  --approach 1 : Naive baseline — uses 12 features as-is from contiguous data
  --approach 3 : Augmented gaps — recomputes prob_gap with skip=1,2,3 to simulate
                 SpecEE's dynamic scheduling (3x training data per layer)

Trained models are saved as full pickle objects (torch.save(model)) for
compatibility with SpecEE's loading: torch.load(path).to(torch.float16).

The MLP class is defined at module level here so that pickle stores the
reference as __main__.MLP. At load time in EEInference.py, the same class
is imported via `from model_llama_ee import MLP`, which resolves correctly.

Usage:
    # Train approach 1 (all 32 layers):
    python mlp_wahab/train.py --approach 1 --data-root ./01_naive_all_layers \\
        --output-dir ./mlp_wahab/trained_models/approach1

    # Train approach 3 (all 32 layers):
    python mlp_wahab/train.py --approach 3 --data-root ./01_naive_all_layers \\
        --output-dir ./mlp_wahab/trained_models/approach3

    # Train a single layer for validation:
    python mlp_wahab/train.py --approach 1 --data-root ./01_naive_all_layers \\
        --output-dir ./mlp_wahab/trained_models/approach1 --layer 15
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure the mlp_wahab package is importable when run as __main__
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlp_wahab.data import prepare_approach1, prepare_approach3

# ---------------------------------------------------------------------------
# MLP class — must be defined at module level for pickle compatibility.
# This is identical to SpecEE_cloud/model_llama_ee.py lines 949-963.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Training hyperparameters (matching plan)
# ---------------------------------------------------------------------------
INPUT_SIZE = 12
HIDDEN_SIZE = 512
OUTPUT_SIZE = 1
BATCH_SIZE = 1024
LR = 1e-3
MAX_EPOCHS = 50
PATIENCE = 5          # early stopping patience (on val loss)
LR_PATIENCE = 3       # ReduceLROnPlateau patience
LR_FACTOR = 0.5
NUM_LAYERS = 32       # LLaMA-2-7B has 32 decoder layers


def compute_class_weight(y_train):
    """
    Compute a scalar weight for the positive class to handle label imbalance.

    Early layers have very few positive labels (e.g., 2% at layer 0), so we
    upweight positives to prevent the model from predicting all-negative.

    Returns a weight tensor usable with BCELoss's weight parameter via manual
    per-sample weighting.
    """
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    if n_pos == 0:
        return 1.0  # no positives — weight is irrelevant
    return float(n_neg / n_pos)


def train_one_layer(layer_idx, approach, data_root, output_dir, verbose=True):
    """
    Train a single MLP predictor for one decoder layer.

    Args:
        layer_idx: which decoder layer (0-31)
        approach: 1 (naive) or 3 (augmented gaps)
        data_root: path to 01_naive_all_layers/
        output_dir: where to save model{layer_idx}.pth
        verbose: print training progress

    Returns:
        dict with final val metrics (loss, accuracy, f1)
    """
    # --- Load data ---
    if approach == 1:
        X_train, X_val, y_train, y_val = prepare_approach1(layer_idx, data_root)
    elif approach == 3:
        X_train, X_val, y_train, y_val = prepare_approach3(layer_idx, data_root)
    else:
        raise ValueError(f"Unknown approach: {approach}")

    # Handle edge case: no positive samples at all (common for layers 0-10)
    n_pos = y_train.sum()
    if n_pos == 0:
        if verbose:
            print(f"  Layer {layer_idx:2d}: 0 positives in training set — saving untrained model")
        model = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model, os.path.join(output_dir, f"model{layer_idx}.pth"))
        return {"loss": 0.0, "accuracy": 1.0, "f1": 0.0}

    # --- Build dataloaders ---
    pos_weight = compute_class_weight(y_train)
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model, loss, optimizer ---
    model = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    # BCELoss with per-sample weighting for class imbalance:
    # positive samples get weight=pos_weight, negative samples get weight=1.0
    criterion = nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_PATIENCE, factor=LR_FACTOR
    )

    # --- Training loop with early stopping ---
    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch).squeeze(-1)
            raw_loss = criterion(y_pred, y_batch)
            # Apply class weighting: scale positive sample losses by pos_weight
            weights = torch.where(y_batch == 1.0, pos_weight, 1.0)
            loss = (raw_loss * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(y_batch)
            train_count += len(y_batch)

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch).squeeze(-1)
                raw_loss = criterion(y_pred, y_batch)
                weights = torch.where(y_batch == 1.0, pos_weight, 1.0)
                loss = (raw_loss * weights).mean()
                val_loss_sum += loss.item() * len(y_batch)
                val_count += len(y_batch)
                all_preds.append((y_pred > 0.5).float())
                all_labels.append(y_batch)

        val_loss = val_loss_sum / val_count
        scheduler.step(val_loss)

        # Compute metrics
        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (preds == labels).float().mean().item()

        if verbose and (epoch % 5 == 0 or epoch == MAX_EPOCHS - 1):
            print(
                f"    Epoch {epoch:3d}: train_loss={train_loss_sum/train_count:.4f} "
                f"val_loss={val_loss:.4f} acc={accuracy:.4f} f1={f1:.4f} "
                f"prec={precision:.4f} rec={recall:.4f}"
            )

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                if verbose:
                    print(f"    Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    # Restore best model and save
    model.load_state_dict(best_model_state)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"model{layer_idx}.pth")
    torch.save(model, save_path)

    if verbose:
        print(f"  Layer {layer_idx:2d}: val_loss={best_val_loss:.4f} acc={accuracy:.4f} f1={f1:.4f} -> {save_path}")

    return {"loss": best_val_loss, "accuracy": accuracy, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Train MLP early-exit predictors")
    parser.add_argument("--approach", type=int, required=True, choices=[1, 3],
                        help="1=naive baseline, 3=augmented gaps")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to 01_naive_all_layers/")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save model{0-31}.pth files")
    parser.add_argument("--layer", type=int, default=None,
                        help="Train a single layer (for validation). Omit to train all 32.")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Path to append progress log (default: <output-dir>/training_log.txt)")
    args = parser.parse_args()

    # Set up log file
    log_path = args.log_file or os.path.join(args.output_dir, "training_log.txt")
    os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    def log(msg):
        """Print to console and append to log file."""
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"{'='*60}")
    log(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Approach {args.approach} | data: {args.data_root} | output: {args.output_dir}")
    log(f"Architecture: MLP({INPUT_SIZE}, {HIDDEN_SIZE}, {OUTPUT_SIZE})")
    log(f"Log file: {log_path}")
    log(f"{'='*60}")
    log("")

    if args.layer is not None:
        layers = [args.layer]
    else:
        layers = list(range(NUM_LAYERS))

    results = {}
    layer_times = []
    total_start = time.time()

    for i, layer_idx in enumerate(layers):
        layer_start = time.time()
        log(f"--- Layer {layer_idx} ({i+1}/{len(layers)}) ---")

        metrics = train_one_layer(layer_idx, args.approach, args.data_root, args.output_dir)
        results[layer_idx] = metrics

        layer_elapsed = time.time() - layer_start
        layer_times.append(layer_elapsed)
        total_elapsed = time.time() - total_start

        # Compute ETA based on average time per layer
        avg_per_layer = total_elapsed / (i + 1)
        remaining_layers = len(layers) - (i + 1)
        eta_seconds = avg_per_layer * remaining_layers
        eta_time = datetime.now() + timedelta(seconds=eta_seconds)

        m = metrics
        log(
            f"  -> Done in {layer_elapsed:.1f}s | "
            f"val_loss={m['loss']:.4f} acc={m['accuracy']:.4f} f1={m['f1']:.4f} | "
            f"Elapsed: {timedelta(seconds=int(total_elapsed))} | "
            f"ETA: {eta_time.strftime('%H:%M:%S')} ({remaining_layers} layers left, ~{timedelta(seconds=int(eta_seconds))})"
        )
        log("")

    # Print summary
    total_time = time.time() - total_start
    log("=" * 70)
    log(f"Training complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (total: {timedelta(seconds=int(total_time))})")
    log(f"{'Layer':>5} {'Val Loss':>10} {'Accuracy':>10} {'F1':>10} {'Time (s)':>10}")
    log("-" * 70)
    for j, layer_idx in enumerate(sorted(results)):
        m = results[layer_idx]
        t = layer_times[j] if j < len(layer_times) else 0
        log(f"{layer_idx:5d} {m['loss']:10.4f} {m['accuracy']:10.4f} {m['f1']:10.4f} {t:10.1f}")
    log(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
