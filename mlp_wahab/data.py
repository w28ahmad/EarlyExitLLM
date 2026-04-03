"""
Data loading and augmentation for MLP early-exit predictor training.

Supports two approaches:
  Approach 1 — Naive baseline: use all 12 features as-is from contiguous layer data.
  Approach 3 — Augmented gaps: recompute prob_gap to simulate SpecEE's dynamic
               scheduler which skips layers (e.g., gap between layer 19 and 17 = skip 2).

Key assumption for approach 3:
  Rows are aligned across layer CSVs within each dataset — row i in layer_5.csv
  and layer_3.csv correspond to the same input token at different decoder layers.
  This is guaranteed by generate_training_data.py's sequential processing.
  We therefore pair rows within each dataset before concatenating across datasets.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Column layout of each CSV: 12 features + 1 label
FEATURE_COLS = [
    "logit_0", "logit_1", "logit_2", "logit_3",
    "prob_0", "prob_1", "prob_2", "prob_3",
    "gap_0", "gap_1", "gap_2", "gap_3",
]
PROB_COLS = ["prob_0", "prob_1", "prob_2", "prob_3"]
GAP_COLS = ["gap_0", "gap_1", "gap_2", "gap_3"]
LABEL_COL = "label"

ALL_DATASETS = ["alpaca", "gsm8k", "commonsenseqa", "sst2"]


def load_layer_csv(data_root, dataset, layer_idx):
    """Load a single layer CSV for one dataset. Returns a DataFrame."""
    path = os.path.join(data_root, dataset, f"layer_{layer_idx}.csv")
    return pd.read_csv(path)


def load_layer_data(layer_idx, data_root, datasets=None):
    """
    Load and concatenate a single layer's CSVs from all specified datasets.

    Returns:
        X: np.ndarray of shape (N, 12) — all 12 features
        y: np.ndarray of shape (N,) — binary labels
    """
    if datasets is None:
        datasets = ALL_DATASETS

    frames = []
    for ds in datasets:
        df = load_layer_csv(data_root, ds, layer_idx)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    X = combined[FEATURE_COLS].values.astype(np.float32)
    y = combined[LABEL_COL].values.astype(np.float32)
    return X, y


def prepare_approach1(layer_idx, data_root, datasets=None, val_ratio=0.2, seed=42):
    """
    Approach 1: Naive contiguous baseline.

    Uses all 12 features exactly as they appear in the CSV — prob_gap is
    computed from consecutive layers during data generation, which may not
    match SpecEE's runtime dynamic scheduling. This is our baseline.

    Returns:
        X_train, X_val, y_train, y_val — numpy arrays
    """
    X, y = load_layer_data(layer_idx, data_root, datasets)
    return train_test_split(X, y, test_size=val_ratio, random_state=seed, stratify=y)


def _recompute_gaps_for_skip(data_root, dataset, layer_idx, skip):
    """
    Recompute gap features for a given skip distance.

    For skip=k, gap = prob[layer_idx] - prob[layer_idx - k].
    If layer_idx - k < 0, gap = prob (same as inference when last_prob is None).

    Returns a DataFrame with the same columns as the original, but gap columns replaced.
    """
    current = load_layer_csv(data_root, dataset, layer_idx)

    prev_layer = layer_idx - skip
    if prev_layer < 0:
        # No previous layer available — gap equals prob (mirrors inference behavior
        # when last_prob is None at the first evaluated layer in the schedule)
        current[GAP_COLS] = current[PROB_COLS].values
    else:
        prev = load_layer_csv(data_root, dataset, prev_layer)
        # Recompute gap as current_prob - prev_prob
        current[GAP_COLS] = current[PROB_COLS].values - prev[PROB_COLS].values

    return current


def prepare_approach3(layer_idx, data_root, datasets=None, val_ratio=0.2, seed=42):
    """
    Approach 3: Augmented gaps to simulate SpecEE's dynamic scheduling.

    For each sample, we create up to 3 variants with different skip distances:
      - skip=1: original contiguous gap (as-is from naive data)
      - skip=2: gap = prob[L] - prob[L-2] (simulates skipping one layer)
      - skip=3: gap = prob[L] - prob[L-3] (simulates skipping two layers)

    The SpecEE default schedule [15,16,17,19,21,24,26,28] has these skip patterns:
      15: first (gap=prob), 16: skip=1, 17: skip=1, 19: skip=2,
      21: skip=2, 24: skip=3, 26: skip=2, 28: skip=2
    The dynamic expansion can add layers +-2 from recent exits, creating varied skips.

    By training on all three skip variants, the MLP learns to handle any skip distance
    it may encounter at runtime.

    Edge cases:
      - layer_idx=0: all skip variants produce gap=prob (no previous layers exist)
      - layer_idx=1: skip=2,3 produce gap=prob

    Returns:
        X_train, X_val, y_train, y_val — numpy arrays
    """
    if datasets is None:
        datasets = ALL_DATASETS

    all_frames = []
    for ds in datasets:
        for skip in [1, 2, 3]:
            augmented = _recompute_gaps_for_skip(data_root, ds, layer_idx, skip)
            all_frames.append(augmented)

    combined = pd.concat(all_frames, ignore_index=True)
    X = combined[FEATURE_COLS].values.astype(np.float32)
    y = combined[LABEL_COL].values.astype(np.float32)

    return train_test_split(X, y, test_size=val_ratio, random_state=seed, stratify=y)
