"""Audio preprocessing pipeline.

Crawls the FoR (Fake-or-Real) `for-norm` dataset:

    data/raw/for-norm/for-norm/{training,validation,testing}/{fake,real}/*.wav

extracts LFCC + delta + delta-delta feature tensors for every clip (see
`src/features/feature_extraction.py`), z-score normalizes them using
statistics computed on the training split, and serializes the result to
`data/processed/` as:

    X_train.npy, y_train.npy
    X_val.npy,   y_val.npy
    X_test.npy,  y_test.npy
    norm_stats.npz   (mean, std -- shape (FEATURE_DIM, 1))
    manifest.json    (counts, label balance, config snapshot)

Run from the project root:

    python -m src.features.preprocess
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.config import (
    RAW_DATA_ROOT,
    PROCESSED_DATA_DIR,
    SPLITS,
    CLASS_MAP,
    MAX_FILES_PER_CLASS,
    FEATURE_DIM,
    NUM_FRAMES,
    SAMPLE_RATE,
    DURATION_SECONDS,
    N_LFCC,
    N_FILTER,
    N_FFT,
    WIN_LENGTH,
    HOP_LENGTH,
    RANDOM_SEED,
)
from src.features.feature_extraction import extract_features_from_file


def gather_files(split_dir_name: str) -> list[tuple[str, int]]:
    """Return [(filepath, label), ...] for one split, applying the per-class cap."""
    split_dir = RAW_DATA_ROOT / split_dir_name
    cap = MAX_FILES_PER_CLASS.get(split_dir_name)
    rng = random.Random(RANDOM_SEED)

    items: list[tuple[str, int]] = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = CLASS_MAP.get(class_dir.name.lower())
        if label is None:
            continue

        files = sorted(class_dir.glob("*.wav"))
        rng.shuffle(files)
        if cap is not None:
            files = files[:cap]

        items.extend((str(f), label) for f in files)

    return items


def _process_one(item: tuple[str, int]):
    path, label = item
    try:
        features = extract_features_from_file(path)
        return features, label, None
    except Exception as exc:  # pragma: no cover - defensive, logged not raised
        return None, None, f"{path}: {exc}"


def process_split(split_dir_name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    items = gather_files(split_dir_name)
    n = len(items)
    print(f"[{split_dir_name}] {n} files queued "
          f"(cap={MAX_FILES_PER_CLASS.get(split_dir_name)} per class)")

    features = np.zeros((n, FEATURE_DIM, NUM_FRAMES), dtype=np.float32)
    labels = np.zeros((n,), dtype=np.int64)
    errors: list[str] = []

    t0 = time.time()
    valid = 0
    with ProcessPoolExecutor() as executor:
        for feats, label, err in executor.map(_process_one, items, chunksize=32):
            if err is not None:
                errors.append(err)
                continue
            features[valid] = feats
            labels[valid] = label
            valid += 1

    features = features[:valid]
    labels = labels[:valid]
    elapsed = time.time() - t0
    print(f"[{split_dir_name}] extracted {features.shape} in {elapsed:.1f}s "
          f"({errors and len(errors) or 0} errors)")

    return features, labels, errors


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    split_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_errors: dict[str, list[str]] = {}

    for split_dir_name in SPLITS:
        X, y, errors = process_split(split_dir_name)
        split_arrays[split_dir_name] = (X, y)
        all_errors[split_dir_name] = errors

    # --- Normalization statistics computed on the training split only ---
    X_train, _ = split_arrays["training"]
    mean = X_train.mean(axis=(0, 2), keepdims=True)            # (1, FEATURE_DIM, 1)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8        # (1, FEATURE_DIM, 1)
    mean = mean[0]  # (FEATURE_DIM, 1)
    std = std[0]    # (FEATURE_DIM, 1)

    manifest = {
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": DURATION_SECONDS,
        "feature_dim": FEATURE_DIM,
        "num_frames": NUM_FRAMES,
        "n_lfcc": N_LFCC,
        "n_filter": N_FILTER,
        "n_fft": N_FFT,
        "win_length": WIN_LENGTH,
        "hop_length": HOP_LENGTH,
        "splits": {},
        "errors": {k: len(v) for k, v in all_errors.items()},
    }

    for split_dir_name, stem in SPLITS.items():
        X, y = split_arrays[split_dir_name]
        X_norm = (X - mean[None, :, :]) / std[None, :, :]

        np.save(PROCESSED_DATA_DIR / f"X_{stem}.npy", X_norm.astype(np.float32))
        np.save(PROCESSED_DATA_DIR / f"y_{stem}.npy", y.astype(np.int64))

        n_total = int(len(y))
        n_fake = int((y == 1).sum())
        n_real = int((y == 0).sum())
        manifest["splits"][stem] = {
            "total": n_total,
            "deepfake": n_fake,
            "genuine": n_real,
            "shape": list(X_norm.shape),
        }
        print(f"[{split_dir_name}] saved X_{stem}.npy {X_norm.shape}, "
              f"y_{stem}.npy -> genuine={n_real}, deepfake={n_fake}")

    np.savez(PROCESSED_DATA_DIR / "norm_stats.npz", mean=mean, std=std)

    with open(PROCESSED_DATA_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    for split_dir_name, errs in all_errors.items():
        for err in errs[:10]:
            print(f"  [warn] {split_dir_name}: {err}")

    print("\nPreprocessing complete. Artifacts written to", PROCESSED_DATA_DIR)


if __name__ == "__main__":
    main()
