"""Quick smoke test for the feature extraction + model pipeline.

Run from the project root:

    python scripts/sanity_check.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.config import RAW_DATA_ROOT, FEATURE_DIM, NUM_FRAMES
from src.features.feature_extraction import extract_features_from_file
from src.models.model import CNNLSTMDetector


def main():
    real_dir = RAW_DATA_ROOT / "training" / "real"
    fake_dir = RAW_DATA_ROOT / "training" / "fake"

    real_file = next(real_dir.glob("*.wav"))
    fake_file = next(fake_dir.glob("*.wav"))

    for label, f in [("real", real_file), ("fake", fake_file)]:
        t0 = time.time()
        feats = extract_features_from_file(f)
        dt = time.time() - t0
        print(f"{label}: {f.name} -> features {feats.shape} dtype={feats.dtype} "
              f"({dt*1000:.1f} ms)")
        assert feats.shape == (FEATURE_DIM, NUM_FRAMES)

    model = CNNLSTMDetector()
    x = torch.randn(4, FEATURE_DIM, NUM_FRAMES)
    out = model(x)
    print(f"model output shape: {tuple(out.shape)}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"total parameters: {n_params:,}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print("OK")


if __name__ == "__main__":
    main()
