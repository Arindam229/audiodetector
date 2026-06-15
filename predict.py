#!/usr/bin/env python
"""Standalone CLI for the deepfake audio detector.

Usage
-----
    python predict.py --file path/to/sample.wav
    python predict.py --file path/to/sample.mp3 --checkpoint models/detector.pt

Loads the trained CNN-LSTM checkpoint, extracts the same LFCC + delta +
delta-delta features used during training, normalizes them with the stored
training statistics, and prints the predicted class with its probability.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Ensure `src` is importable regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import CHECKPOINT_PATH, CLASS_NAMES
from src.features.feature_extraction import extract_features_from_file, normalize
from src.models.model import CNNLSTMDetector


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = checkpoint["architecture"]
    model = CNNLSTMDetector(
        input_dim=arch["input_dim"],
        cnn_channels=tuple(arch["cnn_channels"]),
        lstm_hidden=arch["lstm_hidden"],
        lstm_layers=arch["lstm_layers"],
        bidirectional=arch["bidirectional"],
        dropout=arch["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def predict_file(path: str, model: CNNLSTMDetector, checkpoint: dict, device: torch.device):
    features = extract_features_from_file(path)
    features = normalize(features, checkpoint["norm_mean"], checkpoint["norm_std"])
    x = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.no_grad():
        prob_fake = torch.sigmoid(model(x)).item()
    threshold = checkpoint.get("decision_threshold", 0.5)
    label = 1 if prob_fake >= threshold else 0
    return label, prob_fake, threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Deepfake audio detector (CLI)")
    parser.add_argument("--file", required=True, help="Path to a .wav/.mp3 audio file")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH),
                         help=f"Path to a trained checkpoint (default: {CHECKPOINT_PATH})")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}. "
              f"Run `python -m src.models.train` first.", file=sys.stderr)
        sys.exit(1)

    audio_path = Path(args.file)
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(checkpoint_path, device)
    label, prob_fake, threshold = predict_file(str(audio_path), model, checkpoint, device)

    print("=" * 50)
    print(f" File        : {audio_path.name}")
    print(f" Prediction  : {CLASS_NAMES[label]}")
    print(f" P(deepfake) : {prob_fake:6.2%}")
    print(f" P(genuine)  : {1 - prob_fake:6.2%}")
    print(f" Threshold   : {threshold:6.2%}")
    print("=" * 50)


if __name__ == "__main__":
    main()
