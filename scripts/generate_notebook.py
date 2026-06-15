"""Generate `pipeline.ipynb`, a reproducible walkthrough of the full pipeline:
dataset ingestion -> forensic feature extraction -> model training -> metrics.

This script writes raw nbformat-4 JSON directly (no `nbformat` dependency).
Run once from the project root:

    python scripts/generate_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines][:-1] + [lines[-1]] if lines else []}


def code(*lines: str) -> dict:
    src = [l + "\n" for l in lines][:-1] + ([lines[-1]] if lines else [])
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


cells = []

cells.append(md(
    "# Forensic Audio Deepfake Detection -- End-to-End Pipeline",
    "",
    "This notebook walks through the complete pipeline used by this project:",
    "",
    "1. **Dataset ingestion** -- crawl the FoR (`for-norm`) corpus and inspect class balance.",
    "2. **Forensic feature extraction** -- LFCC + delta + delta-delta biomarkers.",
    "3. **Model architecture** -- CNN-LSTM binary classifier.",
    "4. **Training** -- automated hyperparameter search against validation thresholds.",
    "5. **Metrics** -- accuracy, F1, per-class accuracy, confusion matrix, ROC/EER.",
    "6. **Inference demo** -- run the trained detector on a single file.",
    "",
    "All heavy lifting lives in `src/`; this notebook simply calls into it so the",
    "logic stays in one place and stays in sync with `predict.py` and `app/app.py`.",
))

cells.append(code(
    "import sys",
    "from pathlib import Path",
    "",
    "ROOT = Path.cwd()",
    "if not (ROOT / 'src').exists():",
    "    ROOT = ROOT.parent",
    "sys.path.insert(0, str(ROOT))",
    "",
    "import json",
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    "",
    "from src import config",
    "from src.features.feature_extraction import load_audio, extract_features",
    "from src.features.preprocess import gather_files",
))

cells.append(md("## 1. Dataset ingestion", "",
                 "The pipeline crawls `data/raw/for-norm/for-norm/{training,validation,testing}/{fake,real}` ",
                 "and maps `fake -> 1` (deepfake) and `real -> 0` (genuine)."))

cells.append(code(
    "for split in config.SPLITS:",
    "    items = gather_files(split)",
    "    n_fake = sum(1 for _, label in items if label == 1)",
    "    n_real = sum(1 for _, label in items if label == 0)",
    "    print(f'{split:>10}: total={len(items):6d}  genuine={n_real:6d}  deepfake={n_fake:6d}')",
))

cells.append(md("## 2. Forensic feature extraction", "",
                 "Every clip is resampled to 16 kHz mono, trimmed/padded to "
                 f"4 seconds, and converted into a "
                 "`(FEATURE_DIM, NUM_FRAMES)` = `(60, 401)` tensor of "
                 "LFCC (20 coeffs) + delta + delta-delta features."))

cells.append(code(
    "sample_real = gather_files('training')",
    "real_path = next(p for p, label in sample_real if label == 0)",
    "fake_path = next(p for p, label in sample_real if label == 1)",
    "",
    "waveform_real = load_audio(real_path)",
    "waveform_fake = load_audio(fake_path)",
    "",
    "feat_real = extract_features(waveform_real)",
    "feat_fake = extract_features(waveform_fake)",
    "print('feature shape:', feat_real.shape)",
))

cells.append(code(
    "fig, axes = plt.subplots(2, 2, figsize=(11, 6))",
    "",
    "axes[0, 0].plot(waveform_real.squeeze().numpy(), color='#4ade80', lw=0.5)",
    "axes[0, 0].set_title('Genuine -- waveform')",
    "axes[0, 1].plot(waveform_fake.squeeze().numpy(), color='#fb7185', lw=0.5)",
    "axes[0, 1].set_title('Deepfake -- waveform')",
    "",
    "axes[1, 0].imshow(feat_real, aspect='auto', origin='lower', cmap='magma')",
    "axes[1, 0].set_title('Genuine -- LFCC + Δ + ΔΔ')",
    "axes[1, 1].imshow(feat_fake, aspect='auto', origin='lower', cmap='magma')",
    "axes[1, 1].set_title('Deepfake -- LFCC + Δ + ΔΔ')",
    "",
    "fig.tight_layout()",
    "plt.show()",
))

cells.append(md("## 3. Preprocessed dataset", "",
                 "`src/features/preprocess.py` extracts features for every "
                 "(capped) file and serializes them to `data/processed/`."))

cells.append(code(
    "manifest_path = config.PROCESSED_DATA_DIR / 'manifest.json'",
    "if manifest_path.exists():",
    "    manifest = json.loads(manifest_path.read_text())",
    "    print(json.dumps(manifest, indent=2))",
    "else:",
    "    print('Run `python -m src.features.preprocess` first.')",
))

cells.append(md("## 4. Model architecture", "",
                 "A 1D CNN front-end (two Conv-BN-ReLU-MaxPool blocks) feeds a "
                 "bidirectional LSTM; mean-pooled LSTM outputs go through a small "
                 "MLP head producing a single deepfake-probability logit."))

cells.append(code(
    "from src.models.model import CNNLSTMDetector",
    "",
    "model = CNNLSTMDetector()",
    "n_params = sum(p.numel() for p in model.parameters())",
    "print(model)",
    "print(f'\\nTotal parameters: {n_params:,}')",
))

cells.append(md("## 5. Training", "",
                 "Training (with automated hyperparameter search against the "
                 "validation thresholds) is run via:", "",
                 "```bash",
                 "python -m src.models.train",
                 "```", "",
                 "The cell below loads the resulting metrics report."))

cells.append(code(
    "report_path = config.METRICS_REPORT_PATH",
    "if report_path.exists():",
    "    report = json.loads(report_path.read_text())",
    "    print('Selected config:', report['config']['label'])",
    "    print(json.dumps(report['validation'], indent=2)[:800])",
    "else:",
    "    print('Run `python -m src.models.train` first.')",
))

cells.append(md("## 6. Metrics", "",
                 "Validation accuracy / F1 / per-class accuracy / EER, plus the "
                 "confusion matrix and ROC curve with the EER operating point."))

cells.append(code(
    "if report_path.exists():",
    "    val = report['validation']",
    "    print(f\"Accuracy            : {val['accuracy']:.2%}\")",
    "    print(f\"F1-score            : {val['f1']:.2%}\")",
    "    print(f\"EER                 : {val['eer']:.2%}\")",
    "    print(f\"ROC AUC             : {val['auc']:.4f}\")",
    "    for cls, acc in val['per_class_accuracy'].items():",
    "        print(f'  {cls:<28}: {acc:.2%}')",
    "    print('Confusion matrix:')",
    "    print(np.array(val['confusion_matrix']))",
))

cells.append(code(
    "from matplotlib import image as mpimg",
    "",
    "if config.CONFUSION_MATRIX_PATH.exists() and config.ROC_CURVE_PATH.exists():",
    "    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))",
    "    axes[0].imshow(mpimg.imread(config.ROC_CURVE_PATH)); axes[0].axis('off')",
    "    axes[1].imshow(mpimg.imread(config.CONFUSION_MATRIX_PATH)); axes[1].axis('off')",
    "    fig.tight_layout()",
    "    plt.show()",
))

cells.append(md("## 7. Inference demo", "",
                 "Run the trained detector on a single file, exactly as "
                 "`predict.py` and the Streamlit app do."))

cells.append(code(
    "import torch",
    "from src.features.feature_extraction import extract_features_from_file, normalize",
    "",
    "if config.CHECKPOINT_PATH.exists():",
    "    ckpt = torch.load(config.CHECKPOINT_PATH, map_location='cpu', weights_only=False)",
    "    arch = ckpt['architecture']",
    "    model = CNNLSTMDetector(",
    "        input_dim=arch['input_dim'], cnn_channels=tuple(arch['cnn_channels']),",
    "        lstm_hidden=arch['lstm_hidden'], lstm_layers=arch['lstm_layers'],",
    "        bidirectional=arch['bidirectional'], dropout=arch['dropout'],",
    "    )",
    "    model.load_state_dict(ckpt['model_state_dict'])",
    "    model.eval()",
    "",
    "    demo_path = fake_path  # try real_path too",
    "    feats = extract_features_from_file(demo_path)",
    "    feats = normalize(feats, ckpt['norm_mean'], ckpt['norm_std'])",
    "    x = torch.from_numpy(feats).unsqueeze(0)",
    "    with torch.no_grad():",
    "        prob_fake = torch.sigmoid(model(x)).item()",
    "",
    "    label = config.CLASS_NAMES[1 if prob_fake >= 0.5 else 0]",
    "    print(f'File: {demo_path}')",
    "    print(f'Prediction: {label}')",
    "    print(f'P(deepfake) = {prob_fake:.4f}')",
    "else:",
    "    print('Run `python -m src.models.train` first.')",
))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = ROOT / "pipeline.ipynb"
out_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {out_path}")
