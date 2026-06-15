# VoiceShield: Audio Deepfake Detector

A production-grade pipeline and dashboard for detecting AI-generated
("deepfake") speech using classical acoustic analysis (LFCC + delta +
delta-delta cepstral biomarkers) and a CNN-LSTM binary classifier, trained on
the **FoR (Fake-or-Real) `for-norm`** corpus.

```
┌──────────────┐   ┌────────────────────┐   ┌───────────────────┐   ┌────────────────────┐
│ data/raw/...  │→→│ src/features/       │→→│ src/models/        │→→│ app/app.py          │
│ for-norm/...  │  │ preprocess.py       │  │ model.py, train.py │  │ Streamlit dashboard │
│ (.wav corpus) │  │ → data/processed/*  │  │ → models/detector.pt│ │ predict.py (CLI)    │
└──────────────┘   └────────────────────┘   └───────────────────┘   └────────────────────┘
```

---

## 1. Dataset

**Downloading the Data:**
Because the raw audio files are very large, they are not included in this repository. To replicate the training pipeline or run batch testing, please download the dataset from Kaggle:
1. Download the **Fake-or-Real (FoR) norm** dataset from [Kaggle](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset).
2. Extract the downloaded archive into the `data/raw/` directory.

**Source layout** (read-only, never modified):
Ensure the folder structure after extraction exactly matches the following:

```
data/raw/for-norm/for-norm/
├── training/  {fake, real}/*.wav
├── validation/{fake, real}/*.wav
└── testing/   {fake, real}/*.wav
```

**Class mapping** (`src/config.py::CLASS_MAP`):

| Folder            | Label | Meaning                    |
|-------------------|-------|----------------------------|
| `fake` / `spoof`  | `1`   | Deepfake / AI-generated    |
| `real` / `genuine`/ `bonafide` | `0` | Genuine / human speech |

Filenames carry processing-history suffixes (e.g.
`file1000.mp3.wav_16k.wav_norm.wav_mono.wav_silence.wav`) -- these are parsed
transparently by `soundfile` / `torchaudio`, no special-casing required.

To keep the end-to-end pipeline fast and reproducible, `preprocess.py` caps
the number of files processed per class (`src/config.py::MAX_FILES_PER_CLASS`).
Defaults: **5,000/class for training, 1,200/class for validation, and the
full testing split** (~4.6k files). Set any value to `None` to use every
file in that split.

---

## 2. Acoustic Feature Engineering (`src/features/feature_extraction.py`)

For every clip:

1. **Load & resample** to mono 16 kHz via `soundfile` (with a `torchaudio`
   fallback for formats `soundfile` can't decode, e.g. some `.mp3`s).
2. **Fix duration** -- trim or zero-pad to exactly **4.0 s** (64,000 samples)
   so every feature tensor has an identical shape regardless of the source
   clip's length.
3. **Linear Frequency Cepstral Coefficients (LFCC)** via
   `torchaudio.transforms.LFCC`:
   - `n_filter = 40` linearly spaced filterbank channels
   - `n_lfcc = 20` cepstral coefficients
   - STFT: `n_fft = 512`, `win_length = 400` (25 ms), `hop_length = 160` (10 ms)
   - Unlike Mel-scaled MFCCs, the **linear** frequency spacing of LFCCs
     preserves high-frequency detail where vocoder/TTS synthesis artifacts
     are most prominent.
4. **Delta & delta-delta** coefficients (`torchaudio.functional.compute_deltas`)
   capture frame-to-frame spectral dynamics -- highly discriminative for
   synthetic speech.
5. **Stack**: `[LFCC; Δ; ΔΔ]` → tensor of shape
   **`(FEATURE_DIM, NUM_FRAMES) = (60, 401)`**.
6. **Z-score normalize** every channel using mean/std computed on the
   *training* split only (`data/processed/norm_stats.npz`), applied
   identically at train and inference time.

This module is imported by `preprocess.py`, `train.py`, `predict.py`, and
`app/app.py` -- there is exactly one feature-extraction code path.

---

## 3. Model Architecture (`src/models/model.py`)

**`CNNLSTMDetector`** -- a 1D-CNN front end + bidirectional LSTM back end:

| Stage          | Spec                                                              |
|----------------|--------------------------------------------------------------------|
| Conv block 1   | `Conv1d(60→64, k=5, pad=2) → BatchNorm1d → ReLU → MaxPool1d(2)`     |
| Conv block 2   | `Conv1d(64→128, k=5, pad=2) → BatchNorm1d → ReLU → MaxPool1d(2)` + Dropout |
| LSTM           | 2-layer **bidirectional** LSTM, `hidden=128`, input=128            |
| Pooling        | Mean over time                                                     |
| Classifier head| `Linear(256→64) → ReLU → Dropout → Linear(64→1)` (logit)           |

The output is a single logit; `sigmoid(logit)` is interpreted as
**P(deepfake)**. Trained with `BCEWithLogitsLoss` and a class-balance
`pos_weight` derived from the training split.

---

## 4. Training & Hyperparameter Search (`src/models/train.py`)

```bash
python -m src.models.train
```

- Optimizer: Adam, `ReduceLROnPlateau` on training loss.
- Early stopping on validation EER (patience = 5 epochs).
- **Automated hyperparameter search**: a small grid of
  `(dropout, weight_decay, lr, pos_weight)` configurations is tried in order;
  the search stops as soon as a configuration satisfies *all* validation
  targets, otherwise the configuration with the lowest validation EER is kept.

**Verification thresholds** (`src/config.py`):

| Metric                     | Target   |
|----------------------------|----------|
| Overall accuracy           | ≥ 80%    |
| Equal Error Rate (EER)     | ≤ 12%    |
| F1-score                   | ≥ 80%    |
| Per-class accuracy (both)  | ≥ 75%    |

## 5. Metrics (`src/models/metrics.py`)

- **Accuracy, F1, confusion matrix** via scikit-learn.
- **Per-class accuracy**: diagonal of the confusion matrix normalized by row
  sums (genuine vs. deepfake).
- **Equal Error Rate (EER)**: computed from the ROC curve as the point where
  `FAR (= FPR)` and `FRR (= 1 - TPR)` are closest -- i.e. the FAR/FRR
  crossing point. Plotted on the ROC curve as the EER operating point.
- Outputs: `models/confusion_matrix.png`, `models/roc_curve.png`,
  `models/metrics_report.json` (validation + held-out test).

---

## 6. Dashboard (`app/app.py`)

A dark, Aceternity/shadcn-inspired Streamlit app:

- **Animated grid background** with a violet glow (pure CSS).
- **Drag-and-drop** `.wav` / `.mp3` upload with a built-in audio player.
- **Inference Readout**: a high-contrast `GENUINE (HUMAN)` /
  `DEEPFAKE (AI-GENERATED)` badge, animated confidence gauges for both
  classes, and a heatmap of the extracted LFCC + Δ + ΔΔ feature map.
- **Model Analytics** expander: live metrics pills (accuracy, F1, EER, AUC,
  per-class accuracy), the ROC/EER curve, and the confusion matrix heatmap.

```bash
streamlit run app/app.py
```

---

## 7. CLI Inference (`predict.py`)

```bash
python predict.py --file path/to/sample.wav
```

```
==================================================
 File        : sample.wav
 Prediction  : GENUINE (HUMAN)
 P(deepfake) :   3.42%
 P(genuine)  :  96.58%
==================================================
```

---

## 8. Local Setup

```bash
# 1. Create & activate a virtual environment (Python 3.13)
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Extract features (writes data/processed/*.npy)
python -m src.features.preprocess

# 4. Train (writes models/detector.pt + metrics_report.json + plots)
python -m src.models.train

# 5. Launch the dashboard
streamlit run app/app.py

# 6. Or run a single-file prediction from the terminal
python predict.py --file data/raw/for-norm/for-norm/testing/real/<some_file>.wav
```

---

## 9. Project Layout

```
audio-detecter/
├── data/
│   ├── raw/for-norm/for-norm/      # original corpus (read-only)
│   └── processed/                  # X_*.npy, y_*.npy, norm_stats.npz, manifest.json
├── src/
│   ├── config.py                   # single source of truth for paths/hyperparameters
│   ├── features/
│   │   ├── feature_extraction.py   # LFCC + Δ + ΔΔ, shared by every consumer
│   │   └── preprocess.py           # dataset crawler + feature serialization
│   └── models/
│       ├── model.py                # CNNLSTMDetector
│       ├── metrics.py              # accuracy/F1/EER/confusion matrix/ROC
│       └── train.py                # training + hyperparameter search
├── app/
│   └── app.py                      # Streamlit dashboard
├── models/                          # detector.pt, metrics_report.json, plots
├── predict.py                       # CLI inference
├── pipeline.ipynb                   # end-to-end notebook
└── requirements.txt
```
