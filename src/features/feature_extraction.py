"""Acoustic feature extraction.

This module is the single source of truth for turning a raw waveform into
the fixed-shape tensor consumed by the CNN-LSTM detector. It is imported by
the offline preprocessing pipeline (`preprocess.py`), the training script,
the Streamlit dashboard, and the standalone `predict.py` CLI -- so every
consumer extracts features identically.

Acoustic biomarkers:
  * Linear Frequency Cepstral Coefficients (LFCC) -- unlike MFCCs, LFCCs use
    a linearly spaced filterbank, which better preserves high-frequency
    artifacts typical of vocoder / TTS synthesis pipelines.
  * Delta and delta-delta (acceleration) coefficients capture the
    frame-to-frame dynamics of the spectral envelope, which are highly
    discriminative for synthetic speech detection.

Every clip is resampled to 16 kHz mono and trimmed/zero-padded to a fixed
duration before feature extraction so that the resulting tensor always has
shape (FEATURE_DIM, NUM_FRAMES) = (60, 401).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

from src.config import (
    SAMPLE_RATE,
    NUM_SAMPLES,
    N_LFCC,
    N_FILTER,
    N_FFT,
    WIN_LENGTH,
    HOP_LENGTH,
    NUM_FRAMES,
    FEATURE_DIM,
)

# ---------------------------------------------------------------------------
# Transform (constructed once, reused for every clip)
# ---------------------------------------------------------------------------
_LFCC_TRANSFORM = torchaudio.transforms.LFCC(
    sample_rate=SAMPLE_RATE,
    n_filter=N_FILTER,
    n_lfcc=N_LFCC,
    speckwargs={
        "n_fft": N_FFT,
        "win_length": WIN_LENGTH,
        "hop_length": HOP_LENGTH,
        "center": True,
        "power": 2.0,
    },
)


def load_audio(path: Union[str, Path]) -> torch.Tensor:
    """Load an audio file as a mono float32 waveform at SAMPLE_RATE.

    Uses `soundfile` directly (robust on Windows / handles .wav natively),
    falling back to torchaudio's loader for formats soundfile can't read
    (e.g. some .mp3 files).
    """
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)  # (1, N)
        else:
            waveform = torch.from_numpy(data.T)  # (frames, channels) -> (channels, N)
    except Exception:
        waveform, sr = torchaudio.load(str(path))
        waveform = torch.as_tensor(waveform, dtype=torch.float32)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    return waveform


def fix_length(waveform: torch.Tensor, num_samples: int = NUM_SAMPLES) -> torch.Tensor:
    """Trim or loop-pad a (1, N) waveform to exactly `num_samples` samples.

    Clips shorter than `num_samples` are tiled (looped) rather than
    zero-padded. Zero-padding leaves a large constant block in the LFCC
    feature map -- since short clips are disproportionately common in one
    class, this becomes a trivial "silence block = class X" shortcut that
    the model memorizes but that does not transfer across dataset splits
    with different duration distributions.
    """
    length = waveform.shape[-1]
    if length > num_samples:
        waveform = waveform[..., :num_samples]
    elif length < num_samples:
        if length == 0:
            return torch.zeros((waveform.shape[0], num_samples), dtype=waveform.dtype)
        repeats = (num_samples + length - 1) // length
        waveform = waveform.repeat(1, repeats)[..., :num_samples]
    return waveform


def extract_features(waveform: torch.Tensor) -> np.ndarray:
    """Compute the (FEATURE_DIM, NUM_FRAMES) LFCC + delta + delta-delta tensor.

    Parameters
    ----------
    waveform : torch.Tensor
        Mono waveform of shape (1, N) at SAMPLE_RATE.

    Returns
    -------
    np.ndarray
        float32 array of shape (FEATURE_DIM, NUM_FRAMES), i.e. (60, 401).
    """
    waveform = fix_length(waveform)

    lfcc = _LFCC_TRANSFORM(waveform).squeeze(0)  # (N_LFCC, NUM_FRAMES)
    delta = torchaudio.functional.compute_deltas(lfcc)
    delta2 = torchaudio.functional.compute_deltas(delta)

    features = torch.cat([lfcc, delta, delta2], dim=0)  # (FEATURE_DIM, NUM_FRAMES)

    assert features.shape == (FEATURE_DIM, NUM_FRAMES), (
        f"Unexpected feature shape {tuple(features.shape)}, "
        f"expected {(FEATURE_DIM, NUM_FRAMES)}"
    )
    return features.numpy().astype(np.float32)


def extract_features_from_file(path: Union[str, Path]) -> np.ndarray:
    """Convenience wrapper: load a file and extract its feature tensor."""
    waveform = load_audio(path)
    return extract_features(waveform)


def extract_features_from_bytes(audio_bytes: bytes, suffix: str = ".wav") -> np.ndarray:
    """Extract features from raw audio bytes (e.g. a Streamlit upload).

    Writes to a temporary file on disk so that both the soundfile and
    torchaudio loaders can use their normal format-sniffing-by-extension
    code paths (more robust for compressed formats like .mp3 than an
    in-memory buffer).
    """
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        return extract_features_from_file(tmp_path)
    finally:
        os.unlink(tmp_path)


def normalize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply per-channel z-score normalization using stored training stats."""
    return (features - mean) / std
