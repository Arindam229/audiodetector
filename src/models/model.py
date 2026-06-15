"""CNN-LSTM architecture for deepfake audio detection.

The network treats the (FEATURE_DIM, NUM_FRAMES) LFCC + delta + delta-delta
tensor as a multi-channel 1D sequence:

  1. Two Conv1d -> BatchNorm -> ReLU -> MaxPool blocks extract local
     spectro-temporal patterns and down-sample the time axis.
  2. A bidirectional LSTM models long-range temporal dependencies across
     the resulting sequence (captures prosody / synthesis artifacts that
     span multiple frames).
  3. Mean-pooling over time followed by a small MLP head produces a single
     logit -- the probability (after sigmoid) that the clip is a deepfake.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import (
    FEATURE_DIM,
    CNN_CHANNELS,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    BIDIRECTIONAL,
    DROPOUT,
)


class CNNLSTMDetector(nn.Module):
    """2D CNN front-end + (bi)LSTM back-end binary classifier."""

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        cnn_channels: tuple[int, int] = CNN_CHANNELS,
        lstm_hidden: int = LSTM_HIDDEN,
        lstm_layers: int = LSTM_LAYERS,
        bidirectional: bool = BIDIRECTIONAL,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        c1, c2 = cnn_channels

        self.conv = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(c1, c2, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(dropout),
        )

        # After two MaxPool2d(2, 2), the frequency dimension (input_dim) is downsampled by 4.
        freq_bins = input_dim // 4

        self.lstm = nn.LSTM(
            input_size=c2 * freq_bins,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, FEATURE_DIM, NUM_FRAMES) -> logits: (batch,)"""
        x = x.unsqueeze(1)            # (batch, 1, freq, time)
        x = self.conv(x)              # (batch, c2, freq', time')
        
        batch_size, channels, freqs, time_frames = x.size()
        x = x.permute(0, 3, 1, 2)     # (batch, time', channels, freq')
        x = x.contiguous().view(batch_size, time_frames, channels * freqs)
        
        out, _ = self.lstm(x)         # (batch, time', lstm_out_dim)
        pooled = out.max(dim=1)[0]    # (batch, lstm_out_dim) using max-pooling
        logits = self.classifier(pooled).squeeze(-1)  # (batch,)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return the probability of the DEEPFAKE class (label 1)."""
        return torch.sigmoid(self.forward(x))


def build_model(**overrides) -> CNNLSTMDetector:
    """Factory used by train.py / predict.py / app.py for a consistent model."""
    return CNNLSTMDetector(**overrides)
