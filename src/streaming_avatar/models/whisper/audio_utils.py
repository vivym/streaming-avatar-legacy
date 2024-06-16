from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160


@lru_cache(maxsize=None)
def load_window(device: torch.device, n_fft: int = N_FFT) -> torch.Tensor:
    return torch.hann_window(n_fft, device=device)


@lru_cache(maxsize=None)
def load_mel_filters(device: torch.device, n_mels: int = N_MELS) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(Path(__file__).parent / "assets" / "mel_filters.npz") as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def compute_log_mel_spectrogram(audio: torch.Tensor, n_mels: int = N_MELS) -> torch.Tensor:
    window = load_window(device=audio.device, n_fft=N_FFT)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)

    magnitudes = stft[..., :-1].abs() ** 2

    filters = load_mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec
