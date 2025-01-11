import torch
import torch.nn.functional as F
import torch.nn as nn

def stft2d(x, n_fft, hop_length=None, win_length=None, window=None, center=True, pad_mode='reflect', normalized=False, onesided=True):
    """
    2D Short-Time Fourier Transform (STFT) with batch support.

    Args:
        x (Tensor): Input tensor of shape (B, C, H, W), where B is the batch size, C is the number of channels,
                    H is the height, and W is the width.
        n_fft (int): Size of the Fourier transform.
        hop_length (int): Stride of the sliding window.
        win_length (int): Size of the window.
        window (Tensor): Window function.
        center (bool): Whether to pad the input signal.
        pad_mode (str): Padding mode.
        normalized (bool): Whether to normalize the STFT.
        onesided (bool): Whether to return only the positive frequencies.

    Returns:
        Tensor: STFT of shape (B, C, H, n_freq, W, n_freq), where n_freq = n_fft // 2 + 1 if onesided=True.
    """
    B, C, H, W = x.shape

    # Apply 1D STFT along the height dimension
    x_stft_h = torch.stft(
        x.view(B * C, H, W),  # Reshape to (B*C, H, W) for batch processing
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=normalized,
        onesided=onesided,
        return_complex=True,
    )
    # Reshape back to (B, C, H, n_freq, W)
    x_stft_h = x_stft_h.view(B, C, *x_stft_h.shape[1:])

    # Apply 1D STFT along the width dimension
    x_stft_hw = torch.stft(
        x_stft_h.permute(0, 1, 3, 4, 2),  # Permute to (B, C, n_freq, W, H)
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=normalized,
        onesided=onesided,
        return_complex=True,
    )
    # Permute back to (B, C, H, n_freq, W, n_freq)
    x_stft_hw = x_stft_hw.permute(0, 1, 4, 2, 5, 3)

    return x_stft_hw

class STFT(nn.Module):
    def __init__(self, n_fft, hop_length=1, win_length=3, window=None, center=True, pad_mode='reflect', normalized=True, onesided=True):
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.onesided = onesided

    def forward(self, x):
        return stft2d(x, self.n_fft, self.hop_length, self.win_length, self.window, self.center, self.pad_mode, self.normalized, self.onesided)
