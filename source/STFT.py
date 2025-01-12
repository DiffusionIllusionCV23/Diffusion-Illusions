import torch
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
        Tensor: STFT of shape (B, C, H, N, W, N), where N = n_fft // 2 + 1 if onesided=True.
    """
    B, C, H, W = x.shape

    # Apply 1D STFT along the height dimension
    x_stft_h = torch.stft(
        x.permute(0, 1, 3, 2).reshape(B * C * W, H),  # Reshape to (B*C*W, H)
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=normalized,
        onesided=onesided,
        return_complex=False,
    )

    # Reshape back to (B, C, W, N, T_h,2)
    N = x_stft_h.shape[1]  # Number of frequency bins
    T_h = x_stft_h.shape[2]  # Number of time frames (height dimension)
    x_stft_h = x_stft_h.reshape(B, C, W, N, T_h,2)
    # Permute to (B, C, T_h, N, 2 , W)
    x_stft_h = x_stft_h.permute(0, 1, 4, 3, 5 , 2)

    # Apply 1D STFT along the width dimension
    x_stft_hw = torch.stft(
        x_stft_h.reshape(B * C * T_h * N * 2, W),  # Reshape to (B*C*T_h * N * 2, W)
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=normalized,
        onesided=onesided,
        return_complex=False,
    )
    # Reshape back to (B, C, T_h, N, T_w, N)
    T_w = x_stft_hw.shape[2]  # Number of time frames (width dimension)
    x_stft_hw = x_stft_hw.reshape(B, C, T_h, N,2, T_w, N,2)

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