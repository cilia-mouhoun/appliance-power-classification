"""Spectral feature extraction for time-series data."""

import numpy as np
from scipy import signal
from typing import Dict


def extract_fft_features(data: np.ndarray) -> Dict[str, float]:
    """
    Extract FFT-based features from time-series data.
    
    Args:
        data: Input time-series data
        
    Returns:
        Dictionary of FFT features
    """
    fft = np.fft.fft(data)
    magnitude = np.abs(fft[:len(fft)//2])
    power = magnitude ** 2
    freqs = np.fft.fftfreq(len(data))[:len(fft)//2]
    
    # Find dominant frequency
    dominant_freq_idx = np.argmax(power)
    dominant_freq = freqs[dominant_freq_idx]
    
    features = {
        'fft_max_power': np.max(power),
        'fft_mean_power': np.mean(power),
        'fft_std_power': np.std(power),
        'dominant_freq': dominant_freq,
        'spectral_centroid': np.sum(freqs * power) / np.sum(power),
        'spectral_spread': np.sqrt(np.sum(((freqs - np.sum(freqs * power) / np.sum(power))**2) * power) / np.sum(power)),
    }
    return features


def extract_welch_features(data: np.ndarray, nperseg: int = 256) -> Dict[str, float]:
    """
    Extract Welch PSD features.
    
    Args:
        data: Input time-series data
        nperseg: Length of each segment for Welch's method
        
    Returns:
        Dictionary of Welch PSD features
    """
    freqs, psd = signal.welch(data, nperseg=min(nperseg, len(data)))
    
    features = {
        'welch_max_psd': np.max(psd),
        'welch_mean_psd': np.mean(psd),
        'welch_std_psd': np.std(psd),
        'welch_total_power': np.sum(psd),
    }
    return features


def extract_wavelet_features(data: np.ndarray, wavelet: str = 'db4', levels: int = 3) -> Dict[str, float]:
    """
    Extract wavelet-based features.
    
    Args:
        data: Input time-series data
        wavelet: Wavelet type
        levels: Decomposition levels
        
    Returns:
        Dictionary of wavelet features
    """
    try:
        import pywt
        
        features = {}
        coeffs = pywt.wavedec(data, wavelet, level=levels)
        
        for i, coeff in enumerate(coeffs):
            features[f'wavelet_level_{i}_energy'] = np.sum(coeff**2)
            features[f'wavelet_level_{i}_std'] = np.std(coeff)
        
        return features
    except ImportError:
        return {}
