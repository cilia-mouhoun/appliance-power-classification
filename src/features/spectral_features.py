
import numpy as np
from scipy import signal, fft
from typing import Dict
from ..utils.logger import logger

"""Comprehensive spectral (frequency-domain) feature extraction."""

# Placeholder for wavelet feature extraction (not implemented).
import pywt

def extract_wavelet_features(data: np.ndarray, sampling_rate: float = 1.0, wavelet: str = 'db4', levels: int = 3) -> Dict[str, float]:
    """Extract wavelet features (DWT coefficients statistics).
    
    Args:
        data: 1-D numpy array of the signal.
        sampling_rate: Sampling frequency.
        wavelet: Wavelet name.
        levels: Decomposition levels.
        
    Returns:
        Dictionary of wavelet features.
    """
    coeffs = pywt.wavedec(data, wavelet, level=levels)
    features = {}
    
    for i, c in enumerate(coeffs):
        level_name = 'approx' if i == 0 else f'detail_{levels-i+1}'
        features[f'wavelet_{level_name}_mean'] = np.mean(c)
        features[f'wavelet_{level_name}_std'] = np.std(c)
        features[f'wavelet_{level_name}_energy'] = np.sum(c**2)
        
    return features

def extract_fft_features(data: np.ndarray, sampling_rate: float = 1.0) -> Dict[str, float]:
    """Extract FFT-based spectral features using SpectralFeatureExtractor.

    Args:
        data: 1‑D numpy array of the signal.
        sampling_rate: Sampling frequency of the signal.

    Returns:
        Dictionary of FFT feature names and values.
    """
    extractor = SpectralFeatureExtractor(sampling_rate=sampling_rate)
    return extractor._extract_fft_features(data)

def extract_welch_features(data: np.ndarray, sampling_rate: float = 1.0, nperseg: int = 256) -> Dict[str, float]:
    """Extract PSD features via Welch's method using SpectralFeatureExtractor.

    Args:
        data: 1‑D numpy array of the signal.
        sampling_rate: Sampling frequency of the signal.
        nperseg: Length of each segment for Welch's method (currently unused).

    Returns:
        Dictionary of PSD feature names and values.
    """
    extractor = SpectralFeatureExtractor(sampling_rate=sampling_rate)
    return extractor._extract_psd_features(data)




class SpectralFeatureExtractor:
    """
    Extract comprehensive spectral (frequency-domain) features.
    
    Features extracted:
    - FFT-based: Max power, mean power, dominant frequency
    - Power Spectral Density (PSD): Via Welch's method
    - Spectral characteristics: Centroid, rolloff, entropy, spread
    - Harmonic features: Harmonic content, THD
    - Band powers: Power in specific frequency bands
    
    Why spectral features matter:
    - Different appliances operate at different frequencies
    - AC ripple, harmonics reveal appliance type
    - Switching frequencies unique to each device
    - Motor-driven appliances have distinctive frequency signatures
    """
    
    def __init__(self, sampling_rate: float = 1.0):
        """
        Initialize spectral feature extractor.
        
        Args:
            sampling_rate: Sampling frequency of signal
        """
        self.sampling_rate = sampling_rate
        self.feature_names = []
        self._n_features = 0

    @property
    def n_features(self) -> int:
        """Get number of features."""
        return len(self.feature_names)
    
    def extract(self, X: np.ndarray) -> np.ndarray:
        """
        Extract spectral features from a 2D data matrix.

        Args:
            X: 2D array (n_samples, n_timesteps)

        Returns:
            2D array (n_samples, n_features)
        """
        feature_list = [self.extract_features(row) for row in X]
        return np.array([[v for v in d.values()] for d in feature_list])

    def extract_features(
        self,
        signal_data: np.ndarray,
        spectral_bands: list = None,
        return_feature_names: bool = False
    ) -> Dict[str, float]:
        """
        Extract all spectral features from a time-series signal.
        
        Args:
            signal_data: 1D numpy array of signal values
            spectral_bands: List of frequency boundaries for band powers
            return_feature_names: Included for API compatibility
            
        Returns:
            Dictionary of feature names and values
        """
        features = {}
        
        # FFT features
        features.update(self._extract_fft_features(signal_data))
        
        # PSD features (Welch's method)
        features.update(self._extract_psd_features(signal_data))
        
        # Spectral characteristics
        features.update(self._extract_spectral_characteristics(signal_data))
        
        # Band powers
        if spectral_bands:
            features.update(self._extract_band_powers(signal_data, spectral_bands))
        
        # Harmonic analysis
        features.update(self._extract_harmonic_features(signal_data))
        
        # Wavelet features
        features.update(self._extract_wavelet_features(signal_data))
        
        self.feature_names = list(features.keys())
        if return_feature_names:
            return np.array(list(features.values())), self.feature_names
        return features
    
    def _extract_fft_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        Extract FFT-based features.
        
        Mathematical intuition:
        - FFT reveals frequency components in signal
        - High-frequency content indicates switching behavior
        - Fundamental and harmonic frequencies reveal device type
        """
        # Compute FFT
        fft_vals = np.fft.fft(signal_data)
        magnitude = np.abs(fft_vals)[:len(signal_data) // 2]
        power = magnitude ** 2
        freqs = np.fft.fftfreq(len(signal_data), 1/self.sampling_rate)[:len(signal_data) // 2]
        
        # Dominant frequency
        dominant_idx = np.argmax(power)
        dominant_freq = freqs[dominant_idx] if len(freqs) > 0 else 0
        
        # DC component (index 0)
        dc_component = power[0] if len(power) > 0 else 0
        
        # AC component (everything else)
        ac_power = np.sum(power[1:]) if len(power) > 1 else 0
        
        return {
            'fft_max_power': np.max(power),
            'fft_mean_power': np.mean(power),
            'fft_std_power': np.std(power),
            'fft_sum_power': np.sum(power),
            'dominant_frequency': dominant_freq,
            'dc_component': dc_component,
            'ac_power': ac_power,
            'dc_ac_ratio': dc_component / (ac_power + 1e-10),
        }
    
    def _extract_psd_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        Extract Power Spectral Density features using Welch's method.
        
        Why Welch:
        - Better spectral resolution than simple FFT
        - Less noisy due to windowing and averaging
        - Good for identifying narrow spectral peaks
        """
        nperseg = min(256, len(signal_data) // 4)
        freqs, psd = signal.welch(
            signal_data,
            fs=self.sampling_rate,
            nperseg=nperseg,
            window='hann'
        )
        
        return {
            'welch_max_psd': np.max(psd),
            'welch_mean_psd': np.mean(psd),
            'welch_std_psd': np.std(psd),
            'welch_total_power': np.trapz(psd, freqs),  # AUC of PSD
        }
    
    def _extract_spectral_characteristics(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        Extract spectral characteristics.
        
        Mathematical intuition:
        - Centroid: Average frequency (center of mass)
        - Spread: Width of frequency distribution
        - Entropy: Complexity of frequency distribution
        - Rolloff: Frequency containing 85% of energy
        """
        fft_vals = np.fft.fft(signal_data)
        magnitude = np.abs(fft_vals)[:len(signal_data) // 2]
        freqs = np.fft.fftfreq(len(signal_data), 1/self.sampling_rate)[:len(signal_data) // 2]
        
        # Spectral centroid
        spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
        
        # Spectral spread
        spectral_spread = np.sqrt(
            np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / (np.sum(magnitude) + 1e-10)
        )
        
        # Spectral entropy
        magnitude_norm = magnitude / (np.sum(magnitude) + 1e-10)
        spectral_entropy = -np.sum(magnitude_norm * np.log2(magnitude_norm + 1e-10))
        
        # Spectral rolloff (85th percentile)
        cumsum = np.cumsum(magnitude)
        rolloff_idx = np.argmax(cumsum >= 0.85 * cumsum[-1])
        spectral_rolloff = freqs[rolloff_idx] if rolloff_idx < len(freqs) else freqs[-1]
        
        return {
            'spectral_centroid': spectral_centroid,
            'spectral_spread': spectral_spread,
            'spectral_entropy': spectral_entropy,
            'spectral_rolloff': spectral_rolloff,
            'spectral_bandwidth': np.sqrt(np.sum(((freqs - spectral_centroid)**2) * magnitude) / (np.sum(magnitude) + 1e-10))
        }

    def _extract_wavelet_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        Extract wavelet-based features.
        
        Why Wavelets:
        - Capture both time and frequency information
        - Multi-resolution analysis (good for transients)
        - Different scales reveal different operational patterns
        """
        return extract_wavelet_features(signal_data, self.sampling_rate)
    
    def _extract_band_powers(self, signal_data: np.ndarray, bands: list) -> Dict[str, float]:
        """
        Extract power in specific frequency bands.
        
        Args:
            signal_data: Input signal
            bands: List of frequency boundaries [f0, f1, f2, ...]
                   Power extracted for bands [f0-f1], [f1-f2], etc.
        """
        nperseg = min(256, len(signal_data) // 4)
        freqs, psd = signal.welch(
            signal_data,
            fs=self.sampling_rate,
            nperseg=nperseg
        )
        
        features = {}
        for i in range(len(bands) - 1):
            band_mask = (freqs >= bands[i]) & (freqs < bands[i + 1])
            band_power = np.trapz(psd[band_mask], freqs[band_mask])
            features[f'band_power_{bands[i]}_to_{bands[i+1]}'] = band_power
        
        return features
    
    def _extract_harmonic_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        Extract harmonic analysis features.
        
        Mathematical intuition:
        - Fundamental frequency: First dominant peak
        - Harmonics: Multiples of fundamental
        - THD: Total Harmonic Distortion (energy in harmonics vs fundamental)
        """
        fft_vals = np.fft.fft(signal_data)
        magnitude = np.abs(fft_vals)[:len(signal_data) // 2]
        freqs = np.fft.fftfreq(len(signal_data), 1/self.sampling_rate)[:len(signal_data) // 2]
        
        # Find fundamental (skip DC at index 0)
        if len(magnitude) > 1:
            fundamental_idx = np.argmax(magnitude[1:]) + 1
            fundamental_power = magnitude[fundamental_idx] ** 2
            
            # Find harmonics (multiples of fundamental frequency)
            harmonic_powers = []
            for harmonic_num in range(2, 6):  # 2nd to 5th harmonic
                expected_idx = fundamental_idx * harmonic_num
                if expected_idx < len(magnitude):
                    harmonic_powers.append(magnitude[expected_idx] ** 2)
            
            # THD (Total Harmonic Distortion)
            harmonic_power_sum = np.sum(harmonic_powers) if harmonic_powers else 0
            thd = np.sqrt(harmonic_power_sum / (fundamental_power + 1e-10))
        else:
            fundamental_power = 0
            thd = 0
        
        return {
            'fundamental_power': fundamental_power,
            'total_harmonic_distortion': thd,
            'harmonic_content': harmonic_power_sum if 'harmonic_power_sum' in locals() else 0,
        }


def extract_spectral_features(signal_data: np.ndarray, sampling_rate: float = 1.0) -> Dict[str, float]:
    """
    Extract all spectral features from a signal.
    
    Wrapper function for convenience.
    """
    extractor = SpectralFeatureExtractor(sampling_rate=sampling_rate)
    return extractor.extract_features(signal_data)
