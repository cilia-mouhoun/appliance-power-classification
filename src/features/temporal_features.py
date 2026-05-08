"""Comprehensive temporal (time-domain) feature extraction."""

import numpy as np
import pandas as pd
from scipy import signal, stats
from typing import Dict
from ..utils.logger import logger


class TemporalFeatureExtractor:
    """
    Extract comprehensive temporal features from time-series data.
    
    Features extracted:
    - RMS (Root Mean Square): Energy in the signal
    - Variance: Signal variability
    - Skewness: Asymmetry of distribution
    - Kurtosis: Tailedness of distribution
    - Entropy: Information content/disorder
    - Zero Crossing Rate: Frequency of sign changes
    - Energy: Squared sum of samples
    - Peak statistics: Max value, peak count
    - Statistical moments: Mean, std, min, max, median
    - Autocorrelation: Signal self-similarity
    - Trend features: Slope, intercept, explained variance
    """
    
    def __init__(self):
        """Initialize feature extractor."""
        self.feature_names = []
        self._n_features = 0

    @property
    def n_features(self) -> int:
        """Get number of features."""
        return len(self.feature_names)
    
    def extract(self, X: np.ndarray) -> np.ndarray:
        """
        Extract temporal features from a 2D data matrix.

        Args:
            X: 2D array (n_samples, n_timesteps)

        Returns:
            2D array (n_samples, n_features)
        """
        feature_list = [self.extract_features(row) for row in X]
        return np.array([[v for v in d.values()] for d in feature_list])

    def extract_features(self, signal_data: np.ndarray, return_feature_names: bool = False) -> Dict[str, float]:
        """
        Extract all temporal features from a time-series signal.
        
        Args:
            signal_data: 1D numpy array of signal values
            return_feature_names: Included for API compatibility
            
        Returns:
            Dictionary of feature names and values
        """
        features = {}
        
        # Basic statistics
        features.update(self._extract_basic_statistics(signal_data))
        
        # Energy and power
        features.update(self._extract_energy_features(signal_data))
        
        # Zero crossing and peaks
        features.update(self._extract_zero_crossing_peaks(signal_data))
        
        # Autocorrelation
        features.update(self._extract_autocorrelation(signal_data))
        
        # Trend
        features.update(self._extract_trend(signal_data))
        
        # Entropy
        features.update(self._extract_entropy(signal_data))
        
        # Hjorth Parameters
        features.update(self._extract_hjorth_parameters(signal_data))
        
        # Additional temporal dynamics
        features.update(self._extract_additional_temporal_dynamics(signal_data))
        
        self.feature_names = list(features.keys())
        if return_feature_names:
            return np.array(list(features.values())), self.feature_names
        return features
    
    @staticmethod
    def _extract_basic_statistics(signal_data: np.ndarray) -> Dict[str, float]:
        """
        Extract basic statistical measures.
        
        Why these matter:
        - Different appliances have different average power consumption
        - Variability indicates operational patterns
        """
        return {
            'mean': np.mean(signal_data),
            'std': np.std(signal_data),
            'var': np.var(signal_data),
            'min': np.min(signal_data),
            'max': np.max(signal_data),
            'median': np.median(signal_data),
            'q25': np.percentile(signal_data, 25),
            'q75': np.percentile(signal_data, 75),
            'iqr': np.percentile(signal_data, 75) - np.percentile(signal_data, 25),
            'range': np.max(signal_data) - np.min(signal_data),
            'skewness': stats.skew(signal_data),
            'kurtosis': stats.kurtosis(signal_data),
        }
    
    @staticmethod
    def _extract_energy_features(signal_data: np.ndarray) -> Dict[str, float]:
        """
        Extract energy-related features.
        
        Why these matter:
        - RMS directly correlates with power consumption
        - Energy represents total work done by appliance
        - Useful for distinguishing high vs low power devices
        """
        rms = np.sqrt(np.mean(signal_data ** 2))
        energy = np.sum(signal_data ** 2)
        
        return {
            'rms': rms,
            'energy': energy,
            'mean_absolute_value': np.mean(np.abs(signal_data)),
            'peak_amplitude': np.max(np.abs(signal_data)),
        }
    
    @staticmethod
    def _extract_zero_crossing_peaks(signal_data: np.ndarray) -> Dict[str, float]:
        """
        Extract zero crossing rate and peak statistics.
        
        Why these matter:
        - Zero crossing rate indicates oscillation frequency
        - Peak count shows operational phases (on/off cycles)
        - Important for distinguishing motor-driven appliances
        """
        # Zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(signal_data)))) / (2 * len(signal_data))
        
        # Peaks
        peaks, _ = signal.find_peaks(signal_data, prominence=np.std(signal_data) * 0.5)
        peak_count = len(peaks)
        peak_values = signal_data[peaks] if len(peaks) > 0 else np.array([0])
        
        return {
            'zero_crossing_rate': zero_crossings,
            'peak_count': peak_count,
            'peak_mean': np.mean(peak_values),
            'peak_std': np.std(peak_values),
        }
    
    @staticmethod
    def _extract_autocorrelation(signal_data: np.ndarray, max_lag: int = 10) -> Dict[str, float]:
        """
        Extract autocorrelation features.
        
        Why these matter:
        - Autocorrelation reveals periodic patterns in signal
        - Many appliances have cyclic on/off patterns or ripple
        - Helps distinguish appliances with different operational frequencies
        """
        features = {}
        
        # Normalize signal
        signal_normalized = signal_data - np.mean(signal_data)
        
        # Compute autocorrelation
        acf_values = np.correlate(signal_normalized, signal_normalized, mode='full')
        acf_values = acf_values[len(acf_values) // 2:]
        acf_values = acf_values / acf_values[0]  # Normalize
        
        # Extract lags
        for lag in range(1, min(max_lag + 1, len(acf_values))):
            features[f'acf_lag_{lag}'] = acf_values[lag]
        
        return features
    
    @staticmethod
    def _extract_trend(signal_data: np.ndarray) -> Dict[str, float]:
        """
        Extract trend-related features.
        
        Why these matter:
        - Appliances may show increasing/decreasing trends over time
        - Warm-up behavior of fridges and ovens
        - Battery discharge patterns
        """
        x = np.arange(len(signal_data))
        try:
            coeffs = np.polyfit(x, signal_data, 1)
            trend = np.polyval(coeffs, x)
            residuals = signal_data - trend
            
            return {
                'trend_slope': coeffs[0],
                'trend_intercept': coeffs[1],
                'trend_r_squared': 1 - (np.var(residuals) / (np.var(signal_data) + 1e-10)),
                'detrended_std': np.std(residuals),
            }
        except Exception as e:
            return {
                'trend_slope': 0.0,
                'trend_intercept': 0.0,
                'trend_r_squared': 0.0,
                'detrended_std': np.std(signal_data),
            }
    
    @staticmethod
    def _extract_entropy(signal_data: np.ndarray) -> Dict[str, float]:
        """
        Extract entropy-based features.
        
        Why these matter:
        - Entropy measures complexity/randomness of signal
        - Stable appliances (fridges) have low entropy
        - Variable appliances (computers) have higher entropy
        """
        # Shannon entropy
        try:
            hist, _ = np.histogram(signal_data, bins=10, density=True)
            hist = hist[hist > 0]  # Remove zeros for log
            shannon_entropy = -np.sum(hist * np.log2(hist + 1e-10))
        except:
            shannon_entropy = 0.0
            
        # Approximate entropy (Simplified for robustness)
        try:
            # We use a simplified version to avoid heavy computation or convergence issues
            diff = np.abs(signal_data[1:] - signal_data[:-1])
            approx_entropy = np.mean(diff) / (np.std(signal_data) + 1e-10)
        except:
            approx_entropy = 0.0
            
        return {
            'shannon_entropy': shannon_entropy,
            'approximate_entropy': approx_entropy,
        }

    @staticmethod
    def _extract_hjorth_parameters(signal_data: np.ndarray) -> Dict[str, float]:
        """
        Extract Hjorth parameters: Activity, Mobility, and Complexity.
        
        Why these matter:
        - Activity: Variance of the signal, indicates power.
        - Mobility: Mean frequency, indicates the signal's variability.
        - Complexity: Frequency change, indicates how the signal resembles a sine wave.
        """
        # Activity
        activity = np.var(signal_data)
        
        # Mobility
        diff_1 = np.diff(signal_data)
        var_diff_1 = np.var(diff_1)
        mobility = np.sqrt(var_diff_1 / (activity + 1e-10))
        
        # Complexity
        diff_2 = np.diff(diff_1)
        var_diff_2 = np.var(diff_2)
        mobility_diff = np.sqrt(var_diff_2 / (var_diff_1 + 1e-10))
        complexity = mobility_diff / (mobility + 1e-10)
        
        return {
            'hjorth_activity': activity,
            'hjorth_mobility': mobility,
            'hjorth_complexity': complexity
        }

    @staticmethod
    def _extract_additional_temporal_dynamics(signal_data: np.ndarray) -> Dict[str, float]:
        """
        Extract additional dynamics: Waveform Length, Slope Sign Changes, and Turning Points.
        """
        # Waveform Length
        wl = np.sum(np.abs(np.diff(signal_data)))
        
        # Slope Sign Changes
        diff = np.diff(signal_data)
        ssc = np.sum((diff[:-1] * diff[1:]) < 0)
        
        # Turning Points
        # A point is a turning point if its neighbors are both larger or both smaller
        tp = np.sum(((signal_data[1:-1] > signal_data[:-2]) & (signal_data[1:-1] > signal_data[2:])) | 
                    ((signal_data[1:-1] < signal_data[:-2]) & (signal_data[1:-1] < signal_data[2:])))
        
        # Rolling stats (simplified for 1D)
        window = len(signal_data) // 10
        if window > 0:
            rolling_means = [np.mean(signal_data[i:i+window]) for i in range(0, len(signal_data), window)]
            rolling_stds = [np.std(signal_data[i:i+window]) for i in range(0, len(signal_data), window)]
            roll_mean_var = np.var(rolling_means)
            roll_std_mean = np.mean(rolling_stds)
        else:
            roll_mean_var = 0.0
            roll_std_mean = 0.0

        return {
            'waveform_length': wl,
            'slope_sign_changes': ssc,
            'turning_points': tp,
            'rolling_mean_var': roll_mean_var,
            'rolling_std_mean': roll_std_mean
        }


def extract_temporal_features(signal_data: np.ndarray) -> Dict[str, float]:
    """
    Extract all temporal features from a signal.
    
    Wrapper function for convenience.
    """
    extractor = TemporalFeatureExtractor()
    return extractor.extract_all(signal_data)
