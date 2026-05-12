"""
preprocessing.py
================
EEG Preprocessing Pipeline for DEAP Dataset and Custom EEG files.

Handles:
- Loading DEAP .dat files
- Bandpass filtering (delta, theta, alpha, beta, gamma)
- Normalization
- Windowing (2-4 second segments)
- Synthetic DEAP-like data generation (fallback/demo)
"""

import numpy as np
import pickle
import os
from scipy.signal import butter, filtfilt, welch

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
SAMPLING_RATE = 128          # DEAP sampling rate (Hz)
WINDOW_SEC    = 6            # Window size in seconds
WINDOW_SAMPLES = SAMPLING_RATE * WINDOW_SEC   # 512 samples per window
OVERLAP       = 0.5          # 50% overlap
STEP_SAMPLES  = int(WINDOW_SAMPLES * (1 - OVERLAP))

EEG_CHANNELS  = 32           # DEAP has 32 EEG channels (0–31)

BAND_FREQS = {
    "delta": (0.5, 4),
    "theta": (4,   8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}


# ──────────────────────────────────────────────
# FILTER UTILITIES
# ──────────────────────────────────────────────

def butter_bandpass(lowcut: float, highcut: float, fs: int, order: int = 4):
    """Design a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low  = lowcut  / nyq
    high = highcut / nyq
    low  = np.clip(low,  1e-6, 0.9999)
    high = np.clip(high, 1e-6, 0.9999)
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(signal: np.ndarray, lowcut: float, highcut: float,
                    fs: int = SAMPLING_RATE, order: int = 4) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter to a 1D signal."""
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, signal)


def apply_notch_filter(signal: np.ndarray, notch_freq: float = 50.0,
                       fs: int = SAMPLING_RATE, q: float = 30.0) -> np.ndarray:
    """Apply a simple notch filter (power-line noise removal)."""
    from scipy.signal import iirnotch
    b, a = iirnotch(notch_freq / (fs / 2), q)
    return filtfilt(b, a, signal)


def filter_all_bands(eeg_channel: np.ndarray, fs: int = SAMPLING_RATE) -> dict:
    """
    Filter a single EEG channel into all frequency bands.
    Returns dict: {band_name: filtered_signal}
    """
    filtered = {}
    for band, (low, high) in BAND_FREQS.items():
        filtered[band] = bandpass_filter(eeg_channel, low, high, fs)
    return filtered


# ──────────────────────────────────────────────
# NORMALIZATION
# ──────────────────────────────────────────────

def normalize_signal(signal: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize a signal using z-score or min-max."""
    if method == "zscore":
        std = np.std(signal)
        if std == 0:
            return signal - np.mean(signal)
        return (signal - np.mean(signal)) / std
    elif method == "minmax":
        rng = np.max(signal) - np.min(signal)
        if rng == 0:
            return np.zeros_like(signal)
        return (signal - np.min(signal)) / rng
    return signal


def normalize_eeg_matrix(eeg: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize each channel independently. Shape: (channels, samples)"""
    normed = np.zeros_like(eeg)
    for ch in range(eeg.shape[0]):
        normed[ch] = normalize_signal(eeg[ch], method)
    return normed


# ──────────────────────────────────────────────
# WINDOWING
# ──────────────────────────────────────────────

def extract_windows(eeg: np.ndarray, window_samples: int = WINDOW_SAMPLES,
                    step_samples: int = STEP_SAMPLES) -> list:
    """
    Slide a window over EEG data.
    Input shape:  (channels, total_samples)
    Output:       list of (channels, window_samples) arrays
    """
    n_samples = eeg.shape[1]
    windows   = []
    start     = 0
    while start + window_samples <= n_samples:
        windows.append(eeg[:, start: start + window_samples])
        start += step_samples
    return windows


# ──────────────────────────────────────────────
# DEAP DATASET LOADING
# ──────────────────────────────────────────────

def load_deap_subject(filepath: str) -> dict:
    """
    Load a single DEAP subject .dat file.
    Returns dict with keys: 'data', 'labels'
      data:   (40 trials, 40 channels, 8064 samples)  — EEG is first 32 channels
      labels: (40 trials, 4) — [valence, arousal, dominance, liking]
    """
    with open(filepath, "rb") as f:
        subject = pickle.load(f, encoding="latin1")
    return subject


def preprocess_deap_subject(filepath: str,
                             eeg_channels: int = EEG_CHANNELS,
                             fs: int = SAMPLING_RATE) -> tuple:
    """
    Full preprocessing pipeline for one DEAP subject.

    Returns:
        X_windows : list of np.ndarray  shape (eeg_channels, WINDOW_SAMPLES)
        y_valence : list of float       (one label per window, repeated from trial label)
        y_arousal : list of float       (one label per window, repeated from trial label)
        trial_ids : list of int         which trial each window belongs to
    """
    subject    = load_deap_subject(filepath)
    data       = subject["data"][:, :eeg_channels, :]   # (40, 32, 8064)
    labels     = subject["labels"]                       # (40, 4)

    # DEAP: first 3 seconds are pre-trial baseline → slice to 60 sec of actual data
    # Baseline: 128*3 = 384 samples → keep [384:]
    data = data[:, :, 384:]   # (40, 32, 7680) = 60 seconds

    X_windows  = []
    y_valence  = []
    y_arousal  = []
    trial_ids  = []

    for trial_idx in range(data.shape[0]):
        eeg_trial = data[trial_idx]                        # (32, 7680)

        # 1. Notch filter (power-line)
        for ch in range(eeg_channels):
            try:
                eeg_trial[ch] = apply_notch_filter(eeg_trial[ch], 50, fs)
            except Exception:
                pass

        # 2. Bandpass filter: keep 0.5–45 Hz (covers delta→gamma)
        for ch in range(eeg_channels):
            eeg_trial[ch] = bandpass_filter(eeg_trial[ch], 0.5, 45, fs)

        # 3. Normalize per channel
        eeg_trial = normalize_eeg_matrix(eeg_trial)

        # 4. Window
        windows = extract_windows(eeg_trial)

        val = labels[trial_idx, 0]   # valence  1–9
        aro = labels[trial_idx, 1]   # arousal  1–9

        for w in windows:
            X_windows.append(w)
            y_valence.append(val)
            y_arousal.append(aro)
            trial_ids.append(trial_idx)

    return X_windows, y_valence, y_arousal, trial_ids


def load_all_deap_subjects(deap_folder: str,
                            max_subjects: int = 32) -> tuple:
    """
    Load multiple DEAP subject files from a folder.
    Files should be named s01.dat, s02.dat, ... s32.dat

    Returns:
        X         : np.ndarray  (N_windows, eeg_channels, window_samples)
        y_valence : np.ndarray  (N_windows,)
        y_arousal : np.ndarray  (N_windows,)
    """
    all_X, all_val, all_aro = [], [], []

    dat_files = sorted([
        f for f in os.listdir(deap_folder)
        if f.endswith(".dat") and f.startswith("s")
    ])[:max_subjects]

    if not dat_files:
        raise FileNotFoundError(
            f"No DEAP .dat files found in '{deap_folder}'. "
            "Expected files like s01.dat, s02.dat, ..."
        )

    print(f"[DEAP] Loading {len(dat_files)} subject files from '{deap_folder}'...")

    for fname in dat_files:
        fpath = os.path.join(deap_folder, fname)
        try:
            X_w, y_v, y_a, _ = preprocess_deap_subject(fpath)
            all_X.extend(X_w)
            all_val.extend(y_v)
            all_aro.extend(y_a)
            print(f"  ✓ {fname}: {len(X_w)} windows")
        except Exception as e:
            print(f"  ✗ {fname}: Error — {e}")

    X         = np.array(all_X,  dtype=np.float32)
    y_valence = np.array(all_val, dtype=np.float32)
    y_arousal = np.array(all_aro, dtype=np.float32)

    print(f"[DEAP] Total windows: {X.shape[0]}  |  Shape: {X.shape}")
    return X, y_valence, y_arousal


# ──────────────────────────────────────────────
# CUSTOM EEG FILE LOADING
# ──────────────────────────────────────────────

def load_custom_eeg(filepath: str, fs: int = SAMPLING_RATE,
                    n_channels: int = EEG_CHANNELS) -> np.ndarray:
    """
    Load a custom EEG file for inference.

    Supports:
      - .npy  : numpy array  (channels, samples) or (samples, channels)
      - .csv  : rows = samples, columns = channels
      - .dat  : DEAP-format pickle (first trial used)

    Returns:
        eeg : np.ndarray  shape (n_channels, samples)
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".npy":
        eeg = np.load(filepath)
    elif ext == ".csv":
        import pandas as pd
        eeg = pd.read_csv(filepath, header=None).values
    elif ext == ".dat":
        subject = load_deap_subject(filepath)
        eeg = subject["data"][0, :n_channels, 384:]   # first trial
        return preprocess_raw_eeg(eeg, fs, n_channels)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .npy, .csv, or .dat")

    # Auto-orient: want (channels, samples)
    if eeg.ndim == 1:
        eeg = eeg.reshape(1, -1)
    if eeg.shape[0] > eeg.shape[1]:
        eeg = eeg.T

    eeg = eeg[:n_channels, :]
    return preprocess_raw_eeg(eeg, fs, n_channels)


def preprocess_raw_eeg(eeg: np.ndarray, fs: int = SAMPLING_RATE,
                        n_channels: int = EEG_CHANNELS) -> np.ndarray:
    """Apply full preprocessing to a raw EEG matrix (channels, samples)."""
    eeg = eeg.copy().astype(np.float32)
    for ch in range(min(n_channels, eeg.shape[0])):
        try:
            eeg[ch] = apply_notch_filter(eeg[ch], 50, fs)
        except Exception:
            pass
        eeg[ch] = bandpass_filter(eeg[ch], 0.5, 45, fs)
    eeg = normalize_eeg_matrix(eeg)
    return eeg


# ──────────────────────────────────────────────
# SYNTHETIC DATA (for demo / testing)
# ──────────────────────────────────────────────

def generate_synthetic_deap(n_subjects: int = 5,
                              n_trials_per_subject: int = 40,
                              fs: int = SAMPLING_RATE,
                              duration_sec: int = 60,
                              n_channels: int = EEG_CHANNELS,
                              seed: int = 42) -> tuple:
    """
    Generate synthetic DEAP-like EEG data for demo/testing.
    Emotion states drive frequency content to make features meaningful.

    Returns:
        X         : (N, channels, window_samples)
        y_valence : (N,)
        y_arousal : (N,)
    """
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, duration_sec, fs * duration_sec)

    all_X, all_val, all_aro = [], [], []

    emotion_profiles = {
        # (valence, arousal, dominant_band_hz, band_amplitude)
        "happy":  (7.5, 7.0, 10, 2.0),   # high alpha
        "calm":   (7.0, 3.0,  8, 1.5),   # alpha/theta
        "stress": (3.5, 7.5, 20, 2.5),   # beta dominant
        "sad":    (2.5, 3.0,  5, 1.0),   # theta dominant
    }
    emotions = list(emotion_profiles.keys())

    for _ in range(n_subjects):
        for trial in range(n_trials_per_subject):
            emotion = emotions[trial % len(emotions)]
            val, aro, dom_hz, amp = emotion_profiles[emotion]

            # Add small noise to labels for variability
            val_noisy = float(np.clip(val + rng.normal(0, 0.5), 1, 9))
            aro_noisy = float(np.clip(aro + rng.normal(0, 0.5), 1, 9))

            # Build synthetic multi-channel EEG
            eeg = np.zeros((n_channels, len(t)), dtype=np.float32)
            for ch in range(n_channels):
                # Base broadband noise
                base = rng.standard_normal(len(t)) * 0.5
                # Dominant frequency component
                dominant = amp * np.sin(2 * np.pi * dom_hz * t + rng.uniform(0, 2 * np.pi))
                # Secondary noise components
                sec = 0.3 * np.sin(2 * np.pi * rng.uniform(1, 45) * t)
                eeg[ch] = base + dominant + sec

            # Preprocess
            eeg = preprocess_raw_eeg(eeg, fs, n_channels)

            # Window
            windows = extract_windows(eeg)
            for w in windows:
                all_X.append(w)
                all_val.append(val_noisy)
                all_aro.append(aro_noisy)

    X         = np.array(all_X,  dtype=np.float32)
    y_valence = np.array(all_val, dtype=np.float32)
    y_arousal = np.array(all_aro, dtype=np.float32)

    print(f"[Synthetic] Generated {X.shape[0]} windows | Shape: {X.shape}")
    return X, y_valence, y_arousal


# ──────────────────────────────────────────────
# MAIN (test)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Testing Synthetic Data Generation ===")
    X, yv, ya = generate_synthetic_deap(n_subjects=2)
    print(f"X shape:   {X.shape}")
    print(f"Valence:   min={yv.min():.2f}  max={yv.max():.2f}  mean={yv.mean():.2f}")
    print(f"Arousal:   min={ya.min():.2f}  max={ya.max():.2f}  mean={ya.mean():.2f}")
