"""
features.py
===========
EEG Feature Extraction Pipeline.

Extracts per-window features:
  - Power Spectral Density (PSD) per channel per band
  - Band power (absolute + relative)
  - Statistical features (mean, std, variance, skew, kurtosis)
  - Asymmetry features (left vs right hemisphere)
  - Focus pseudo-labels from band ratios (used to train Focus model)

Feature vector per window: ~1000–1500 features (depending on channels)
"""

import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from preprocessing import (
    SAMPLING_RATE, BAND_FREQS, EEG_CHANNELS,
    WINDOW_SAMPLES, filter_all_bands
)


# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

# DEAP standard 10-20 electrode positions (channel indices 0-31)
# Left hemisphere: F3,F7,C3,T7,P3,P7,O1 → approx indices
LEFT_CH  = [1, 3, 5, 7, 9, 11, 13]
RIGHT_CH = [2, 4, 6, 8, 10, 12, 14]

# Frontal channels for asymmetry (F3=1, F4=2 in DEAP)
FRONT_LEFT  = 1
FRONT_RIGHT = 2


# ──────────────────────────────────────────────
# PSD FEATURES
# ──────────────────────────────────────────────

def compute_psd(signal: np.ndarray, fs: int = SAMPLING_RATE,
                nperseg: int = None) -> tuple:
    """
    Compute Power Spectral Density using Welch's method.
    Returns: (frequencies, power)
    """
    if nperseg is None:
        nperseg = min(len(signal), 256)
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    return freqs, psd


def band_power_from_psd(freqs: np.ndarray, psd: np.ndarray,
                         low: float, high: float) -> float:
    """Integrate PSD within a frequency band using trapezoidal rule."""
    idx = np.logical_and(freqs >= low, freqs <= high)
    if idx.sum() == 0:
        return 0.0
    return float(np.trapezoid(psd[idx], freqs[idx]))


def extract_band_powers(signal: np.ndarray,
                         fs: int = SAMPLING_RATE) -> dict:
    """
    Extract absolute and relative band powers for a single channel.
    Returns dict with keys: delta, theta, alpha, beta, gamma + relative versions.
    """
    freqs, psd = compute_psd(signal, fs)
    total_power = max(np.trapezoid(psd, freqs), 1e-10)

    powers = {}
    for band, (low, high) in BAND_FREQS.items():
        bp = band_power_from_psd(freqs, psd, low, high)
        powers[f"abs_{band}"] = bp
        powers[f"rel_{band}"] = bp / total_power

    return powers


# ──────────────────────────────────────────────
# STATISTICAL FEATURES
# ──────────────────────────────────────────────

def extract_statistical_features(signal: np.ndarray) -> dict:
    """Extract time-domain statistical features from a 1D signal."""
    feats = {
        "mean":       float(np.mean(signal)),
        "std":        float(np.std(signal)),
        "variance":   float(np.var(signal)),
        "skewness":   float(skew(signal)),
        "kurtosis":   float(kurtosis(signal)),
        "max":        float(np.max(signal)),
        "min":        float(np.min(signal)),
        "peak_to_peak": float(np.ptp(signal)),
        "rms":        float(np.sqrt(np.mean(signal ** 2))),
        "zero_crossings": float(np.sum(np.diff(np.sign(signal)) != 0)),
    }
    return feats


# ──────────────────────────────────────────────
# ASYMMETRY FEATURES
# ──────────────────────────────────────────────

def compute_frontal_asymmetry(eeg: np.ndarray,
                               left_ch: int = FRONT_LEFT,
                               right_ch: int = FRONT_RIGHT,
                               fs: int = SAMPLING_RATE) -> dict:
    """
    Compute frontal alpha asymmetry (FAA): ln(right_alpha) - ln(left_alpha)
    Positive FAA → approach motivation / positive emotion
    """
    freqs_l, psd_l = compute_psd(eeg[left_ch],  fs)
    freqs_r, psd_r = compute_psd(eeg[right_ch], fs)

    alpha_low, alpha_high = BAND_FREQS["alpha"]
    alpha_l = band_power_from_psd(freqs_l, psd_l, alpha_low, alpha_high)
    alpha_r = band_power_from_psd(freqs_r, psd_r, alpha_low, alpha_high)

    alpha_l = max(alpha_l, 1e-10)
    alpha_r = max(alpha_r, 1e-10)

    faa = np.log(alpha_r) - np.log(alpha_l)

    # Theta asymmetry (related to workload)
    theta_low, theta_high = BAND_FREQS["theta"]
    theta_l = band_power_from_psd(freqs_l, psd_l, theta_low, theta_high)
    theta_r = band_power_from_psd(freqs_r, psd_r, theta_low, theta_high)
    theta_l = max(theta_l, 1e-10)
    theta_r = max(theta_r, 1e-10)
    fta = np.log(theta_r) - np.log(theta_l)

    return {
        "frontal_alpha_asymmetry": float(faa),
        "frontal_theta_asymmetry": float(fta),
    }


def compute_hemisphere_power_diff(eeg: np.ndarray,
                                   left_chs: list = LEFT_CH,
                                   right_chs: list = RIGHT_CH,
                                   fs: int = SAMPLING_RATE) -> dict:
    """Compute average power difference between hemispheres per band."""
    feats = {}
    for band, (low, high) in BAND_FREQS.items():
        left_powers  = []
        right_powers = []
        for ch in left_chs:
            if ch < eeg.shape[0]:
                f, p = compute_psd(eeg[ch], fs)
                left_powers.append(band_power_from_psd(f, p, low, high))
        for ch in right_chs:
            if ch < eeg.shape[0]:
                f, p = compute_psd(eeg[ch], fs)
                right_powers.append(band_power_from_psd(f, p, low, high))
        mean_l = np.mean(left_powers)  if left_powers  else 0.0
        mean_r = np.mean(right_powers) if right_powers else 0.0
        feats[f"hemi_diff_{band}"] = float(mean_r - mean_l)
    return feats


# ──────────────────────────────────────────────
# FOCUS PSEUDO-LABELS
# ──────────────────────────────────────────────

def compute_focus_score(eeg: np.ndarray, fs: int = SAMPLING_RATE) -> float:
    """
    Improved focus score using 4 neuroscience-backed ratios.
    """
    frontal_chs = list(range(min(8, eeg.shape[0])))
    parietal_chs = list(range(min(8, eeg.shape[0]), min(14, eeg.shape[0])))

    def mean_band(chs, band):
        powers = []
        for ch in chs:
            f, p = compute_psd(eeg[ch], fs)
            powers.append(band_power_from_psd(f, p, *BAND_FREQS[band]))
        return max(np.mean(powers), 1e-10)

    # Frontal bands
    f_alpha = mean_band(frontal_chs, "alpha")
    f_beta  = mean_band(frontal_chs, "beta")
    f_theta = mean_band(frontal_chs, "theta")
    f_delta = mean_band(frontal_chs, "delta")

    # Parietal alpha (attention suppression)
    p_alpha = mean_band(parietal_chs, "alpha") if parietal_chs else f_alpha

    # 4 ratios — all neuroscience backed
    beta_alpha   = f_beta  / f_alpha        # alertness
    theta_alpha  = f_theta / f_alpha        # working memory load
    engagement   = f_beta  / (f_alpha + f_theta)  # engagement index
    alpha_supp   = 1.0 / (p_alpha + 1e-10) # parietal alpha suppression → focus

    # Weighted combination
    raw = (
        0.35 * beta_alpha +
        0.25 * theta_alpha +
        0.25 * engagement +
        0.15 * min(alpha_supp, 10)   # cap to avoid extreme values
    )

    # Sigmoid to 0-100
    score = 100.0 / (1.0 + np.exp(-0.6 * (raw - 2.5)))
    return float(np.clip(score, 0, 100))


# ──────────────────────────────────────────────
# MAIN FEATURE EXTRACTION
# ──────────────────────────────────────────────

def extract_features_window(eeg_window: np.ndarray,
                              fs: int = SAMPLING_RATE) -> np.ndarray:
    n_channels = eeg_window.shape[0]
    
    # ALL channels now — no skipping
    ch_subset = list(range(n_channels))

    feature_list = []

    # 1. Band powers per channel (all channels)
    for ch in ch_subset:
        bp = extract_band_powers(eeg_window[ch], fs)
        feature_list.extend(bp.values())

    # 2. Statistical features per channel
    for ch in ch_subset:
        sf = extract_statistical_features(eeg_window[ch])
        feature_list.extend(sf.values())

    # 3. Frontal asymmetry
    if n_channels > max(FRONT_LEFT, FRONT_RIGHT):
        asym = compute_frontal_asymmetry(eeg_window, FRONT_LEFT, FRONT_RIGHT, fs)
        feature_list.extend(asym.values())

    # 4. Hemisphere power differences
    hemi = compute_hemisphere_power_diff(eeg_window, LEFT_CH, RIGHT_CH, fs)
    feature_list.extend(hemi.values())

    # 5. Global band statistics across ALL channels
    for band, (low, high) in BAND_FREQS.items():
        band_powers_all = []
        for ch in range(n_channels):
            f, p = compute_psd(eeg_window[ch], fs)
            band_powers_all.append(band_power_from_psd(f, p, low, high))
        feature_list.append(float(np.mean(band_powers_all)))
        feature_list.append(float(np.std(band_powers_all)))
        feature_list.append(float(np.max(band_powers_all)))
        feature_list.append(float(np.min(band_powers_all)))
        # NEW: median and range
        feature_list.append(float(np.median(band_powers_all)))
        feature_list.append(float(np.max(band_powers_all) - np.min(band_powers_all)))

    # 6. Inter-channel correlation (full matrix now)
    corr_mat  = np.corrcoef(eeg_window)
    upper_tri = corr_mat[np.triu_indices(n_channels, k=1)]
    feature_list.append(float(np.mean(upper_tri)))
    feature_list.append(float(np.std(upper_tri)))
    feature_list.append(float(np.max(upper_tri)))
    feature_list.append(float(np.min(upper_tri)))

    # 7. NEW: Theta/Alpha ratio per channel (cognitive load marker)
    for ch in range(n_channels):
        f, p = compute_psd(eeg_window[ch], fs)
        theta = band_power_from_psd(f, p, *BAND_FREQS["theta"])
        alpha = band_power_from_psd(f, p, *BAND_FREQS["alpha"])
        alpha = max(alpha, 1e-10)
        feature_list.append(float(theta / alpha))

    # 8. NEW: Beta/Alpha ratio per channel (alertness marker)
    for ch in range(n_channels):
        f, p = compute_psd(eeg_window[ch], fs)
        beta  = band_power_from_psd(f, p, *BAND_FREQS["beta"])
        alpha = band_power_from_psd(f, p, *BAND_FREQS["alpha"])
        alpha = max(alpha, 1e-10)
        feature_list.append(float(beta / alpha))

    # 9. NEW: (Theta + Alpha) / (Alpha + Beta) engagement index
    for ch in range(0, n_channels, 4):  # every 4th to keep size manageable
        f, p   = compute_psd(eeg_window[ch], fs)
        theta  = band_power_from_psd(f, p, *BAND_FREQS["theta"])
        alpha  = band_power_from_psd(f, p, *BAND_FREQS["alpha"])
        beta   = band_power_from_psd(f, p, *BAND_FREQS["beta"])
        denom  = max(alpha + beta, 1e-10)
        feature_list.append(float((theta + alpha) / denom))

    # Replace NaN/Inf
    feature_vector = np.array(feature_list, dtype=np.float32)
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
    return feature_vector


def get_feature_names(n_channels: int = EEG_CHANNELS,
                       use_all_channels: bool = False) -> list:
    """Return feature names matching extract_features_window output."""
    if use_all_channels:
        ch_subset = list(range(n_channels))
    else:
        ch_subset = list(range(0, n_channels, 4)) + [0, 1, 2, FRONT_LEFT, FRONT_RIGHT]
        ch_subset = list(set(ch_subset))
        ch_subset = [c for c in ch_subset if c < n_channels]

    names = []

    bp_keys = []
    for band in BAND_FREQS:
        bp_keys += [f"abs_{band}", f"rel_{band}"]
    sf_keys = ["mean", "std", "variance", "skewness", "kurtosis",
               "max", "min", "peak_to_peak", "rms", "zero_crossings"]

    for ch in ch_subset:
        for k in bp_keys:
            names.append(f"ch{ch}_{k}")
        for k in sf_keys:
            names.append(f"ch{ch}_{k}")

    names += ["frontal_alpha_asymmetry", "frontal_theta_asymmetry"]

    for band in BAND_FREQS:
        names.append(f"hemi_diff_{band}")

    for band in BAND_FREQS:
        names += [f"global_{band}_mean", f"global_{band}_std", f"global_{band}_max"]

    names += ["inter_ch_corr_mean", "inter_ch_corr_std"]

    return names


def extract_features_batch(X_windows: np.ndarray,
                             fs: int = SAMPLING_RATE,
                             verbose: bool = True) -> np.ndarray:
    """
    Extract features from all windows.

    Parameters:
        X_windows : (N, channels, samples)

    Returns:
        features  : (N, n_features)
    """
    N       = X_windows.shape[0]
    feat_0  = extract_features_window(X_windows[0], fs)
    n_feats = len(feat_0)

    features = np.zeros((N, n_feats), dtype=np.float32)
    features[0] = feat_0

    for i in range(1, N):
        if verbose and i % 500 == 0:
            print(f"  Features: {i}/{N} windows processed...")
        features[i] = extract_features_window(X_windows[i], fs)

    print(f"[Features] Extracted {n_feats} features from {N} windows")
    return features


def compute_focus_labels_batch(X_windows: np.ndarray,
                                fs: int = SAMPLING_RATE) -> np.ndarray:
    """Compute focus pseudo-labels for all windows."""
    N      = X_windows.shape[0]
    labels = np.zeros(N, dtype=np.float32)
    for i in range(N):
        labels[i] = compute_focus_score(X_windows[i], fs)
    print(f"[Focus] Pseudo-labels — min={labels.min():.1f}  max={labels.max():.1f}  mean={labels.mean():.1f}")
    return labels


def make_emotion_labels(y_valence: np.ndarray,
                         y_arousal: np.ndarray) -> np.ndarray:
    """
    Create SOFT emotion labels for Stage-2 training using valence/arousal quadrants.
    These are used ONLY to bootstrap the emotion model.
    The model learns the mapping — not hardcoded rules.

    Quadrant mapping (Russell's circumplex):
      High V + High A  → Happy  (0)
      High V + Low  A  → Calm   (1)
      Low  V + High A  → Stress (2)
      Low  V + Low  A  → Sad    (3)

    Boundary = median of DEAP labels (~5 on 1-9 scale)
    """
    mid = 5.0
    labels = np.zeros(len(y_valence), dtype=np.int32)
    for i, (v, a) in enumerate(zip(y_valence, y_arousal)):
        high_v = v >= mid
        high_a = a >= mid
        if high_v and high_a:
            labels[i] = 0   # Happy
        elif high_v and not high_a:
            labels[i] = 1   # Calm
        elif not high_v and high_a:
            labels[i] = 2   # Stress
        else:
            labels[i] = 3   # Sad
    return labels


EMOTION_NAMES = {0: "Happy", 1: "Calm", 2: "Stress", 3: "Sad"}
EMOTION_COLORS = {0: "#FFD700", 1: "#90EE90", 2: "#FF6347", 3: "#87CEEB"}


# ──────────────────────────────────────────────
# EXPLAINABILITY HELPERS
# ──────────────────────────────────────────────

def get_top_features_explanation(feature_vector: np.ndarray,
                                  feature_names: list,
                                  task: str = "focus",
                                  top_n: int = 3) -> list:
    """
    Simple feature-importance explanation based on known neuroscience.
    Returns list of human-readable strings.
    """
    explanations = []

    # Map feature names to human-readable insights
    insights = {
        "abs_beta":  "High beta activity → active thinking / alertness",
        "abs_theta": "High theta activity → working memory / mental load",
        "abs_alpha": "High alpha activity → relaxed / reduced focus",
        "abs_delta": "High delta activity → deep sleep / low arousal",
        "abs_gamma": "High gamma activity → cognitive processing",
        "rel_beta":  "Elevated relative beta → concentration",
        "rel_alpha": "Elevated relative alpha → calm / relaxed state",
        "frontal_alpha_asymmetry": "Positive frontal asymmetry → positive emotion",
        "hemi_diff_beta": "Right-hemisphere beta dominance → approach behavior",
    }

    # Find features with high absolute values
    abs_vals = np.abs(feature_vector)
    sorted_idx = np.argsort(abs_vals)[::-1]

    for idx in sorted_idx[:top_n * 5]:
        if idx >= len(feature_names):
            continue
        fname = feature_names[idx]
        for key, insight in insights.items():
            if key in fname:
                if insight not in explanations:
                    explanations.append(insight)
                if len(explanations) >= top_n:
                    break
        if len(explanations) >= top_n:
            break

    if not explanations:
        explanations = [
            "Beta/Alpha ratio indicates cognitive load",
            "Theta activity reflects working memory engagement",
            "Alpha suppression suggests focused attention",
        ]

    return explanations[:top_n]


# ──────────────────────────────────────────────
# MAIN (test)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from preprocessing import generate_synthetic_deap

    print("=== Testing Feature Extraction ===")
    X, yv, ya = generate_synthetic_deap(n_subjects=1, n_trials_per_subject=4)
    print(f"Windows: {X.shape}")

    feats = extract_features_batch(X[:20], verbose=True)
    print(f"Feature matrix: {feats.shape}")

    focus_labels = compute_focus_labels_batch(X[:20])
    emotion_labels = make_emotion_labels(yv[:20], ya[:20])
    print(f"Emotion label counts: {np.bincount(emotion_labels)}")

    names = get_feature_names()
    print(f"Feature names: {len(names)}")
