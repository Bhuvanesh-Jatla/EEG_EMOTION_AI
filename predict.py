"""
predict.py
==========
Inference Engine for EEG Emotion AI.

Dual-mode prediction system:
  Mode 1: DEAP Demo    — load DEAP .dat, run full pipeline
  Mode 2: Custom EEG   — load user file (.npy / .csv / .dat)

Returns structured prediction results with confidence scores
and explainability insights.
"""

import os
import numpy as np
import pickle
from typing import Optional, Union

from preprocessing import (
    SAMPLING_RATE, EEG_CHANNELS, WINDOW_SAMPLES, STEP_SAMPLES,
    load_custom_eeg, preprocess_deap_subject, generate_synthetic_deap,
    extract_windows, normalize_eeg_matrix
)
from features import (
    extract_features_batch, extract_features_window,
    compute_focus_labels_batch, make_emotion_labels,
    get_feature_names, get_top_features_explanation,
    EMOTION_NAMES, EMOTION_COLORS
)
from models import load_best_models, MODELS_DIR

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

EMOTION_LABEL_MAP = {0: "Happy", 1: "Calm", 2: "Stress", 3: "Sad"}
EMOTION_EMOJI     = {0: "😄",     1: "😌",   2: "😤",     3: "😢"}
VALENCE_SCALE     = (1, 9)
AROUSAL_SCALE     = (1, 9)


# ──────────────────────────────────────────────
# PREDICTOR CLASS
# ──────────────────────────────────────────────

class EEGPredictor:
    """
    End-to-end EEG inference engine.
    Loads trained models and runs the full two-stage prediction pipeline.
    """

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir  = models_dir
        self.pipelines   = {}
        self.feature_names = get_feature_names()
        self._loaded     = False

    def load(self):
        """Load all best-model pipelines from disk."""
        print("[Predictor] Loading models...")
        self.pipelines = load_best_models(self.models_dir)
        self._loaded   = True
        print(f"[Predictor] Loaded tasks: {list(self.pipelines.keys())}")
        return self

    def _check_loaded(self):
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call predictor.load() first.")

    # ── Single Window Prediction ──────────────

    def predict_window(self, eeg_window: np.ndarray,
                        fs: int = SAMPLING_RATE) -> dict:
        """
        Predict all four outputs for a single EEG window.

        Parameters:
            eeg_window : np.ndarray  shape (channels, samples)

        Returns:
            result : dict with all predictions + confidence + explanation
        """
        self._check_loaded()

        # 1. Feature extraction
        feat_vec = extract_features_window(eeg_window, fs)
        feat_2d  = feat_vec.reshape(1, -1)

        result = {}

        # 2. Stage 1: Valence
        if "valence" in self.pipelines:
            val_raw = float(self.pipelines["valence"].predict(feat_2d)[0])
            val_clipped = float(np.clip(val_raw, *VALENCE_SCALE))
            result["valence"] = {
                "value":        round(val_clipped, 3),
                "raw":          round(val_raw, 3),
                "scale":        "1–9",
                "interpretation": valence_label(val_clipped),
            }

        # 3. Stage 1: Arousal
        if "arousal" in self.pipelines:
            aro_raw = float(self.pipelines["arousal"].predict(feat_2d)[0])
            aro_clipped = float(np.clip(aro_raw, *AROUSAL_SCALE))
            result["arousal"] = {
                "value":        round(aro_clipped, 3),
                "raw":          round(aro_raw, 3),
                "scale":        "1–9",
                "interpretation": arousal_label(aro_clipped),
            }

        # 4. Stage 2: Emotion (from predicted V + A)
        if "emotion" in self.pipelines and "valence" in result and "arousal" in result:
            va_input = np.array([[
                result["valence"]["value"],
                result["arousal"]["value"]
            ]])
            emotion_pred = int(self.pipelines["emotion"].predict(va_input)[0])
            emotion_name = EMOTION_LABEL_MAP.get(emotion_pred, "Unknown")

            # Confidence via predict_proba if available
            confidence = {}
            try:
                proba = self.pipelines["emotion"].predict_proba(va_input)[0]
                for i, p in enumerate(proba):
                    label = EMOTION_LABEL_MAP.get(i, str(i))
                    confidence[label] = round(float(p), 4)
                top_conf = round(float(max(proba)), 4)
            except Exception:
                top_conf = None
                confidence = {}

            result["emotion"] = {
                "label":      emotion_name,
                "class_id":   emotion_pred,
                "emoji":      EMOTION_EMOJI.get(emotion_pred, ""),
                "confidence": top_conf,
                "all_probs":  confidence,
                "color":      EMOTION_COLORS.get(emotion_pred, "#888"),
            }

        # 5. Focus Score
        if "focus" in self.pipelines:
            focus_raw  = float(self.pipelines["focus"].predict(feat_2d)[0])
            focus_score = float(np.clip(focus_raw, 0, 100))
            result["focus"] = {
                "score":     round(focus_score, 2),
                "category":  focus_category(focus_score),
                "scale":     "0–100",
            }

        # 6. Explainability
        result["explanation"] = get_top_features_explanation(
            feat_vec, self.feature_names,
            task="focus" if "focus" in result else "general",
            top_n=3
        )

        # 7. Band powers for visualization
        result["band_powers"] = _compute_mean_band_powers(eeg_window, fs)

        return result

    # ── Batch Prediction (many windows) ──────

    def predict_batch(self, X_windows: np.ndarray,
                       fs: int = SAMPLING_RATE,
                       aggregate: bool = True) -> Union[dict, list]:
        """
        Predict on multiple windows and optionally aggregate.

        Parameters:
            X_windows : (N, channels, samples)
            aggregate : if True, return mean/mode summary
                        if False, return list of per-window dicts
        """
        self._check_loaded()

        if len(X_windows) == 0:
            return {}

        features = extract_features_batch(X_windows, fs, verbose=False)
        n = len(features)

        # Valence + Arousal
        val_preds = aro_preds = None
        if "valence" in self.pipelines:
            val_preds = np.clip(self.pipelines["valence"].predict(features),
                                *VALENCE_SCALE)
        if "arousal" in self.pipelines:
            aro_preds = np.clip(self.pipelines["arousal"].predict(features),
                                *AROUSAL_SCALE)

        # Emotion
        emo_preds = emo_probs = None
        if "emotion" in self.pipelines and val_preds is not None and aro_preds is not None:
            va = np.stack([val_preds, aro_preds], axis=1)
            emo_preds = self.pipelines["emotion"].predict(va)
            try:
                emo_probs = self.pipelines["emotion"].predict_proba(va)
            except Exception:
                emo_probs = None

        # Focus
        focus_preds = None
        if "focus" in self.pipelines:
            focus_preds = np.clip(self.pipelines["focus"].predict(features), 0, 100)

        if not aggregate:
            results = []
            for i in range(n):
                r = {}
                if val_preds   is not None: r["valence"] = float(val_preds[i])
                if aro_preds   is not None: r["arousal"] = float(aro_preds[i])
                if emo_preds   is not None: r["emotion"] = EMOTION_LABEL_MAP.get(int(emo_preds[i]), "?")
                if focus_preds is not None: r["focus"]   = float(focus_preds[i])
                results.append(r)
            return results

        # Aggregate summary
        summary = {}
        if val_preds is not None:
            summary["valence"] = {
                "mean": round(float(np.mean(val_preds)), 3),
                "std":  round(float(np.std(val_preds)),  3),
                "min":  round(float(np.min(val_preds)),  3),
                "max":  round(float(np.max(val_preds)),  3),
                "interpretation": valence_label(float(np.mean(val_preds))),
                "timeseries": val_preds.tolist(),
            }
        if aro_preds is not None:
            summary["arousal"] = {
                "mean": round(float(np.mean(aro_preds)), 3),
                "std":  round(float(np.std(aro_preds)),  3),
                "min":  round(float(np.min(aro_preds)),  3),
                "max":  round(float(np.max(aro_preds)),  3),
                "interpretation": arousal_label(float(np.mean(aro_preds))),
                "timeseries": aro_preds.tolist(),
            }
        if emo_preds is not None:
            counts   = np.bincount(emo_preds.astype(int), minlength=4)
            dominant = int(np.argmax(counts))
            summary["emotion"] = {
                "dominant":  EMOTION_LABEL_MAP.get(dominant, "?"),
                "class_id":  dominant,
                "emoji":     EMOTION_EMOJI.get(dominant, ""),
                "counts":    {EMOTION_LABEL_MAP.get(i, str(i)): int(c)
                              for i, c in enumerate(counts)},
                "timeseries": [EMOTION_LABEL_MAP.get(int(e), "?") for e in emo_preds],
            }
            if emo_probs is not None:
                mean_probs = np.mean(emo_probs, axis=0)
                summary["emotion"]["mean_probs"] = {
                    EMOTION_LABEL_MAP.get(i, str(i)): round(float(p), 4)
                    for i, p in enumerate(mean_probs)
                }
        if focus_preds is not None:
            summary["focus"] = {
                "mean":     round(float(np.mean(focus_preds)), 2),
                "std":      round(float(np.std(focus_preds)),  2),
                "min":      round(float(np.min(focus_preds)),  2),
                "max":      round(float(np.max(focus_preds)),  2),
                "category": focus_category(float(np.mean(focus_preds))),
                "timeseries": focus_preds.tolist(),
            }

        # Add band powers (average across all windows)
        summary["band_powers"] = _compute_mean_band_powers(X_windows.mean(axis=0), fs)

        # Add explanation
        feat_mean = features.mean(axis=0)
        summary["explanation"] = get_top_features_explanation(
            feat_mean, get_feature_names(), top_n=3
        )

        return summary

    # ── Mode 1: DEAP Demo ─────────────────────

    def predict_from_deap(self, deap_file: str,
                           trial_idx: int = 0) -> dict:
        """
        Load a DEAP .dat file, pick a trial, run full inference.

        Returns: aggregated prediction summary
        """
        print(f"[DEAP Mode] Loading: {deap_file} | Trial: {trial_idx}")
        X_w, yv, ya, trial_ids = preprocess_deap_subject(deap_file)

        # Filter windows for selected trial
        trial_mask = [i for i, t in enumerate(trial_ids) if t == trial_idx]
        if not trial_mask:
            trial_mask = list(range(min(10, len(X_w))))

        X_trial = np.array([X_w[i] for i in trial_mask])
        print(f"[DEAP Mode] {len(X_trial)} windows for trial {trial_idx}")

        summary = self.predict_batch(X_trial, aggregate=True)
        summary["source"]      = f"DEAP: {os.path.basename(deap_file)} trial {trial_idx}"
        summary["ground_truth"] = {
            "valence": round(float(yv[trial_mask[0]]), 3),
            "arousal": round(float(ya[trial_mask[0]]), 3),
        }
        summary["raw_eeg"] = X_trial[0]   # First window for visualization
        return summary

    # ── Mode 2: Custom EEG ───────────────────

    def predict_from_file(self, filepath: str,
                           fs: int = SAMPLING_RATE) -> dict:
        """
        Load a custom EEG file and run full inference.

        Returns: aggregated prediction summary
        """
        print(f"[Custom Mode] Loading: {filepath}")
        eeg = load_custom_eeg(filepath, fs)

        windows = extract_windows(eeg)
        if not windows:
            raise ValueError("EEG file too short to extract windows.")

        X_windows = np.array(windows)
        print(f"[Custom Mode] {len(X_windows)} windows from file")

        summary = self.predict_batch(X_windows, aggregate=True)
        summary["source"]  = f"Custom: {os.path.basename(filepath)}"
        summary["raw_eeg"] = X_windows[0]
        return summary

    # ── Simulated Real-time Stream ────────────

    def predict_realtime_chunk(self, eeg_chunk: np.ndarray,
                                fs: int = SAMPLING_RATE) -> dict:
        """
        Predict from a single chunk (one window) for real-time demo.
        eeg_chunk shape: (channels, WINDOW_SAMPLES)
        """
        return self.predict_window(eeg_chunk, fs)


# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────

def valence_label(v: float) -> str:
    if v >= 7:   return "Very Positive 😊"
    if v >= 5.5: return "Positive"
    if v >= 4:   return "Neutral"
    if v >= 2.5: return "Negative"
    return "Very Negative 😞"


def arousal_label(a: float) -> str:
    if a >= 7:   return "Very Aroused / Excited"
    if a >= 5.5: return "Moderately Aroused"
    if a >= 4:   return "Neutral"
    if a >= 2.5: return "Low Arousal / Calm"
    return "Very Calm / Drowsy"


def focus_category(score: float) -> str:
    if score >= 70: return "High 🎯"
    if score >= 40: return "Medium 🔆"
    return "Low 😴"


def _compute_mean_band_powers(eeg: np.ndarray,
                               fs: int = SAMPLING_RATE) -> dict:
    """Compute average band power across all channels for visualization."""
    from preprocessing import BAND_FREQS
    from features import compute_psd, band_power_from_psd

    result = {}
    for band, (low, high) in BAND_FREQS.items():
        powers = []
        for ch in range(eeg.shape[0]):
            f, p = compute_psd(eeg[ch], fs)
            powers.append(band_power_from_psd(f, p, low, high))
        result[band] = round(float(np.mean(powers)), 6)
    return result


# ──────────────────────────────────────────────
# SIMULATE REAL-TIME EEG STREAM
# ──────────────────────────────────────────────

def generate_realtime_stream(predictor: EEGPredictor,
                               n_chunks: int = 10,
                               fs: int = SAMPLING_RATE,
                               n_channels: int = EEG_CHANNELS) -> list:
    """
    Simulate a real-time EEG stream and yield predictions.
    Used for the demo tab in Streamlit.

    Returns list of prediction dicts, one per chunk.
    """
    from preprocessing import generate_synthetic_deap

    X, _, _ = generate_synthetic_deap(
        n_subjects=1, n_trials_per_subject=1,
        duration_sec=max(n_chunks * 4, 60)
    )

    results = []
    for i in range(min(n_chunks, len(X))):
        chunk = X[i]    # (channels, window_samples)
        pred  = predictor.predict_window(chunk, fs)
        pred["chunk_id"] = i
        results.append(pred)

    return results


# ──────────────────────────────────────────────
# CLI DEMO
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("=== EEG Predictor Demo ===\n")

    # Try loading trained models
    try:
        predictor = EEGPredictor().load()
    except FileNotFoundError:
        print("[!] No trained models found. Training on synthetic data first...")

        from preprocessing import generate_synthetic_deap
        from features      import (extract_features_batch,
                                    compute_focus_labels_batch,
                                    make_emotion_labels)
        from models        import run_full_training

        X, yv, ya = generate_synthetic_deap(n_subjects=3)
        feats      = extract_features_batch(X)
        y_focus    = compute_focus_labels_batch(X)
        y_emot     = make_emotion_labels(yv, ya)
        run_full_training(feats, yv, ya, y_emot, y_focus)

        predictor = EEGPredictor().load()

    # Simulate a single prediction
    from preprocessing import generate_synthetic_deap
    X, _, _ = generate_synthetic_deap(n_subjects=1, n_trials_per_subject=1)
    result   = predictor.predict_window(X[0])

    print("\n📊 PREDICTION RESULT:")
    print(f"  Valence : {result['valence']['value']}  ({result['valence']['interpretation']})")
    print(f"  Arousal : {result['arousal']['value']}  ({result['arousal']['interpretation']})")
    print(f"  Emotion : {result['emotion']['emoji']} {result['emotion']['label']}  "
          f"(conf: {result['emotion']['confidence']})")
    print(f"  Focus   : {result['focus']['score']}  ({result['focus']['category']})")
    print(f"\n💡 Explanations:")
    for e in result["explanation"]:
        print(f"  • {e}")
    print(f"\n🌊 Band Powers:")
    for band, power in result["band_powers"].items():
        print(f"  {band:6s}: {power:.6f}")
