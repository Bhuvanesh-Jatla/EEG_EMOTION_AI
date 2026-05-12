"""
train.py
========
Standalone training script. Run this to train all models on your DEAP data.

Usage:
    python train.py                          # Synthetic data (demo)
    python train.py --deap ./data            # Real DEAP data
    python train.py --deap ./data --subjects 10
"""

import argparse
import os
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="EEG Emotion AI — Training Script")
    parser.add_argument("--deap",     type=str, default=None,
                        help="Path to DEAP dataset folder (e.g., ./data)")
    parser.add_argument("--subjects", type=int, default=5,
                        help="Max number of subjects to load (default: 5)")
    parser.add_argument("--models-dir", type=str, default="models",
                        help="Directory to save trained models (default: models)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data instead of DEAP")
    args = parser.parse_args()

    use_synthetic = args.synthetic or (args.deap is None)
    deap_folder   = args.deap
    max_subjects  = args.subjects
    models_dir    = args.models_dir

    print("="*60)
    print("  EEG EMOTION AI — TRAINING SCRIPT")
    print("="*60)
    print(f"  Data source : {'Synthetic' if use_synthetic else deap_folder}")
    print(f"  Max subjects: {max_subjects}")
    print(f"  Models dir  : {models_dir}")
    print("="*60 + "\n")

    # ── 1. Load / Generate Data ────────────────
    from preprocessing import load_all_deap_subjects, generate_synthetic_deap

    if use_synthetic:
        print(f"[*] Generating synthetic DEAP-like data ({max_subjects} subjects)...")
        X, y_valence, y_arousal = generate_synthetic_deap(n_subjects=max_subjects)
    else:
        if not os.path.isdir(deap_folder):
            print(f"[ERROR] DEAP folder not found: '{deap_folder}'")
            sys.exit(1)
        print(f"[*] Loading DEAP data from '{deap_folder}'...")
        X, y_valence, y_arousal = load_all_deap_subjects(deap_folder, max_subjects)

    print(f"\n[*] Dataset: {X.shape[0]} windows × {X.shape[1]} channels × {X.shape[2]} samples")

    # ── 2. Feature Extraction ─────────────────
    print("\n[*] Extracting features...")
    from features import (extract_features_batch, compute_focus_labels_batch,
                           make_emotion_labels)

    features   = extract_features_batch(X, verbose=True)
    y_focus    = compute_focus_labels_batch(X)
    y_emotion  = make_emotion_labels(y_valence, y_arousal)

    print(f"\n    Feature matrix: {features.shape}")
    print(f"    Valence range:  {y_valence.min():.2f} – {y_valence.max():.2f}")
    print(f"    Arousal range:  {y_arousal.min():.2f} – {y_arousal.max():.2f}")
    print(f"    Focus range:    {y_focus.min():.2f} – {y_focus.max():.2f}")

    from numpy import bincount
    counts = bincount(y_emotion)
    emotion_names = ["Happy", "Calm", "Stress", "Sad"]
    print("    Emotion dist:   " +
          "  ".join(f"{emotion_names[i]}={counts[i]}" for i in range(len(counts))))

    # ── 3. Train All Models ────────────────────
    print("\n[*] Training models...")
    from models import run_full_training

    os.makedirs(models_dir, exist_ok=True)
    summary = run_full_training(
        features, y_valence, y_arousal, y_emotion, y_focus,
        test_size=0.2,
        models_dir=models_dir
    )

    # ── 4. Final Summary ──────────────────────
    print("\n" + "="*60)
    print("  ✅ TRAINING COMPLETE")
    print("="*60)
    print("\n  Best Models:")
    for task, model in summary["best_models"].items():
        print(f"    {task:10s} → {model}")

    print(f"\n  Saved to: {models_dir}/")
    print("  Run `streamlit run app.py` to launch the dashboard.")
    print("="*60)


if __name__ == "__main__":
    main()
