"""
models.py
=========
Multi-Task EEG Emotion & Focus Prediction Models.

Trains and compares for EACH task:
  - Random Forest
  - XGBoost
  - SVM

Tasks:
  Stage 1: Valence (regression), Arousal (regression)
  Stage 2: Emotion (classification), Focus (regression)

Automatically selects best model per task and saves all models.
"""

import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.ensemble           import RandomForestRegressor, RandomForestClassifier
from sklearn.svm                import SVR, SVC
from sklearn.preprocessing      import StandardScaler, LabelEncoder
from sklearn.model_selection    import train_test_split, cross_val_score
from sklearn.metrics            import (
    mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report
)
from sklearn.pipeline           import Pipeline

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[Warning] XGBoost not installed. pip install xgboost")

from features import EMOTION_NAMES

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

TASK_TYPES = {
    "valence":  "regression",
    "arousal":  "regression",
    "emotion":  "classification",
    "focus":    "regression",
}

EMOTION_LABEL_MAP = {0: "Happy", 1: "Calm", 2: "Stress", 3: "Sad"}


# ──────────────────────────────────────────────
# MODEL DEFINITIONS
# ──────────────────────────────────────────────

def get_regression_models() -> dict:
    models = {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators=200,      # was 100
                max_depth=15,          # was 10
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42
            ))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  SVR(
                kernel="rbf",
                C=50,                  # was 10
                gamma="scale",
                epsilon=0.3            # was 0.5
            ))
        ]),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  XGBRegressor(
                n_estimators=300,      # was 100
                max_depth=7,           # was 6
                learning_rate=0.05,    # was 0.1 — slower but better
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,    # NEW — reduces overfitting
                reg_alpha=0.1,         # NEW — L1 regularization
                reg_lambda=1.5,        # NEW — L2 regularization
                n_jobs=-1,
                random_state=42,
                verbosity=0
            ))
        ])
    return models


def get_classification_models() -> dict:
    models = {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestClassifier(
                n_estimators=200,      # was 100
                max_depth=15,          # was 10
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42,
                class_weight="balanced"
            ))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  SVC(
                kernel="rbf",
                C=50,                  # was 10
                gamma="scale",
                probability=True,
                class_weight="balanced"
            ))
        ]),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  XGBClassifier(
                n_estimators=300,      # was 100
                max_depth=7,           # was 6
                learning_rate=0.05,    # was 0.1
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.5,
                n_jobs=-1,
                random_state=42,
                verbosity=0,
                eval_metric="mlogloss",
                use_label_encoder=False
            ))
        ])
    return models


# ──────────────────────────────────────────────
# TRAIN & EVALUATE ONE TASK
# ──────────────────────────────────────────────

def train_and_evaluate_task(task_name: str,
                              X_train: np.ndarray, X_test: np.ndarray,
                              y_train: np.ndarray, y_test: np.ndarray,
                              task_type: str) -> dict:
    """
    Train all models for a task, evaluate them, and return results.

    Returns:
        results : dict with keys = model names
                  each value = dict with metrics + fitted model
    """
    print(f"\n{'='*55}")
    print(f"  Training Task: {task_name.upper()} ({task_type})")
    print(f"{'='*55}")
    print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    if task_type == "regression":
        models = get_regression_models()
    else:
        models = get_classification_models()

    results = {}

    for model_name, pipeline in models.items():
        print(f"\n  [{model_name}] Training...")
        t0 = time.time()

        try:
            pipeline.fit(X_train, y_train)
            train_time = time.time() - t0

            y_pred = pipeline.predict(X_test)

            if task_type == "regression":
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                r2   = float(r2_score(y_test, y_pred))
                mae  = float(np.mean(np.abs(y_test - y_pred)))
                metrics = {"RMSE": rmse, "R2": r2, "MAE": mae}
                print(f"    RMSE={rmse:.4f}  R²={r2:.4f}  MAE={mae:.4f}  "
                      f"time={train_time:.1f}s")
            else:
                acc  = float(accuracy_score(y_test, y_pred))
                f1   = float(f1_score(y_test, y_pred, average="weighted",
                                       zero_division=0))
                metrics = {"Accuracy": acc, "F1": f1}
                print(f"    Acc={acc:.4f}  F1={f1:.4f}  time={train_time:.1f}s")

                # Print detailed report
                print(classification_report(
                    y_test, y_pred,
                    target_names=[EMOTION_LABEL_MAP.get(i, str(i))
                                  for i in sorted(set(y_test))],
                    zero_division=0
                ))

            results[model_name] = {
                "pipeline":   pipeline,
                "metrics":    metrics,
                "train_time": train_time,
                "y_pred":     y_pred,
                "y_test":     y_test,
            }

        except Exception as e:
            print(f"    ✗ {model_name} failed: {e}")

    return results


def select_best_model(results: dict, task_type: str) -> str:
    """
    Automatically select the best model based on primary metric.
    Regression → lowest RMSE
    Classification → highest F1
    """
    if not results:
        raise ValueError("No results to compare.")

    if task_type == "regression":
        best = min(results.keys(),
                   key=lambda m: results[m]["metrics"].get("RMSE", float("inf")))
    else:
        best = max(results.keys(),
                   key=lambda m: results[m]["metrics"].get("F1", 0.0))

    print(f"\n  ★ Best model for this task: {best}")
    return best


# ──────────────────────────────────────────────
# STAGE 2 EMOTION MODEL (trained on predicted V/A)
# ──────────────────────────────────────────────

def train_emotion_stage2(val_pred_train: np.ndarray,
                          aro_pred_train: np.ndarray,
                          y_emotion_train: np.ndarray,
                          val_pred_test:  np.ndarray,
                          aro_pred_test:  np.ndarray,
                          y_emotion_test: np.ndarray) -> dict:
    """
    Train emotion classifier on PREDICTED valence + arousal values.
    Input features = [valence_pred, arousal_pred]  shape (N, 2)
    This is Stage 2 of the two-stage pipeline.
    """
    X_train_va = np.stack([val_pred_train, aro_pred_train], axis=1)
    X_test_va  = np.stack([val_pred_test,  aro_pred_test],  axis=1)

    print("\n" + "="*55)
    print("  Stage 2: EMOTION MODEL (on predicted V/A)")
    print("="*55)
    print(f"  Input shape: {X_train_va.shape}")

    models = get_classification_models()
    results = {}

    for model_name, pipeline in models.items():
        print(f"\n  [{model_name}]")
        t0 = time.time()
        try:
            pipeline.fit(X_train_va, y_emotion_train)
            y_pred = pipeline.predict(X_test_va)

            acc = float(accuracy_score(y_emotion_test, y_pred))
            f1  = float(f1_score(y_emotion_test, y_pred,
                                  average="weighted", zero_division=0))
            print(f"    Acc={acc:.4f}  F1={f1:.4f}  time={time.time()-t0:.1f}s")
            print(classification_report(
                y_emotion_test, y_pred,
                target_names=["Happy", "Calm", "Stress", "Sad"],
                zero_division=0
            ))

            results[model_name] = {
                "pipeline":   pipeline,
                "metrics":    {"Accuracy": acc, "F1": f1},
                "train_time": time.time() - t0,
                "y_pred":     y_pred,
                "y_test":     y_emotion_test,
                "X_test":     X_test_va,
            }
        except Exception as e:
            print(f"    ✗ {model_name} failed: {e}")

    return results


# ──────────────────────────────────────────────
# MODEL PERSISTENCE
# ──────────────────────────────────────────────

def save_model(pipeline, task_name: str, model_name: str,
               models_dir: str = MODELS_DIR) -> str:
    """Save a fitted pipeline to disk."""
    fname = os.path.join(models_dir, f"{task_name}_{model_name}.pkl")
    with open(fname, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"  Saved: {fname}")
    return fname


def save_best_models(all_results: dict, best_models: dict,
                     models_dir: str = MODELS_DIR) -> dict:
    """
    Save the best model for each task.

    Parameters:
        all_results  : {task_name: {model_name: result_dict}}
        best_models  : {task_name: best_model_name}

    Returns:
        saved_paths  : {task_name: filepath}
    """
    os.makedirs(models_dir, exist_ok=True)
    saved = {}

    for task_name, model_name in best_models.items():
        if task_name not in all_results or model_name not in all_results[task_name]:
            print(f"  [Skip] {task_name} — no result for {model_name}")
            continue
        pipeline = all_results[task_name][model_name]["pipeline"]
        path     = save_model(pipeline, task_name, model_name, models_dir)
        saved[task_name] = path

    # Also save the registry (which model name is best per task)
    registry_path = os.path.join(models_dir, "model_registry.pkl")
    with open(registry_path, "wb") as f:
        pickle.dump(best_models, f)
    print(f"\n  Registry saved: {registry_path}")

    return saved


def load_best_models(models_dir: str = MODELS_DIR) -> dict:
    """
    Load all best-model pipelines from disk.

    Returns:
        pipelines : {task_name: fitted_pipeline}
    """
    registry_path = os.path.join(models_dir, "model_registry.pkl")
    if not os.path.exists(registry_path):
        raise FileNotFoundError(
            f"Model registry not found at '{registry_path}'. "
            "Please run training first."
        )

    with open(registry_path, "rb") as f:
        best_models = pickle.load(f)

    pipelines = {}
    for task_name, model_name in best_models.items():
        fname = os.path.join(models_dir, f"{task_name}_{model_name}.pkl")
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                pipelines[task_name] = pickle.load(f)
            print(f"  Loaded: {task_name} → {model_name}")
        else:
            print(f"  Missing: {fname}")

    return pipelines


# ──────────────────────────────────────────────
# FULL TRAINING PIPELINE
# ──────────────────────────────────────────────

def run_full_training(features:   np.ndarray,
                       y_valence:  np.ndarray,
                       y_arousal:  np.ndarray,
                       y_emotion:  np.ndarray,
                       y_focus:    np.ndarray,
                       test_size:  float = 0.2,
                       models_dir: str   = MODELS_DIR) -> dict:
    """
    Complete two-stage training pipeline.

    Returns:
        summary : {
            "all_results":  {task: {model: result}},
            "best_models":  {task: best_model_name},
            "saved_paths":  {task: filepath},
            "comparison_df":{task: DataFrame},
        }
    """
    print("\n" + "🧠 "*15)
    print("  EEG EMOTION AI — FULL TRAINING PIPELINE")
    print("🧠 "*15)
    print(f"\n  Dataset: {features.shape[0]} windows × {features.shape[1]} features")

    # ── Train/Test Split ──────────────────────
    idx = np.arange(len(features))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, random_state=42, shuffle=True
    )

    X_train = features[idx_train]
    X_test  = features[idx_test]

    # ── STAGE 1: Valence & Arousal ────────────
    all_results = {}
    best_models = {}

    for task_name, y_all in [("valence", y_valence), ("arousal", y_arousal)]:
        y_train = y_all[idx_train]
        y_test  = y_all[idx_test]

        results = train_and_evaluate_task(
            task_name, X_train, X_test, y_train, y_test,
            task_type="regression"
        )
        all_results[task_name] = results

        best = select_best_model(results, "regression")
        best_models[task_name] = best

    # ── Stage 1 Predictions (for Stage 2) ────
    best_val_pipe = all_results["valence"][best_models["valence"]]["pipeline"]
    best_aro_pipe = all_results["arousal"][best_models["arousal"]]["pipeline"]

    val_pred_train = best_val_pipe.predict(X_train)
    val_pred_test  = best_val_pipe.predict(X_test)
    aro_pred_train = best_aro_pipe.predict(X_train)
    aro_pred_test  = best_aro_pipe.predict(X_test)

    # ── STAGE 2: Emotion (on predicted V/A) ──
    emo_results = train_emotion_stage2(
        val_pred_train, aro_pred_train, y_emotion[idx_train],
        val_pred_test,  aro_pred_test,  y_emotion[idx_test]
    )
    all_results["emotion"] = emo_results
    best_models["emotion"] = select_best_model(emo_results, "classification")

    # ── STAGE 2: Focus (on EEG features) ─────
    focus_results = train_and_evaluate_task(
        "focus", X_train, X_test,
        y_focus[idx_train], y_focus[idx_test],
        task_type="regression"
    )
    all_results["focus"] = focus_results
    best_models["focus"] = select_best_model(focus_results, "regression")

    # ── Save Best Models ──────────────────────
    print("\n\n📦 SAVING BEST MODELS...")
    saved_paths = save_best_models(all_results, best_models, models_dir)

    # ── Build Comparison DataFrames ───────────
    comparison_dfs = {}
    for task_name, results in all_results.items():
        rows = []
        task_type = TASK_TYPES[task_name]
        for mname, rdict in results.items():
            row = {"Model": mname}
            row.update(rdict["metrics"])
            row["Train Time (s)"] = round(rdict["train_time"], 2)
            row["Best"] = "★" if mname == best_models[task_name] else ""
            rows.append(row)
        comparison_dfs[task_name] = pd.DataFrame(rows)

    # ── Print Summary ─────────────────────────
    print("\n\n" + "="*55)
    print("  FINAL MODEL COMPARISON SUMMARY")
    print("="*55)
    for task_name, df in comparison_dfs.items():
        print(f"\n  ── {task_name.upper()} ──")
        print(df.to_string(index=False))

    print("\n\n✅ TRAINING COMPLETE")
    print("  Best Models Selected:")
    for task, model in best_models.items():
        print(f"    {task:10s} → {model}")

    return {
        "all_results":   all_results,
        "best_models":   best_models,
        "saved_paths":   saved_paths,
        "comparison_dfs": comparison_dfs,
        "idx_train":     idx_train,
        "idx_test":      idx_test,
    }


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from preprocessing import generate_synthetic_deap
    from features      import (extract_features_batch,
                                compute_focus_labels_batch,
                                make_emotion_labels)

    print("=== Running Full Training on Synthetic Data ===")
    X, yv, ya = generate_synthetic_deap(n_subjects=3)
    feats   = extract_features_batch(X)
    y_focus = compute_focus_labels_batch(X)
    y_emot  = make_emotion_labels(yv, ya)

    summary = run_full_training(feats, yv, ya, y_emot, y_focus)
    print("\nDone!")
