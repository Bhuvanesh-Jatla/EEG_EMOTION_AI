# Hi, I'm Bhuvanesh 
2nd year undergrad pursuing Computer Science engineering with specialization in cybersecurity. | ML & Signal Processing enthusiast  
studying in VIT chennai,India | Open to remote internships globally

## 🔥 Featured Project
**EEG Emotion AI** — Decoding brain states from EEG signals  
→ Two-stage ML pipeline | DEAP dataset | Streamlit dashboard  
→ [View Project](https://github.com/Bhuvanesh-Jatla/EEG_EMOTION_AI)

## 🛠 Skills
Python, C,C++,Java,Wireshark,Burpsuite,Nmap,Cisco packet tracer,  · XGBoost · Signal Processing · Streamlit · Git

## 📫 Reach me
[Linkldn] www.linkedin.com/in/bhuvanesh-jatla-0201b737b
[Email] (j.bhuvan30@gmail.com)
# 🧠 EEG Emotion AI — BrainTech 32+

> **Multi-Task EEG-Based Emotion & Focus Prediction System**
> Built for Clarity BrainTech 32+ (CMEEG-01) | DEAP Dataset | Two-Stage ML Pipeline

---
Copyright © 2026 Bhuvanesh Jatla and Kiran Nambiar

This project is publicly visible for educational and portfolio purposes only.

You may not copy, redistribute, modify, or use this code commercially without explicit permission.

## 🎯 Project Overview

A production-ready AI system that decodes **four brain states** from raw EEG signals:

| Task | Type | Output |
|------|------|--------|
| 💚 **Valence** | Regression | Score 1–9 |
| ⚡ **Arousal** | Regression | Score 1–9 |
| 🎭 **Emotion** | Classification | Happy / Calm / Stress / Sad |
| 🎯 **Focus** | Regression | Score 0–100 |

---
## 📦 Dataset & Models

This repo does not include large files directly.
Download them and place them in the correct folders:

### DEAP Dataset (preprocessed)
📥 [Download data/ folder](https://drive.google.com/drive/folders/193jcaRb9J_i90PPEOdpX6vgZjxmCYN62?usp=sharing)
→ Place contents inside `data/` folder

### Trained Model Files (.pkl)
📥 [Download models/ folder](https://drive.google.com/drive/folders/1DNBRisKwd0Yu1SzuozAQ293W34Tf43qM?usp=sharing)
→ Place contents inside `models/` folder

## 🚀 Setup Instructions
1. Clone this repo 
git clone https://github.com/Boo1230/EEG_EMOTION_AI.git
2. Download data and models from links above



## 🏗️ Folder Structure

```
eeg_emotion_ai/
├── preprocessing.py      ← EEG loading, filtering, normalization, windowing
├── features.py           ← PSD, band powers, statistics, asymmetry features
├── models.py             ← Train RF / XGBoost / SVM, evaluate, select best
├── predict.py            ← Inference engine (DEAP + Custom file modes)
├── train.py              ← Standalone CLI training script
├── app.py                ← Streamlit dark-theme dashboard
├── requirements.txt      ← Python dependencies
├── models/               ← Saved model files (auto-created after training)
│   ├── valence_*.pkl
│   ├── arousal_*.pkl
│   ├── emotion_*.pkl
│   ├── focus_*.pkl
│   └── model_registry.pkl
└── data/                 ← Place DEAP .dat files here (s01.dat … s32.dat)
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare DEAP Dataset

Place your DEAP `.dat` files in the `data/` folder:

```
data/
├── s01.dat
├── s02.dat
...
└── s32.dat
```

> **Don't have DEAP?** The system supports **synthetic demo mode** — just check "Use Synthetic Data" in the Streamlit sidebar.

### 3. Train models (CLI)

```bash
# With real DEAP data
python train.py --deap ./data --subjects 10

# With synthetic data (demo)
python train.py --synthetic --subjects 5
```

### 4. Launch Streamlit Dashboard

```bash
streamlit run app.py
```

Open: **http://localhost:8501**

---

## 🔁 Two-Stage Architecture

```
Raw EEG (32ch × 8064 samples per trial)
         ↓
┌────────────────────────────────────┐
│ PREPROCESSING                      │
│  • Notch filter (50 Hz)            │
│  • Bandpass filter (0.5–45 Hz)     │
│  • Z-score normalization           │
│  • Windowing (4s, 50% overlap)     │
└────────────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│ FEATURE EXTRACTION                 │
│  • PSD (Welch) per channel         │
│  • Band powers: δ θ α β γ         │
│  • Statistical features (10/ch)    │
│  • Frontal alpha asymmetry (FAA)   │
│  • Hemisphere power differences    │
│  • Inter-channel correlations      │
└────────────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│ STAGE 1 — Regression               │
│  Valence model: RF / XGB / SVM     │
│  Arousal model: RF / XGB / SVM     │
│  → Best model selected by RMSE     │
└────────────────────────────────────┘
         ↓ (predicted V + A)
┌────────────────────────────────────┐
│ STAGE 2 — Classification + Reg     │
│  Emotion: trained on pred. V/A     │
│    → Happy / Calm / Stress / Sad   │
│  Focus: trained on EEG features    │
│    → Score 0–100                   │
└────────────────────────────────────┘
```

---

## 🤖 Models Compared Per Task

Each task trains and evaluates **3 models**:

| Model | Valence | Arousal | Emotion | Focus |
|-------|---------|---------|---------|-------|
| Random Forest | ✅ | ✅ | ✅ | ✅ |
| XGBoost | ✅ | ✅ | ✅ | ✅ |
| SVM (RBF) | ✅ | ✅ | ✅ | ✅ |

**Selection criteria:**
- Regression tasks → lowest RMSE
- Classification tasks → highest weighted F1

---

## 🌊 Brainwave Bands

| Band | Frequency | Role |
|------|-----------|------|
| δ Delta | 0.5–4 Hz | Deep sleep, unconscious |
| θ Theta | 4–8 Hz | Working memory, creativity |
| α Alpha | 8–13 Hz | Relaxation, calm alertness |
| β Beta | 13–30 Hz | Active thinking, focus |
| γ Gamma | 30–45 Hz | Cognitive processing |

---

## 📊 Features Extracted (~1000–1500 per window)

| Feature Group | Count | Description |
|--------------|-------|-------------|
| Band powers (abs + rel) | 10 × N_ch | δ θ α β γ per channel |
| Statistical features | 10 × N_ch | mean, std, var, skew, kurtosis, rms… |
| Frontal asymmetry | 2 | FAA, frontal theta asymmetry |
| Hemisphere diffs | 5 | Left vs right per band |
| Global band stats | 15 | Mean/std/max across all channels |
| Inter-channel corr | 2 | Mean + std of correlation matrix |

---

## 🎨 Streamlit UI — Tabs

| Tab | Description |
|-----|-------------|
| 🏠 Home | System overview, architecture |
| 🧪 Train Models | One-click training with progress |
| 📊 Model Comparison | Bar charts + metric tables per task |
| 🔮 DEAP Demo | Full pipeline on DEAP trial |
| 📁 Custom EEG | Upload .npy / .csv / .dat file |
| ⚡ Real-time Stream | Simulated live EEG stream |

---

## 🧪 Supported Custom EEG Formats

| Format | Shape | Notes |
|--------|-------|-------|
| `.npy` | `(channels, samples)` or `(samples, channels)` | Auto-transposed |
| `.csv` | rows=samples, cols=channels | No header |
| `.dat` | DEAP pickle format | First trial used |

---

## 💡 Explainability

The system provides **neuroscience-grounded explanations** for each prediction:

```
• High beta activity → active thinking / alertness
• Elevated theta/alpha ratio → high cognitive load
• Positive frontal asymmetry → positive emotional state
```

---

## ⚡ Hardware Requirements

- **CPU only** (no GPU needed)
- Minimum 4 GB RAM (8 GB recommended)
- Training time: < 10 minutes for 5 subjects
- Python 3.9+

---

## 📦 Dependencies

```
numpy        scipy        pandas
scikit-learn xgboost      streamlit
plotly
```

---

## Internship Pitch

> *"We decode the invisible — translating raw electrical signals from the human brain into actionable emotion and focus insights using a scientifically grounded, fully learned, two-stage machine learning pipeline. No rules. No shortcuts. Just EEG → Intelligence."*

**Key differentiators:**
1. **Two-stage pipeline** — emotion is learned from predicted V/A, not hardcoded
2. **Multi-model comparison** — automatically selects best architecture per task
3. **Scientific feature engineering** — FAA, PSD, band asymmetry
4. **Dual inference modes** — DEAP dataset + custom EEG files
5. **Real-time demo** — simulated live EEG stream

---

*Built with Clarity BrainTech 32+ (CMEEG-01) — 32 Ch EEG USB Powered Machine*
*Marketed by DigiMed Technologies | Clarity Medical Pvt. Ltd.*
