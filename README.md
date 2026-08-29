# 🧠 DepressNet: An Interpretable Machine Learning Framework for Depression Severity Stratification Using Psychometric Data

### 🏆 85.38% Test Accuracy | 59.01% Macro F1 | 0.7158 MCC | Explainable | Statistically Validated

**Journal:** Frontiers in Computer Science (Human-Media Interaction)
**Status:** Accepted for Publication

---

## 👨‍🔬 Authors & Contact Information

| | Author | Affiliation |
|---|---|---|
| 👩‍💻 | **Anika Rahman** (anika.rahman@stamforduniversity.edu.bd) | Dept. of CSE, Stamford University Bangladesh, Dhaka, Bangladesh |
| 👨‍💻 | **Fahmid Al Farid** | Centre for Image and Vision Computing (CIVC), Centre of Excellence for AI, Faculty of AI and Engineering (FAIE), Multimedia University, Cyberjaya, Selangor, Malaysia |
| 👨‍💻 | **Md Humaion Kabir Mehedi** (humaion.kabir.mehedi@g.bracu.ac.bd) | Dept. of CSE, BRAC University, Dhaka, Bangladesh |
| 👨‍💻 | **Jia Uddin** (jia.uddin@wsu.ac.kr) | AI and Big Data Department, Endicott College, Woosong University, Daejeon, South Korea |
| ⭐ | **Hezerul Abdul Karim** *(Corresponding Author)* (hezerul@mmu.edu.my) | CIVC, Centre of Excellence for AI, FAIE, Multimedia University, Cyberjaya, Selangor, Malaysia |

---

## 🌍 Research Motivation

Depression affects **280 million** people globally and is a leading cause of disability.

**Current limitations in computational approaches:**

- ❌ Most studies use **binary classification** (depressed vs. non-depressed)
- ❌ Miss the **ordinal severity structure** of clinical instruments
- ❌ **Black-box models** lack clinical interpretability
- ❌ Severe **class imbalance** in real-world psychometric data
- ❌ Limited **multi-class severity stratification** research

👉 We introduce **DepressNet** — a stacking ensemble framework that addresses all of these challenges with built-in explainability.

---

## 🚀 Key Contributions

### 1️⃣ Stacking Ensemble Architecture
- Four complementary base learners: **XGBoost**, **Random Forest**, **LightGBM**, **Calibrated SVM**
- PyTorch-based **MLP meta-learner** trained on **20-dimensional out-of-fold** probability features
- Prevents information leakage via 5-fold stratified OOF protocol

### 2️⃣ GPU-Accelerated Training Pipeline
- CUDA-compatible base learners + PyTorch MLP
- Cosine annealing scheduling, class-weighted loss, best-model checkpointing
- SMOTETomek resampling for severe class imbalance

### 3️⃣ Dual Explainability Module (SHAP)
- **Global** feature attribution across all severity classes
- **Class-wise** SHAP decomposition revealing symptom progression pathways
- First study to apply class-wise SHAP across all five DASS-42 severity levels

### 4️⃣ Comprehensive Statistical Validation
- Friedman test (p < 0.001)
- Nemenyi post-hoc test
- Wilcoxon signed-rank test
- McNemar test

---

## 🏗 Architecture Overview

```
┌─────────────────────────────────────────────────┐
│         DASS-42 Raw Input Features              │
│   42 items + 11 demographic = 53 features       │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│       Preprocessing + Feature Selection         │
│   Z-score · ANOVA F-test (top 20) · 80/20 split│
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│       SMOTETomek (training set only)            │
│   31,820 → 108,535 samples · 21,707/class       │
└──────────────────────┬──────────────────────────┘
                       ▼
┌───────────┬───────────┬───────────┬─────────────┐
│  XGBoost  │ Random    │ LightGBM  │ Calibrated  │
│  (GPU)    │ Forest    │ (GPU)     │ SVM (CPU)   │
│  300 est. │ (CPU)     │ 300 est.  │ LinearSVC   │
│  depth=6  │ 300 est.  │ leaves=63 │ C=1.0       │
└─────┬─────┴─────┬─────┴─────┬─────┴──────┬──────┘
      └───────────┴───────────┴────────────┘
                       ▼
        5-fold Stratified CV → OOF predictions
                       ▼
┌─────────────────────────────────────────────────┐
│         Meta-feature matrix [concat]            │
│     4 base learners × 5 classes = 20-dim        │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│     PyTorch MLP Meta-Learner (Tesla T4 GPU)     │
│   Dense 256 → ReLU → BN → Dropout(0.4)         │
│   Dense 128 → ReLU → BN → Dropout(0.3)         │
│   Dense 5 → Softmax · Class-weighted CE Loss    │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│      Predicted Depression Severity Class        │
│  Normal · Mild · Moderate · Severe · Ext.Severe │
└─────────────────────────────────────────────────┘
```

---

## 📊 Dataset

### DASS-42 OpenPsychometrics Dataset

| Class | Severity Level | Scaled Score Range | Count (N) | Proportion |
|-------|---------------|-------------------|-----------|------------|
| 0 | Normal | 0–9 | 4,318 | 10.9% |
| 1 | Mild | 10–13 | 1,755 | 4.4% |
| 2 | Moderate | 14–20 | 3,698 | 9.3% |
| 3 | Severe | 21–27 | 2,871 | 7.2% |
| 4 | Extremely Severe | ≥ 28 | 27,133 | 68.2% |
| | **Total** | | **39,775** | **100%** |

**Dataset:** [OpenPsychometrics DASS-42](https://openpsychometrics.org/_rawdata/DASS_data_21.02.19.zip)

**Split:** 80% Train (31,820) / 20% Test (7,955) — test set unbalanced to reflect real-world distribution.

---

## 🏆 Performance Results

### Overall Performance (Held-out Test Set, N = 7,955)

| Model | Test Accuracy | Macro F1 | MCC | AUC-ROC |
|-------|:---:|:---:|:---:|:---:|
| Logistic Regression | 83.51% | 53.24% | 0.6818 | 0.9687 |
| Decision Tree | 78.30% | 53.00% | 0.5935 | 0.9261 |
| Random Forest | 84.15% | 57.71% | 0.6917 | 0.9558 |
| XGBoost | 83.39% | 56.53% | 0.6822 | 0.9662 |
| LightGBM | 83.19% | 54.75% | 0.6753 | 0.9646 |
| MLP (standalone) | 85.02% | 58.41% | 0.7095 | 0.9727 |
| 🌟 **DepressNet** | **85.38%** | **59.01%** | **0.7158** | 0.9714 |

### Per-Class Performance (DepressNet)

| Severity Class | Precision | Recall | F1-Score | Support |
|---------------|:---------:|:------:|:--------:|:-------:|
| Normal | 1.00 | 0.54 | 0.70 | 864 |
| Mild | 0.10 | 0.10 | 0.10 | 351 |
| Moderate | 0.45 | 0.46 | 0.45 | 740 |
| Severe | 0.56 | 0.93 | 0.93 | 574 |
| Extremely Severe | 1.00 | 1.00 | 1.00 | 5,426 |

---

## 🔎 Explainable AI

### Global SHAP Top Features

| Rank | Feature | DASS-42 Item | Mean SHAP |
|:----:|---------|-------------|:-------:|
| 1 | Q13A | I felt sad and depressed | 0.4426 |
| 2 | Q17A | I felt I wasn't worth much as a person | 0.3982 |
| 3 | Q34A | I felt I was pretty worthless | 0.3657 |
| 4 | Q26A | I felt down-hearted and blue | 0.3479 |
| 5 | Q16A | I felt that I had lost interest in everything | 0.3476 |

### Class-wise Symptom Progression

| Severity | Top Predictors | Dominant Symptom Domain |
|----------|---------------|----------------------|
| Normal | Q26A, Q13A, Q5A | Transient mood fluctuation |
| Mild | Q21A, Q17A, Q5A | Emerging cognitive distortion |
| Moderate | Q34A, Q13A, Q16A | Self-schema disruption; anhedonia |
| Severe | Q17A, Q38A, Q34A | Entrenched negative self-belief |
| Ext. Severe | Q13A, Q10A, Q42A | Existential collapse |

---

## ⚙ Training Configuration

| Parameter | Value |
|-----------|-------|
| Meta-Learner | PyTorch MLP (256→128→5) |
| Optimizer | Adam (lr=0.001, weight_decay=1e-4) |
| Scheduler | Cosine Annealing (80 epochs) |
| Dropout | 0.4 / 0.3 |
| Batch Size | 512 |
| Loss | Class-weighted CrossEntropyLoss |
| Resampling | SMOTETomek (training only) |
| GPU | Tesla T4 (CUDA) |
| Cross-Validation | 5-fold Stratified |

---

## 📁 Repository Structure

```
DepressNet_Frontiers-in-Computer-Science_2026/
├── README.md
├── DepressNet_Final_Submitted.ipynb
├── figures/
│   ├── Figure1_Architecture.png
│   ├── Figure2_SHAP_Pipeline.png
│   └── Figure5_Ablation.png
└── LICENSE
```

---

## 🔧 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/rahman-anika/DepressNet_Frontiers-in-Computer-Science_2026.git
cd DepressNet_Frontiers-in-Computer-Science_2026
```

### 2. Download the dataset
```bash
wget https://openpsychometrics.org/_rawdata/DASS_data_21.02.19.zip
unzip DASS_data_21.02.19.zip
```

### 3. Run in Google Colab
- Upload `DepressNet_Final_Submitted.ipynb` to [Google Colab](https://colab.research.google.com/)
- Enable **GPU runtime** (Runtime → Change runtime type → T4 GPU)
- Run all cells sequentially

### Requirements
```
torch >= 2.0
scikit-learn >= 1.3
xgboost >= 2.0
lightgbm >= 4.0
shap >= 0.43
imbalanced-learn >= 0.11
matplotlib >= 3.7
numpy >= 1.24
pandas >= 2.0
```

---

## ⚠ Known Limitations

1. **Mild class difficulty (F1 = 0.10)** — Narrow 4-point severity band with substantial symptom overlap with Normal and Moderate classes. Structural, not model-related.

2. **Circular structural dependency** — Severity labels are derived from the same DASS-42 items used as input features. DepressNet learns scoring patterns, not independent clinical ground truth.

3. **No external clinical validation** — Dataset consists of voluntary online self-reports without clinician assessment. Validation on hospital-based cohorts is necessary before deployment.

---

### 🧠 Advancing Interpretable AI for Mental Health Research 🧠
