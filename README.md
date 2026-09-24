# Titanic — Machine Learning from Disaster

An end-to-end Machine Learning project on the [Kaggle Titanic](https://www.kaggle.com/c/titanic) dataset. Predicts passenger survival using structured tabular data, following a professional workflow: **EDA → Feature Engineering → Preprocessing Pipeline → Modeling → Tuning → Kaggle Submission.**

> **🏆 Final Kaggle Score:** `0.77511` (Public Leaderboard)

---

## Table of Contents

- [Objective](#objective)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Key Findings](#key-findings)
- [Modeling Strategy](#modeling-strategy)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Acknowledgments](#acknowledgments)

---

## Objective

Predict which passengers survived the Titanic disaster using features such as class, sex, age, fare, and family relations.

The primary aim is **understanding the full ML pipeline** — not maximizing leaderboard rank. Every decision (feature, model, metric) is documented to reflect what a working ML engineer does on a real tabular problem.

---

## Project Structure
titanic_ml_project/
├── data/
│ ├── raw/ # Original Kaggle CSVs (immutable)
│ └── processed/ # Cleaned + feature-engineered data
├── notebooks/ # Phase-by-phase analysis
│ ├── 01_data_loading_and_initial_analysis.ipynb
│ ├── 02_eda_part1.ipynb
│ ├── 03_feature_engineering.ipynb
│ ├── 04_preprocessing_pipeline.ipynb
│ ├── 05_baseline_models.ipynb
│ ├── 06_model_tuning.ipynb
│ └── 07_final_submission.ipynb
├── src/ # Reusable Python modules (extensible)
├── models/ # Saved trained model (rf_final.joblib)
├── reports/figures/ # Exported plots
├── submission.csv # Final Kaggle submission
├── requirements.txt
├── .gitignore
└── README.md


**Design principles:**
- **`data/raw/`** is never modified — the original source is always recoverable.
- **`data/processed/`** is regeneratable by re-running the Day 3 notebook.
- **Notebooks map 1:1 to project phases** — easy to navigate, easy to rerun a single stage.
- **`models/`** stores the final trained pipeline, ready for inference.

---

## Dataset

| File | Rows | Columns | Notes |
|------|------|---------|-------|
| `train.csv` | 891 | 12 | Includes target `Survived` |
| `test.csv` | 418 | 11 | Predictions submitted to Kaggle |
| `gender_submission.csv` | 418 | 2 | Sample submission format |

**Original features:** `PassengerId`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`, `Survived` (train only).

---

## Workflow

| Day | Focus | Notebook |
|-----|-------|----------|
| 1 | Data loading & initial audit | `01_data_loading_and_initial_analysis.ipynb` |
| 2 | Exploratory Data Analysis | `02_eda_part1.ipynb` |
| 3 | Feature engineering & missing-value handling | `03_feature_engineering.ipynb` |
| 4 | Preprocessing pipeline (`ColumnTransformer` + `Pipeline`) | `04_preprocessing_pipeline.ipynb` |
| 5 | Baseline models (LR, DT, RF) + CV | `05_baseline_models.ipynb` |
| 6 | Hyperparameter tuning + feature importance | `06_model_tuning.ipynb` |
| 7 | Final submission to Kaggle | `07_final_submission.ipynb` |

---

## Key Findings

### Data Audit
- **Missing values:** `Age` ~20%, `Cabin` ~77%, `Embarked` ~0.2%.
- **Class imbalance:** 61.6% died / 38.4% survived.

### EDA
- **`Sex`** — female 74.2% survival vs male 18.9%. Strongest single predictor.
- **`Pclass`** — 1st 63.0%, 2nd 47.3%, 3rd 24.2%. Second strongest.
- **`FamilySize`** — U-shaped: solo travelers and large families did worst; small families (2–4) best.
- **`HasCabin`** — 67% survival with a recorded cabin vs 30% without.
- **`Embarked`** — Cherbourg passengers survived more, likely because more boarded 1st class.

### Engineered Features
| Feature | Purpose |
|---------|---------|
| `Title` | Extracted from `Name` — captures gender + age + social status |
| `FamilySize` | `SibSp + Parch + 1` — combines weak features into a strong one |
| `IsAlone` | Binary — solo travelers flagged |
| `HasCabin` | Binary — replaces 77%-missing `Cabin` |

### Feature Importance (Random Forest)
Top drivers — perfectly aligned with domain knowledge:
1. `Sex_female` (~0.195)
2. `Title_Mr` (~0.149)
3. `Sex_male` (~0.148)
4. `Fare` (~0.093)
5. `Pclass` (~0.091)

---

## Modeling Strategy

**Three candidate models:**

| Model | Role |
|-------|------|
| Logistic Regression | Linear baseline — interpretable, robust |
| Decision Tree | Nonlinear baseline — prone to overfitting |
| Random Forest | Ensembled trees — robust on tabular data |

**Evaluation:**
- 80/20 stratified train/validation split
- **5-fold cross-validation** for robust model selection
- Metrics: accuracy, precision, recall, F1, confusion matrix

**Key insight:** Hyperparameter tuning **did not improve** over defaults on this small dataset — untuned Random Forest won. A real reminder that tuning is not always the answer.

---

## Results

### Cross-Validation (5-Fold, Full Training Data)

| Model | CV Mean Accuracy | Std |
|-------|------------------|-----|
| **Random Forest (untuned)** 🏆 | **0.8339** | ±0.0228 |
| Random Forest (tuned) | 0.8327 | ±0.0241 |
| Logistic Regression (untuned) | 0.8260 | ±0.0271 |
| Logistic Regression (tuned) | 0.8159 | ±0.0237 |

### Final Model

**Random Forest** — `n_estimators=100`, `max_depth=5`, `random_state=42`

### Kaggle Submission

| Metric | Value |
|--------|-------|
| CV Accuracy | 0.8339 |
| **Public Leaderboard** | **0.77511** |
| Prediction distribution | 254 died / 164 survived |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.10+ |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn |
| Model Persistence | joblib |
| Environment | Jupyter Notebooks (VS Code) |
| Version Control | Git + GitHub |

---

## Getting Started

### 1. Clone
```bash
git clone https://github.com/Anime631/titanic-ml-project.git
cd titanic-ml-project

2. Environment
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows
pip install -r requirements.txt

3. Kaggle API
Place kaggle.json in ~/.kaggle/ (Linux/macOS) or %USERPROFILE%\.kaggle\ (Windows).

4. Download Data
kaggle competitions download -c titanic -p data/raw
unzip data/raw/titanic.zip -d data/raw

5. Reproduce
Run the notebooks in order (01_ → 07_). Each notebook is self-contained.

6. Predict on New Data
import joblib
import pandas as pd

model = joblib.load("models/rf_final.joblib")
X = pd.read_csv("data/processed/test_clean.csv")
predictions = model.predict(X)

Acknowledgments
Dataset: Kaggle — Titanic: Machine Learning from Disaster

Educational context: Built as a structured 7-day ML sprint to master the end-to-end workflow.

License
Educational project. Titanic dataset © Kaggle.


**Save the file.**

---

## **🧱 Block B — Full `.gitignore`**

Replace your `.gitignore` with this comprehensive version:

```gitignore
# ============================================
# .gitignore — Titanic ML Project
# ============================================

# ---------- Data ----------
# Raw data: re-downloadable from Kaggle
# Processed data: regeneratable by running 03_feature_engineering.ipynb
data/raw/
data/processed/
data/interim/

# ---------- Models ----------
# Binary model files (large, regeneratable via 06_model_tuning.ipynb)
models/
*.pkl
*.joblib
*.h5
*.pt
*.onnx

# ---------- Notebooks ----------
.ipynb_checkpoints/
*/.ipynb_checkpoints/*

# ---------- Python ----------
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/
.eggs/

# ---------- Virtual Environments ----------
.venv/
venv/
env/
ENV/
.python-version

# ---------- Testing & Linting ----------
.pytest_cache/
.coverage
htmlcov/
.tox/
.mypy_cache/
.ruff_cache/

# ---------- Secrets ----------
kaggle.json
.env
.env.*
*.pem
*.key
secrets/

# ---------- IDE ----------
.vscode/
.idea/
*.swp
*.swo

# ---------- OS ----------
.DS_Store
.DS_Store?
._*
Thumbs.db
ehthumbs.db
Desktop.ini

# ---------- Logs & Temp ----------
*.log
logs/
tmp/
temp/