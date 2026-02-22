# ML Diabetes Prediction (Python / scikit-learn)

End-to-end machine learning pipeline for predicting diabetes from tabular medical data.  
Includes preprocessing with pipelines, model comparison, hyperparameter tuning, and evaluation.

## Dataset
**File:** `diabetes_prediction_dataset.csv`  
**Target:** `diabetes` (binary)

**Main columns:** `gender`, `age`, `hypertension`, `heart_disease`, `smoking_history`, `bmi`, `HbA1c_level`, `blood_glucose_level`

## What’s inside
- EDA and basic visualization
- Data preprocessing using **`ColumnTransformer` + `Pipeline`**
- Model training and comparison:
  - Logistic Regression
  - KNN
  - Naive Bayes
  - Decision Tree
- Hyperparameter tuning with **`GridSearchCV`**
- Final evaluation with confusion matrix + classification report

## Results (Test Set)
**Best model:** **Decision Tree** (`max_depth=3`, `min_samples_split=2`)  
- **Accuracy:** `0.9724`  
- **Class 1 (diabetes):** precision `1.00`, recall `0.68`, F1 `0.81`  
- **Confusion Matrix:**
  - TN=27450, FP=0
  - FN=828, TP=1722

> **Note:** The dataset is imbalanced, so recall/F1 for the positive class is an important metric beyond accuracy.

## Tech Stack
- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- jupyter

## Install & Run
```bash
# 1) Clone
git clone https://github.com/EvyatarRidi/ML_Diabetes_Prediction.git
cd ML_Diabetes_Prediction

# 2) Create venv (macOS/Linux)
python -m venv .venv
source .venv/bin/activate

# 2b) Create venv (Windows PowerShell) - use instead of the two lines above
# python -m venv .venv
# .venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run the notebook
jupyter notebook ML_Diabetes_Prediction.ipynb
