# ML Diabetes Prediction (Python / scikit-learn)

End-to-end machine learning pipeline for predicting diabetes from tabular medical data.
Includes preprocessing with pipelines, model comparison, hyperparameter tuning, and evaluation.

## Dataset
File: `diabetes_prediction_dataset.csv`  
Columns include: gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, diabetes (target).

## What’s inside
- EDA and basic visualization
- Data preprocessing using `ColumnTransformer` + `Pipeline`
- Model training and comparison:
  - Logistic Regression
  - KNN
  - Naive Bayes
  - Decision Tree
- Hyperparameter tuning with `GridSearchCV`
- Final evaluation with confusion matrix + classification report

## Results (Test Set)
Best model: **Decision Tree** (`max_depth=3`, `min_samples_split=2`)  
- **Accuracy:** 0.9724  
- **Class 1 (diabetes):** precision 1.00, recall 0.68, F1 0.81  
- Confusion Matrix:
  - TN=27450, FP=0
  - FN=828,  TP=1722

> Note: The dataset is imbalanced, so recall/F1 for the positive class is a key metric beyond accuracy.

## Tech Stack
- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## How to Run
1. Clone:
   ```bash
   git clone https://github.com/EvyatarRidi/ML_Diabetes_Prediction.git
   cd ML_Diabetes_Prediction

   python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
jupyter notebook ML_Diabetes_Prediction.ipynb
