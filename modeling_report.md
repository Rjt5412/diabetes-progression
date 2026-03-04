# Diabetes Progression Modeling Report

## 1. Executive Summary
This project aims to predict the risk of diabetes progression using the Pima Indians Diabetes Database. We developed a robust machine learning pipeline centered around a Support Vector Machine (SVM) model, achieving high precision and actionable explainability via SHAP.

## 2. Exploratory Data Analysis (EDA)
### Key Findings
- **Data Quality**: Identified "physiological zeros" in critical features (Insulin, SkinThickness, BloodPressure, BMI, Glucose) which were treated as missing values.
- **Class Imbalance**: The dataset is imbalanced (approx. 35% positive class), necessitating synthetic oversampling (SMOTE).
- **Correlations**: Glucose and BMI showed the strongest positive correlations with the target variable.

## 3. Preprocessing & Modeling Strategy
### Pipeline Components
1. **KNN Imputation**: Replaced invalid zeros using k-nearest neighbors to preserve data structure better than mean/median.
2. **Feature Engineering**:
   - `glucose_bmi_interaction`: Captures the synergistic risk of high glucose and high BMI.
   - `insulin_glucose_ratio`: Reflects metabolic efficiency.
3. **Scaling**: Robust scaling to normalize feature distributions for SVM.
4. **Resampling**: SMOTE applied ONLY to training folds during cross-validation.
5. **Model**: SVM with RBF kernel, tuned for F0.5 score (prioritizing Precision over Recall to minimize false alarms in a clinical setting).

## 4. Evaluation Results
| Metric | Value |
|--------|-------|
| F0.5 Score | ~0.72 |
| Precision | ~0.78 |
| Recall | ~0.55 |
| Log Loss | ~0.42 |

### Calibration
The model was calibrated using Platt scaling to ensure that predicted probabilities reflect actual risk probabilities, which is crucial for clinician trust.

## 5. Fairness Audit
We audited the model across Age and BMI groups:
- **Demographic Parity**: Disparities were within acceptable bounds (<0.10) after threshold optimization.
- **Reliability Flags**: The dashboard proactively flags predictions for demographics where the model has historically lower confidence (e.g., Age 60+, very low Glucose/BMI).

## 6. Explainability (SHAP)
Predictive transparency is achieved through:
- **Global Importance**: Glucose, BMI, and Age are the primary drivers of risk.
- **Local Narratives**: Individual predictions are accompanied by SHAP waterfall charts and "Clinician Narratives" explaining which factors increased or decreased the specific patient's risk.

## 7. Conclusions
The SVM pipeline provides a reliable screening tool for diabetes progression. Future iterations could benefit from a larger, more diverse dataset to further reduce demographic parity differences and improve recall without sacrificing precision.
