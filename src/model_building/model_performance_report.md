# Model Performance Report

*Report generated on: 2025-12-29 15:01:11
*MLflow Experiment: 'Heart Disease Prediction'

## Logistic Regression

- **MLflow Run ID**: `539b75ba876248f6a1de9be347749e66`
- **Best Params**: `{'C': 0.1, 'solver': 'liblinear'}`
- **CV Accuracy (Mean)**: 0.8345
- **Test Accuracy**: 0.8852
- **Precision**: 0.8387
- **Recall**: 0.9286
- **F1-Score**: 0.8814
- **ROC-AUC**: 0.9621

### Classification Report

```
              precision    recall  f1-score   support

           0       0.93      0.85      0.89        33
           1       0.84      0.93      0.88        28

    accuracy                           0.89        61
   macro avg       0.89      0.89      0.89        61
weighted avg       0.89      0.89      0.89        61

```

## Random Forest

- **MLflow Run ID**: `2b434a4e366a46fcab1e9b48829d4204`
- **Best Params**: `{'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 100}`
- **CV Accuracy (Mean)**: 0.8136
- **Test Accuracy**: 0.8852
- **Precision**: 0.8387
- **Recall**: 0.9286
- **F1-Score**: 0.8814
- **ROC-AUC**: 0.9394

### Classification Report

```
              precision    recall  f1-score   support

           0       0.93      0.85      0.89        33
           1       0.84      0.93      0.88        28

    accuracy                           0.89        61
   macro avg       0.89      0.89      0.89        61
weighted avg       0.89      0.89      0.89        61

```

