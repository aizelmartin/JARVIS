# JARVIS — ML Model Benchmark Report

- **Trained at:** 2026-09-27 10:52:12
- **Total Samples:** 1000
- **Feature Dimension:** 126 (Dual-Hand Normalized Landmarks)
- **Target Classes:** hello, help, no, thankyou, yes

## Comparison Table

| Model | CV_Accuracy_Mean(%) | CV_Accuracy_Std(%) | Test_Accuracy(%) | Precision(%) | Recall(%) | F1_Score(%) | Fit_Time(ms) | Latency(ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KNN (k=5) | 98.75 | 1.19 | 99.0 | 99.02 | 99.0 | 99.0 | 1.0 | 3.44 |
| Random Forest | 98.75 | 0.4 | 98.5 | 98.6 | 98.5 | 98.51 | 348.55 | 13.072 |
| SVM (RBF Kernel) | 98.62 | 0.83 | 98.5 | 98.6 | 98.5 | 98.51 | 67.0 | 0.296 |
| Logistic Regression | 97.75 | 1.02 | 97.5 | 97.56 | 97.5 | 97.51 | 59.13 | 0.268 |
| Decision Tree | 96.62 | 1.29 | 96.0 | 96.09 | 96.0 | 96.0 | 22.0 | 0.22 |


## Best Model: KNN (k=5)

- **Accuracy:** 99.00%
- **Weighted F1:** 99.00%
- **Inference Latency:** 3.44 ms/frame

## Visualizations

1. `models/confusion_matrix_best.png`
2. `models/model_comparison.png`
3. `models/pca_clusters.png`
