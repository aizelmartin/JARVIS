# JARVIS — ML Model Benchmark Report

- **Trained at:** 2026-09-27 14:31:44
- **Total Samples:** 3400
- **Feature Dimension:** 126 (Dual-Hand Normalized Landmarks)
- **Target Classes:** communicate, everyone, hello, help, is, no, on, our, people, project, signlangauge, signlanguage, thankyou, this, traslation, using, yes

## Comparison Table

| Model | CV_Accuracy_Mean(%) | CV_Accuracy_Std(%) | Test_Accuracy(%) | Precision(%) | Recall(%) | F1_Score(%) | Fit_Time(ms) | Latency(ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 99.04 | 0.14 | 99.26 | 99.28 | 99.26 | 99.27 | 1636.43 | 12.871 |
| SVM (RBF Kernel) | 99.23 | 0.34 | 99.12 | 99.13 | 99.12 | 99.12 | 478.97 | 0.345 |
| KNN (k=5) | 98.68 | 0.47 | 98.82 | 98.85 | 98.82 | 98.83 | 3.0 | 4.805 |
| Logistic Regression | 98.42 | 0.66 | 98.24 | 98.25 | 98.24 | 98.23 | 310.91 | 0.259 |
| Decision Tree | 97.1 | 0.81 | 97.94 | 98.03 | 97.94 | 97.94 | 218.35 | 0.237 |


## Best Model: Random Forest

- **Accuracy:** 99.26%
- **Weighted F1:** 99.27%
- **Inference Latency:** 12.871 ms/frame

## Visualizations

1. `models/confusion_matrix_best.png`
2. `models/model_comparison.png`
3. `models/pca_clusters.png`
