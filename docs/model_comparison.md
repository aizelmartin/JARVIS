# JARVIS — ML Model Benchmark Report

- **Trained at:** 2026-10-10 16:02:02
- **Total Samples:** 4499
- **Feature Dimension:** 126 (Dual-Hand Normalized Landmarks)
- **Target Classes:** anil, cat, communicate, dog, everyone, fck, hello, help, iloveyou, is, misselsa, no, on, our, people, project, signlangauge, signlanguage, thankyou, this, traslation, using, yes

## Comparison Table

| Model | CV_Accuracy_Mean(%) | CV_Accuracy_Std(%) | Test_Accuracy(%) | Precision(%) | Recall(%) | F1_Score(%) | Fit_Time(ms) | Latency(ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SVM (RBF Kernel) | 98.97 | 0.19 | 99.22 | 99.24 | 99.22 | 99.22 | 1191.53 | 0.64 |
| Random Forest | 98.83 | 0.68 | 99.11 | 99.13 | 99.11 | 99.11 | 2915.75 | 16.659 |
| KNN (k=5) | 98.53 | 0.26 | 98.56 | 98.6 | 98.56 | 98.55 | 1.0 | 6.954 |
| Logistic Regression | 97.78 | 0.17 | 98.0 | 98.04 | 98.0 | 98.0 | 730.24 | 0.298 |
| Decision Tree | 90.44 | 4.49 | 90.22 | 93.03 | 90.22 | 89.24 | 406.91 | 0.289 |


## Best Model: SVM (RBF Kernel)

- **Accuracy:** 99.22%
- **Weighted F1:** 99.22%
- **Inference Latency:** 0.64 ms/frame

## Visualizations

1. `models/confusion_matrix_best.png`
2. `models/model_comparison.png`
3. `models/pca_clusters.png`
