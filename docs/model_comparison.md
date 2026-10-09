# JARVIS — ML Model Benchmark Report

- **Trained at:** 2026-10-09 13:40:54
- **Total Samples:** 4299
- **Feature Dimension:** 126 (Dual-Hand Normalized Landmarks)
- **Target Classes:** anil, communicate, dog, everyone, fck, hello, help, iloveyou, is, misselsa, no, on, our, people, project, signlangauge, signlanguage, thankyou, this, traslation, using, yes

## Comparison Table

| Model | CV_Accuracy_Mean(%) | CV_Accuracy_Std(%) | Test_Accuracy(%) | Precision(%) | Recall(%) | F1_Score(%) | Fit_Time(ms) | Latency(ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SVM (RBF Kernel) | 99.27 | 0.13 | 99.42 | 99.42 | 99.42 | 99.41 | 895.77 | 0.424 |
| Random Forest | 99.07 | 0.44 | 99.07 | 99.08 | 99.07 | 99.06 | 2479.93 | 12.896 |
| KNN (k=5) | 98.49 | 0.35 | 98.95 | 98.98 | 98.95 | 98.95 | 0.0 | 5.701 |
| Logistic Regression | 98.02 | 0.52 | 98.14 | 98.17 | 98.14 | 98.14 | 528.39 | 0.292 |
| Decision Tree | 93.89 | 2.05 | 95.0 | 95.58 | 95.0 | 95.13 | 297.95 | 0.251 |


## Best Model: SVM (RBF Kernel)

- **Accuracy:** 99.42%
- **Weighted F1:** 99.41%
- **Inference Latency:** 0.424 ms/frame

## Visualizations

1. `models/confusion_matrix_best.png`
2. `models/model_comparison.png`
3. `models/pca_clusters.png`
