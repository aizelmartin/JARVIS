"""
train_models.py
---------------
Trains, evaluates, and compares multiple ML classifiers on extracted hand landmarks.

Academic syllabus coverage:
  - KNN (K-Nearest Neighbors)
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - Logistic Regression (Linear baseline)
  - K-Means Clustering & PCA analysis

Outputs:
  - models/saved_models/sign_classifier.pkl (best model + label encoder)
  - models/evaluation_metrics.csv
  - models/model_comparison.png
  - models/confusion_matrix_best.png
  - docs/model_comparison.md
"""

import os
import sys
import time
import argparse
import joblib
import warnings
warnings.filterwarnings("ignore")

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    silhouette_score,
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# ── Paths ─────────────────────────────────────────────────────────────── #
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_CUSTOM   = os.path.join(PROJECT_ROOT, "data", "processed", "landmarks_custom.csv")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
SAVED_DIR     = os.path.join(MODELS_DIR, "saved_models")
DOCS_DIR      = os.path.join(PROJECT_ROOT, "docs")


def load_dataset(csv_path: str):
    """Loads landmark CSV and separates features (X) and labels (y)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values
    return X, y, feature_cols


def run_unsupervised_analysis(X, y, class_names):
    """Performs K-Means clustering and PCA 2D visualization."""
    print("\n" + "=" * 60)
    print("[>] ACADEMIC COMPONENT: UNSUPERVISED ANALYSIS (K-Means & PCA)")
    print("=" * 60)

    n_clusters = len(class_names)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    sil_score = silhouette_score(X, cluster_labels)
    print(f"[*] K-Means ({n_clusters} clusters) Silhouette Score: {sil_score:.4f}")

    # PCA 2D projection
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    explained_var = np.sum(pca.explained_variance_ratio_) * 100
    print(f"[*] PCA 2-Components Explained Variance: {explained_var:.2f}%")

    # Save PCA scatter plot
    os.makedirs(MODELS_DIR, exist_ok=True)
    plt.figure(figsize=(9, 7))
    unique_labels = np.unique(y)
    for lbl in unique_labels:
        mask = (y == lbl)
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=str(lbl), alpha=0.7, edgecolors="none", s=35)

    plt.title(f"PCA 2D Projection of Hand Landmarks (Var Explained: {explained_var:.1f}%)", fontsize=13, pad=12)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="True Gesture", bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    pca_plot_path = os.path.join(MODELS_DIR, "pca_clusters.png")
    plt.savefig(pca_plot_path, dpi=200)
    plt.close()
    print(f"[+] Saved PCA cluster visualization → {pca_plot_path}")


def train_and_evaluate(csv_path: str):
    print("=" * 60)
    print("      JARVIS ML MODEL TRAINING & COMPARISON SUITE      ")
    print("=" * 60)

    X, y_raw, feature_cols = load_dataset(csv_path)
    print(f"[Dataset] Loaded {X.shape[0]} samples with {X.shape[1]} features")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_
    print(f"[Classes] {len(class_names)} unique signs: {list(class_names)}")

    # Unsupervised analysis
    run_unsupervised_analysis(X, y_raw, class_names)

    # Train / Test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[Split] Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # Define model dictionary for academic comparison
    models = {
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, metric="cosine"),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
        "SVM (RBF Kernel)": SVC(kernel="rbf", C=10.0, probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    results = []
    trained_models = {}
    cv_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "=" * 60)
    print("[>] TRAINING & 5-FOLD CROSS-VALIDATION EVALUATION")
    print("=" * 60)

    for name, clf in models.items():
        print(f"\n[*] Training {name}...")

        # 5-fold cross validation
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv_kfold, scoring="accuracy")

        # Fit model on training set
        t_start = time.time()
        clf.fit(X_train, y_train)
        fit_time_ms = (time.time() - t_start) * 1000

        # Inference latency (1000 single predictions)
        sample = X_test[:1]
        t0 = time.time()
        for _ in range(500):
            _ = clf.predict(sample)
        latency_ms = ((time.time() - t0) / 500) * 1000

        # Predictions on test set
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0) * 100
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0) * 100
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0) * 100
        cv_mean = cv_scores.mean() * 100
        cv_std = cv_scores.std() * 100

        print(f"    - 5-Fold CV Accuracy: {cv_mean:.2f}% (+/- {cv_std:.2f}%)")
        print(f"    - Test Accuracy:     {acc:.2f}%")
        print(f"    - Weighted F1-Score: {f1:.2f}%")
        print(f"    - Inference Latency: {latency_ms:.3f} ms/frame")

        results.append({
            "Model": name,
            "CV_Accuracy_Mean(%)": round(cv_mean, 2),
            "CV_Accuracy_Std(%)": round(cv_std, 2),
            "Test_Accuracy(%)": round(acc, 2),
            "Precision(%)": round(prec, 2),
            "Recall(%)": round(rec, 2),
            "F1_Score(%)": round(f1, 2),
            "Fit_Time(ms)": round(fit_time_ms, 2),
            "Latency(ms)": round(latency_ms, 3),
        })

        trained_models[name] = {
            "model": clf,
            "acc": acc,
            "f1": f1,
            "y_pred": y_pred,
        }

    results_df = pd.DataFrame(results).sort_values(by="F1_Score(%)", ascending=False)

    print("\n" + "=" * 60)
    print("[>] MODEL BENCHMARK SUMMARY TABLE")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Save metrics to CSV
    metrics_path = os.path.join(MODELS_DIR, "evaluation_metrics.csv")
    results_df.to_csv(metrics_path, index=False)
    print(f"\n[+] Saved metrics CSV -> {metrics_path}")

    # Identify best model
    best_model_name = results_df.iloc[0]["Model"]
    best_entry = trained_models[best_model_name]
    best_clf = best_entry["model"]
    print(f"\n[BEST MODEL] Best Performing Model: {best_model_name} (F1: {best_entry['f1']:.2f}%)")

    # Detailed Classification Report of the best model
    print(f"\n[Classification Report: {best_model_name}]")
    print(classification_report(y_test, best_entry["y_pred"], target_names=class_names))

    # ── Confusion Matrix Plot ────────────────────────────────────────── #
    cm = confusion_matrix(y_test, best_entry["y_pred"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix — {best_model_name}", fontsize=13, pad=12)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_plot_path = os.path.join(MODELS_DIR, "confusion_matrix_best.png")
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    print(f"[+] Saved Confusion Matrix → {cm_plot_path}")

    # ── Model Comparison Bar Chart ────────────────────────────────────── #
    plt.figure(figsize=(10, 5))
    bar_width = 0.35
    indices = np.arange(len(results_df))
    plt.bar(indices - bar_width/2, results_df["Test_Accuracy(%)"], bar_width, label="Test Accuracy (%)", color="#2b5c8f")
    plt.bar(indices + bar_width/2, results_df["F1_Score(%)"], bar_width, label="F1-Score (%)", color="#34a853")
    plt.xticks(indices, results_df["Model"], rotation=15)
    plt.ylabel("Score (%)")
    plt.ylim(max(0, results_df["Test_Accuracy(%)"].min() - 10), 105)
    plt.title("Sign Language Classifier Comparison", fontsize=13, pad=12)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    bar_plot_path = os.path.join(MODELS_DIR, "model_comparison.png")
    plt.savefig(bar_plot_path, dpi=200)
    plt.close()
    print(f"[+] Saved Model Comparison Chart → {bar_plot_path}")

    # ── Save Best Model Artifact ─────────────────────────────────────── #
    os.makedirs(SAVED_DIR, exist_ok=True)
    saved_model_path = os.path.join(SAVED_DIR, "sign_classifier.pkl")
    model_payload = {
        "model": best_clf,
        "model_name": best_model_name,
        "label_encoder": le,
        "classes": list(class_names),
        "num_features": X.shape[1],
        "feature_names": feature_cols,
        "metrics": results_df.to_dict(orient="records"),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    joblib.dump(model_payload, saved_model_path)
    print(f"[+] Saved Best Model Payload → {saved_model_path}")

    # ── Write Markdown Report for College Viva / Documentation ────────── #
    os.makedirs(DOCS_DIR, exist_ok=True)
    report_md_path = os.path.join(DOCS_DIR, "model_comparison.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# JARVIS — ML Model Benchmark Report\n\n")
        f.write(f"- **Trained at:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Total Samples:** {X.shape[0]}\n")
        f.write(f"- **Feature Dimension:** {X.shape[1]} (Dual-Hand Normalized Landmarks)\n")
        f.write(f"- **Target Classes:** {', '.join(class_names)}\n\n")
        f.write("## Comparison Table\n\n")
        # Format markdown table manually to avoid tabulate dependency
        headers = list(results_df.columns)
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for _, row in results_df.iterrows():
            f.write("| " + " | ".join(str(val) for val in row.values) + " |\n")
        f.write("\n\n")
        f.write(f"## Best Model: {best_model_name}\n\n")
        f.write(f"- **Accuracy:** {best_entry['acc']:.2f}%\n")
        f.write(f"- **Weighted F1:** {best_entry['f1']:.2f}%\n")
        f.write(f"- **Inference Latency:** {results_df[results_df['Model'] == best_model_name]['Latency(ms)'].values[0]} ms/frame\n\n")
        f.write("## Visualizations\n\n")
        f.write("1. `models/confusion_matrix_best.png`\n")
        f.write("2. `models/model_comparison.png`\n")
        f.write("3. `models/pca_clusters.png`\n")

    print(f"[+] Saved Documentation Report -> {report_md_path}")
    print("\n" + "=" * 60)
    print("[SUCCESS] PHASE 4 TRAINING & BENCHMARK COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and benchmark ML models for JARVIS")
    parser.add_argument("--data", default=DATA_CUSTOM, help="Path to landmark CSV file")
    args = parser.parse_args()

    train_and_evaluate(args.data)
