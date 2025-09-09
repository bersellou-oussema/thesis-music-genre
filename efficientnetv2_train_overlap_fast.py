import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import LinearSVC

# =========================
# Config
# =========================
IN_DIR = "models"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

# Choose one: "merged", "gtzan_to_fma", "fma_to_gtzan"
MODE = "merged"

# RFF dimensionality (increase if you have more RAM/CPU; 2000 is a safe start)
RFF_COMPONENTS = 2000
LINEAR_SVC_C = 1.0
LINEAR_SVC_MAXIT = 20000
RANDOM_STATE = 42
TEST_SIZE = 0.20

# =========================
# Helpers
# =========================
def load_features(pkl_name):
    p = os.path.join(IN_DIR, pkl_name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing features file: {p}")
    print(f"📄 Loading: {p}")
    X, y, paths = joblib.load(p)
    print(f"   -> X: {X.shape}, labels: {len(y)}")
    return X.astype(np.float32), np.asarray(y), np.asarray(paths)

def restrict_to_labels(X, y, labels_keep):
    mask = np.isin(y, labels_keep)
    return X[mask], y[mask]

def print_label_counts(tag, y):
    cnt = Counter(y)
    print(f"\n🔢 Class counts [{tag}]:")
    for k in sorted(cnt.keys()):
        print(f"  {k:15s}: {cnt[k]}")

def save_confusion(y_true, y_pred, labels, out_png, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def summarize_report(y_true, y_pred, labels, tag):
    rep = classification_report(y_true, y_pred, target_names=labels, digits=4, output_dict=True)
    acc  = rep["accuracy"]
    prec = rep["weighted avg"]["precision"]
    rec  = rep["weighted avg"]["recall"]
    f1   = rep["weighted avg"]["f1-score"]
    print(f"\n✅ {tag} — Accuracy: {acc:.4f} | Weighted Precision: {prec:.4f} | "
          f"Weighted Recall: {rec:.4f} | Weighted F1: {f1:.4f}")
    return rep

# =========================
# Pipelines
# =========================
def build_pipeline_and_fit(Xtr, ytr):
    # Scale
    scaler = StandardScaler(with_mean=True)
    Xtr_s = scaler.fit_transform(Xtr)

    # RFF mapping (RBF kernel approximation)
    rff = RBFSampler(gamma="scale", n_components=RFF_COMPONENTS, random_state=RANDOM_STATE)
    Ztr = rff.fit_transform(Xtr_s)

    # Linear SVM
    clf = LinearSVC(C=LINEAR_SVC_C, max_iter=LINEAR_SVC_MAXIT, dual=True, random_state=RANDOM_STATE)
    clf.fit(Ztr, ytr)
    return scaler, rff, clf

def pipeline_predict(scaler, rff, clf, X):
    Xs = scaler.transform(X)
    Z  = rff.transform(Xs)
    return clf.predict(Z)

# =========================
# Main
# =========================
if __name__ == "__main__":
    # Load both feature sets
    Xg, yg, _ = load_features("features_effnet_gtzan.pkl")
    Xf, yf, _ = load_features("features_effnet_fma.pkl")

    # Find overlapping labels
    overlap = sorted(list(set(yg).intersection(set(yf))))
    if not overlap:
        raise SystemExit("No overlapping genres between GTZAN and FMA.")
    print(f"\n🔗 Overlapping genres ({len(overlap)}): {overlap}")

    # Restrict both datasets to overlap only
    Xg_o, yg_o = restrict_to_labels(Xg, yg, overlap)
    Xf_o, yf_o = restrict_to_labels(Xf, yf, overlap)

    print_label_counts("GTZAN overlap", yg_o)
    print_label_counts("FMA overlap",   yf_o)

    if MODE == "merged":
        # Merge datasets then stratified split
        X_all = np.vstack([Xg_o, Xf_o])
        y_all = np.concatenate([yg_o, yf_o])
        print_label_counts("Merged overlap", y_all)

        Xtr, Xte, ytr, yte = train_test_split(
            X_all, y_all, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE
        )

        scaler, rff, clf = build_pipeline_and_fit(Xtr, ytr)
        yp = pipeline_predict(scaler, rff, clf, Xte)

        print("\n🎯 Classification Report (Merged overlap):")
        labels = overlap
        print(classification_report(yte, yp, target_names=labels, digits=4))
        rep = summarize_report(yte, yp, labels, "Merged overlap (GTZAN+FMA)")

        # Save model & CM
        out_model = os.path.join(OUT_DIR, "svm_overlap_rff_linear_merged.pkl")
        joblib.dump((clf, rff, scaler, labels), out_model)
        save_confusion(yte, yp, labels, os.path.join(OUT_DIR, "cm_overlap_merged.png"),
                       title="Confusion Matrix (Overlap: Merged GTZAN+FMA)")

        # Save a txt report
        with open(os.path.join(OUT_DIR, "report_overlap_merged.txt"), "w", encoding="utf-8") as f:
            from sklearn.metrics import classification_report as cr_text
            f.write("Classification Report (Merged overlap)\n\n")
            f.write(cr_text(yte, yp, target_names=labels, digits=4))
        print("💾 Saved: models/svm_overlap_rff_linear_merged.pkl, cm_overlap_merged.png, report_overlap_merged.txt")

    elif MODE == "gtzan_to_fma":
        # Train on GTZAN overlap, test on FMA overlap
        print("\n🚀 Training on GTZAN (overlap), testing on FMA (overlap) ...")
        scaler, rff, clf = build_pipeline_and_fit(Xg_o, yg_o)
        yp = pipeline_predict(scaler, rff, clf, Xf_o)

        labels = overlap
        print("\n🎯 Classification Report (GTZAN → FMA overlap):")
        print(classification_report(yf_o, yp, target_names=labels, digits=4))
        rep = summarize_report(yf_o, yp, labels, "GTZAN → FMA (overlap)")

        out_model = os.path.join(OUT_DIR, "svm_overlap_rff_linear_gtzan_to_fma.pkl")
        joblib.dump((clf, rff, scaler, labels), out_model)
        save_confusion(yf_o, yp, labels, os.path.join(OUT_DIR, "cm_overlap_gtzan_to_fma.png"),
                       title="Confusion Matrix (GTZAN → FMA, Overlap)")
        with open(os.path.join(OUT_DIR, "report_overlap_gtzan_to_fma.txt"), "w", encoding="utf-8") as f:
            from sklearn.metrics import classification_report as cr_text
            f.write("Classification Report (GTZAN → FMA overlap)\n\n")
            f.write(cr_text(yf_o, yp, target_names=labels, digits=4))
        print("💾 Saved: models/svm_overlap_rff_linear_gtzan_to_fma.pkl, cm_overlap_gtzan_to_fma.png, report_overlap_gtzan_to_fma.txt")

    elif MODE == "fma_to_gtzan":
        # Train on FMA overlap, test on GTZAN overlap
        print("\n🚀 Training on FMA (overlap), testing on GTZAN (overlap) ...")
        scaler, rff, clf = build_pipeline_and_fit(Xf_o, yf_o)
        yp = pipeline_predict(scaler, rff, clf, Xg_o)

        labels = overlap
        print("\n🎯 Classification Report (FMA → GTZAN overlap):")
        print(classification_report(yg_o, yp, target_names=labels, digits=4))
        rep = summarize_report(yg_o, yp, labels, "FMA → GTZAN (overlap)")

        out_model = os.path.join(OUT_DIR, "svm_overlap_rff_linear_fma_to_gtzan.pkl")
        joblib.dump((clf, rff, scaler, labels), out_model)
        save_confusion(yg_o, yp, labels, os.path.join(OUT_DIR, "cm_overlap_fma_to_gtzan.png"),
                       title="Confusion Matrix (FMA → GTZAN, Overlap)")
        with open(os.path.join(OUT_DIR, "report_overlap_fma_to_gtzan.txt"), "w", encoding="utf-8") as f:
            from sklearn.metrics import classification_report as cr_text
            f.write("Classification Report (FMA → GTZAN overlap)\n\n")
            f.write(cr_text(yg_o, yp, target_names=labels, digits=4))
        print("💾 Saved: models/svm_overlap_rff_linear_fma_to_gtzan.pkl, cm_overlap_fma_to_gtzan.png, report_overlap_fma_to_gtzan.txt")

    else:
        raise ValueError(f"Unknown MODE: {MODE}")
