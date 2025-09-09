# efficientnetv2_train_svm.py
# Train an RBF-SVM on EfficientNetV2B0 features.
# FMA ONLY is active; GTZAN and overlap blocks are kept but commented.

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

IN_DIR  = "models"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

def load_features(name: str):
    """Load (X, y, paths) from a joblib .pkl produced by feature extraction."""
    p = os.path.join(IN_DIR, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not found")
    X, y, paths = joblib.load(p)
    return X, y, paths

def save_confusion(y_true, y_pred, labels, out_png, title="Confusion Matrix"):
    """Save a confusion matrix with numbers inside cells."""
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

# =========================
# FMA (ACTIVE)
# =========================
def train_eval_fma():
    X, y, _ = load_features("features_effnet_fma.pkl")

    # Scale features (important for SVM)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Stratified split
    Xtr, Xte, ytr, yte = train_test_split(
        Xs, y, test_size=0.20, stratify=y, random_state=42
    )

    # Train SVM (tune C if needed)
    clf = SVC(kernel="rbf", C=10, gamma="scale")
    clf.fit(Xtr, ytr)

    # Evaluate
    yp = clf.predict(Xte)
    labels = sorted(list(set(y)))

    print("\n🎯 Classification Report (FMA):")
    print(classification_report(yte, yp, target_names=labels, digits=4))

    # One-line summary
    rep = classification_report(yte, yp, target_names=labels, digits=4, output_dict=True)
    acc  = rep["accuracy"]
    prec = rep["weighted avg"]["precision"]
    rec  = rep["weighted avg"]["recall"]
    f1   = rep["weighted avg"]["f1-score"]
    print(f"\n✅ FMA — Accuracy: {acc:.4f} | Weighted Precision: {prec:.4f} | "
          f"Weighted Recall: {rec:.4f} | Weighted F1: {f1:.4f}")

    # Save artifacts
    joblib.dump((clf, scaler, labels), os.path.join(OUT_DIR, "svm_fma.pkl"))
    save_confusion(yte, yp, labels, os.path.join(OUT_DIR, "cm_fma.png"),
                   title="Confusion Matrix (FMA)")
    print("💾 Saved: models/svm_fma.pkl  and  models/cm_fma.png")

if __name__ == "__main__":
    # ---- FMA (active) ----
    train_eval_fma()

    # ---- GTZAN (commented) ----
    # def train_eval_gtzan():
    #     X, y, _ = load_features("features_effnet_gtzan.pkl")
    #     scaler = StandardScaler()
    #     Xs = scaler.fit_transform(X)
    #     Xtr, Xte, ytr, yte = train_test_split(
    #         Xs, y, test_size=0.20, stratify=y, random_state=42
    #     )
    #     clf = SVC(kernel="rbf", C=10, gamma="scale").fit(Xtr, ytr)
    #     yp = clf.predict(Xte)
    #     labels = sorted(list(set(y)))
    #     print("\n🎯 Classification Report (GTZAN):")
    #     print(classification_report(yte, yp, target_names=labels, digits=4))
    #     joblib.dump((clf, scaler, labels), os.path.join(OUT_DIR, "svm_gtzan.pkl"))
    #     save_confusion(yte, yp, labels, os.path.join(OUT_DIR, "cm_gtzan.png"),
    #                    title="Confusion Matrix (GTZAN)")
    #     print("💾 Saved: models/svm_gtzan.pkl  and  models/cm_gtzan.png")
    #
    # # To run later:
    # # train_eval_gtzan()

    # ---- Overlap model (commented) ----
    # def train_eval_overlap():
    #     Xg, yg, _ = load_features("features_effnet_gtzan.pkl")
    #     Xf, yf, _ = load_features("features_effnet_fma.pkl")
    #     common = sorted(list(set(yg).intersection(set(yf))))
    #     if not common:
    #         print("ℹ️ No overlapping genres; skipping overlap model.")
    #         return
    #     mask_g = np.isin(yg, common)
    #     mask_f = np.isin(yf, common)
    #     Xc = np.vstack([Xg[mask_g], Xf[mask_f]])
    #     yc = np.concatenate([yg[mask_g], yf[mask_f]])
    #     scaler = StandardScaler()
    #     Xs = scaler.fit_transform(Xc)
    #     Xtr, Xte, ytr, yte = train_test_split(
    #         Xs, yc, test_size=0.20, stratify=yc, random_state=42
    #     )
    #     clf = SVC(kernel="rbf", C=10, gamma="scale").fit(Xtr, ytr)
    #     yp = clf.predict(Xte)
    #      # Use 'common' for label order
    #     save_confusion(yte, yp, common, os.path.join(OUT_DIR, "cm_overlap.png"),
    #                    title="Confusion Matrix (Overlap GTZAN∩FMA)")
    #     print("💾 Saved: models/svm_overlap.pkl and models/cm_overlap.png")
    #
    # # To run later:
    # # train_eval_overlap()
