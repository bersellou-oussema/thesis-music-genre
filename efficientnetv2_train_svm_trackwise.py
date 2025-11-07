# efficientnetv2_train_svm_trackwise.py
# Train an RBF-SVM on EfficientNetV2B0 features with track-wise split (GTZAN).
# This fixes the data leakage issue (no segment from the same track appears in both train and test).

import os, re
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

IN_DIR  = "models"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

def load_features(name: str):
    """Load (X, y, paths) from a joblib .pkl produced by feature extraction."""
    p = os.path.join(IN_DIR, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not found")
    X, y, paths = joblib.load(p)
    return np.array(X), np.array(y), np.array(paths)

def track_id_from_path(p: str) -> str:
    """
    Extract a track ID so that all segments from the same track
    are grouped together.
    Example: blues_00001_44100.png → blues_00001
    """
    b = os.path.basename(p)
    b = re.sub(r'_\d+\.(png|jpg)$', '', b)
    return b

def save_confusion(y_true, y_pred, labels, out_png, title="Confusion Matrix"):
    """Save confusion matrix with numbers inside cells."""
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

def train_eval_gtzan_trackwise():
    print("🚀 Loading features...")
    X, y, paths = load_features("features_effnet_gtzan.pkl")

    # Build groups = track IDs
    groups = np.array([track_id_from_path(p) for p in paths])

    # StratifiedGroupKFold ensures same-track segments stay together
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y, groups))

    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    # Scale features
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    # Train SVM
    clf = SVC(kernel="rbf", C=10, gamma="scale")
    clf.fit(Xtr, ytr)

    # Evaluate
    yp = clf.predict(Xte)
    labels = sorted(list(set(y)))

    print("\n🎯 Classification Report (GTZAN, track-wise):")
    print(classification_report(yte, yp, target_names=labels, digits=4))

    # Final one-line summary
    rep = classification_report(yte, yp, target_names=labels, digits=4, output_dict=True)
    acc  = rep["accuracy"]
    prec = rep["weighted avg"]["precision"]
    rec  = rep["weighted avg"]["recall"]
    f1   = rep["weighted avg"]["f1-score"]
    print(f"\n✅ GTZAN (Track-wise) — Accuracy: {acc:.4f} | "
          f"Weighted Precision: {prec:.4f} | Weighted Recall: {rec:.4f} | Weighted F1: {f1:.4f}")

    # Save artifacts
    joblib.dump((clf, scaler, labels), os.path.join(OUT_DIR, "svm_gtzan_trackwise.pkl"))
    save_confusion(yte, yp, labels, os.path.join(OUT_DIR, "cm_gtzan_trackwise.png"),
                   title="Confusion Matrix (GTZAN, Track-wise)")
    print("💾 Saved: models/svm_gtzan_trackwise.pkl  and  models/cm_gtzan_trackwise.png")

if __name__ == "__main__":
    train_eval_gtzan_trackwise()
