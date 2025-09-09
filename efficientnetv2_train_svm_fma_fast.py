# efficientnetv2_train_svm_fma_fast.py
import os, joblib, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import LinearSVC

IN_DIR  = "models"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

def load_features(name):
    p = os.path.join(IN_DIR, name)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    print(f"📄 Loading: {p}")
    X, y, paths = joblib.load(p)
    print(f"   -> X: {X.shape}, labels: {len(y)}")
    return X.astype(np.float32), np.array(y), paths

def save_confusion(y_true, y_pred, labels, out_png, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(out_png, bbox_inches="tight"); plt.close()

def main():
    # ----- Load FMA features -----
    X, y, _ = load_features("features_effnet_fma.pkl")

    # ----- Scale -----
    scaler = StandardScaler(with_mean=True)   # will cast to float64 internally but small
    Xs = scaler.fit_transform(X)

    # ----- Train / Test split -----
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=42)

    # ----- Random Fourier Features to approximate RBF -----
    # Increase n_components for better accuracy if you have RAM (e.g., 4000)
    rbf = RBFSampler(gamma="scale", n_components=2000, random_state=42)
    Ztr = rbf.fit_transform(Xtr)
    Zte = rbf.transform(Xte)

    # ----- Linear SVM on transformed features (fast, memory-friendly) -----
    clf = LinearSVC(C=1.0, max_iter=20000, dual=True)   # try C in [0.5, 1, 2]
    clf.fit(Ztr, ytr)
    yp = clf.predict(Zte)

    labels = sorted(list(set(y)))
    print("\n🎯 Classification Report (FMA, RFF + LinearSVC):")
    print(classification_report(yte, yp, target_names=labels, digits=4))

    rep = classification_report(yte, yp, target_names=labels, digits=4, output_dict=True)
    acc  = rep["accuracy"]
    prec = rep["weighted avg"]["precision"]
    rec  = rep["weighted avg"]["recall"]
    f1   = rep["weighted avg"]["f1-score"]
    print(f"\n✅ FMA — Accuracy: {acc:.4f} | Weighted Precision: {prec:.4f} | "
          f"Weighted Recall: {rec:.4f} | Weighted F1: {f1:.4f}")

    joblib.dump((clf, rbf, scaler, labels), os.path.join(OUT_DIR, "svm_fma_rff_linear.pkl"))
    save_confusion(yte, yp, labels, os.path.join(OUT_DIR, "cm_fma_rff_linear.png"),
                   title="Confusion Matrix (FMA: RFF + LinearSVC)")
    print("💾 Saved: models/svm_fma_rff_linear.pkl and models/cm_fma_rff_linear.png")

if __name__ == "__main__":
    main()
