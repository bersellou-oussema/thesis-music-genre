# EfficientNetV2B0 on FMA (windowed spectrograms) with track-wise (grouped) split
# - Splits by track_id so no leakage between train/val/test
# - Saves: best model (.keras), classification report (.txt), confusion matrix (.png)
# - Prints a one-line summary of final test metrics

import os
import re
import glob
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2B0
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# -----------------------
# Paths & hyperparameters
# -----------------------
DATA_DIR = os.path.join("spectrogram_dataset_windowed", "fma")   # <genre> subfolders here
OUT_DIR  = "models"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE  = (224, 224)
BATCH_SIZE = 32
SEED       = 42

# Train/val/test as group splits (by track_id)
TRAIN_PCT = 0.70
VAL_PCT   = 0.15  # of the whole dataset
TEST_PCT  = 0.15

WARMUP_EPOCHS  = 4
FINETUNE_EPOCHS = 8  # total epochs ≈ 18 (adjust if you want longer)

# -------------------------------------------------------
# Utility: list files, labels, and extract numeric track
# -------------------------------------------------------
def extract_track_id(path):
    """Extract numeric track id from filename. Example: '12345_0007.png' -> '12345'."""
    name = os.path.basename(path)
    m = re.match(r"(\d+)[_\-]", name)
    if m:
        return m.group(1)
    # fallback: remove extension and take first chunk before underscore
    stem = os.path.splitext(name)[0]
    return stem.split("_")[0]

def load_index(data_dir):
    """Return lists: filepaths, label_ids, label_names, groups(track_id)."""
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not class_names:
        raise RuntimeError(f"No class folders found under {data_dir}")

    class_to_idx = {c: i for i, c in enumerate(class_names)}
    fpaths, labels, groups = [], [], []

    for c in class_names:
        patt = os.path.join(data_dir, c, "*.png")
        for p in glob.glob(patt):
            fpaths.append(p)
            labels.append(class_to_idx[c])
            groups.append(extract_track_id(p))

    return fpaths, np.array(labels, dtype=np.int64), class_names, np.array(groups)

# -----------------------
# tf.data input pipeline
# -----------------------
def decode_png(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    # light color/contrast jitter
    img = tf.image.random_brightness(img, max_delta=0.10)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    return img, label

def make_dataset(paths, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: decode_png(p, y), num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# ---------------
# Model builder
# ---------------
def build_model(num_classes):
    base = EfficientNetV2B0(include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,))
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(base.input, out)
    return model, base

# ---------------------------
# Confusion matrix plotter
# ---------------------------
def save_confusion(y_true, y_pred, labels, out_png, title):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

# -----------
# Main flow
# -----------
def main():
    print(">>> Loading index...")
    paths, y, class_names, groups = load_index(DATA_DIR)
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    print(f"Total windows: {len(paths)}")
    print("Per-class counts:", Counter(y))

    # -------------------------
    # Grouped split (by track)
    # -------------------------
    # 1) Train vs temp (val+test)
    gss = GroupShuffleSplit(n_splits=1, train_size=TRAIN_PCT, random_state=SEED)
    train_idx, temp_idx = next(gss.split(paths, y, groups))

    # 2) Split temp into val/test by groups
    # fraction for val relative to temp
    val_frac_of_temp = VAL_PCT / (VAL_PCT + TEST_PCT)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_frac_of_temp, random_state=SEED)
    val_idx, test_idx = next(gss2.split(np.array(paths)[temp_idx],
                                        y[temp_idx],
                                        groups[temp_idx]))
    val_idx  = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    # Sanity: no group overlap
    train_groups = set(groups[train_idx])
    val_groups   = set(groups[val_idx])
    test_groups  = set(groups[test_idx])
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    # Build datasets
    tr_paths, tr_y = np.array(paths)[train_idx], y[train_idx]
    va_paths, va_y = np.array(paths)[val_idx],   y[val_idx]
    te_paths, te_y = np.array(paths)[test_idx],  y[test_idx]

    print("\n---- SPLIT SIZES (windows) ----")
    print("train:", len(tr_paths), "| val:", len(va_paths), "| test:", len(te_paths))

    print("\nUnique groups counts:")
    print("train:", len(set(groups[train_idx])),
          " val:", len(set(groups[val_idx])),
          " test:", len(set(groups[test_idx])))

    # Datasets
    train_ds = make_dataset(tr_paths, tr_y, training=True)
    val_ds   = make_dataset(va_paths, va_y, training=False)
    test_ds  = make_dataset(te_paths, te_y, training=False)

    # -------------
    # Build model
    # -------------
    model, base = build_model(num_classes)

    # Warm-up (freeze base)
    base.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    ckpt_path = os.path.join(OUT_DIR, "efficientnetv2_fma_trackwise_best.keras")
    callbacks_warm = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy",
                                           save_best_only=True, save_weights_only=False),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1),
    ]

    print("\n>>> Warm-up training (frozen base)")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=WARMUP_EPOCHS,
        callbacks=callbacks_warm,
        verbose=1,
    )

    # Fine-tune (unfreeze)
    base.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_ft = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy",
                                           save_best_only=True, save_weights_only=False),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1),
    ]

    print("\n>>> Fine-tuning (unfrozen base)")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINETUNE_EPOCHS,
        callbacks=callbacks_ft,
        verbose=1,
    )

    # ----------------
    # Evaluation / IO
    # ----------------
    # Use the best checkpoint (already loaded due to restore_best_weights=True)
    print("\n>>> Evaluating on TEST")
    test_probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(test_probs, axis=1)

    # Classification report
    report_txt = os.path.join(OUT_DIR, "report_efficientnetv2_fma_trackwise.txt")
    rep = classification_report(te_y, y_pred, target_names=class_names, digits=4)
    print("\n" + rep)
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(rep)

    # Confusion matrix
    cm_png = os.path.join(OUT_DIR, "cm_efficientnetv2_fma_trackwise.png")
    save_confusion(te_y, y_pred, class_names, cm_png, "Confusion Matrix (FMA, track-wise, EfficientNetV2B0)")

    # One-line summary
    rep_dict = classification_report(te_y, y_pred, target_names=class_names, output_dict=True)
    acc  = rep_dict["accuracy"]
    prec = rep_dict["weighted avg"]["precision"]
    rec  = rep_dict["weighted avg"]["recall"]
    f1   = rep_dict["weighted avg"]["f1-score"]
    print(f"\n✅ EfficientNetV2B0 (FMA Track-wise, grouped) — "
          f"Accuracy: {acc:.4f} | Weighted Precision: {prec:.4f} | "
          f"Weighted Recall: {rec:.4f} | Weighted F1: {f1:.4f}")
    print(f"💾 Saved: {ckpt_path}, {cm_png}, {report_txt}")

if __name__ == "__main__":
    # Windows-safe multiprocessing guard
    tf.keras.utils.set_random_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    main()
