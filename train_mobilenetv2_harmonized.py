import os, json, glob, math, random
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.applications import MobileNetV2

# =============================
# Config
# =============================
IMG_H = 224
IMG_W = 224
BATCH = 32
NUM_WORKERS = 4
PATIENCE = 8
DROPOUT = 0.35
LABEL_SMOOTH = 0.05
WEIGHT_DECAY = 1e-4
MIXUP_ALPHA = 0.2   # set 0 to disable

random.seed(42); np.random.seed(42); tf.random.set_seed(42)

# =============================
# Augment: mild SpecAugment
# =============================
class SpecAugment:
    def __init__(self, max_time_masks=1, max_freq_masks=1, time_mask_frac=0.05, freq_mask_frac=0.10):
        self.max_time_masks = max_time_masks
        self.max_freq_masks = max_freq_masks
        self.time_mask_frac = time_mask_frac
        self.freq_mask_frac = freq_mask_frac

    def __call__(self, mel):
        M = mel.copy()
        H, W = M.shape
        # freq mask
        F = max(1, int(self.freq_mask_frac * H))
        for _ in range(self.max_freq_masks):
            f = np.random.randint(0, F+1)
            f0 = np.random.randint(0, max(1, H - f))
            M[f0:f0+f, :] = 0.0
        # time mask
        T = max(1, int(self.time_mask_frac * W))
        for _ in range(self.max_time_masks):
            t = np.random.randint(0, T+1)
            t0 = np.random.randint(0, max(1, W - t))
            M[:, t0:t0+t] = 0.0
        return M

# =============================
# Data utils
# =============================
def list_label_names(root_dir):
    return sorted([d.name for d in Path(root_dir).iterdir() if d.is_dir()])

def collect_pairs(root_dir, label_map):
    pairs = []
    for name, idx in label_map.items():
        for f in glob.glob(os.path.join(root_dir, name, "*.npy")):
            pairs.append((f, idx))
    random.shuffle(pairs)
    return pairs

def center_crop_or_pad_width(x, target_w):
    h, w, c = x.shape
    if w == target_w: return x
    if w > target_w:
        s = (w - target_w)//2
        return x[:, s:s+target_w, :]
    pad = target_w - w
    l = pad // 2; r = pad - l
    return np.pad(x, ((0,0),(l,r),(0,0)), mode="constant")

def resize_to(img_np, h=IMG_H, w=IMG_W):
    t = tf.convert_to_tensor(img_np[None, ...])
    t = tf.image.resize(t, (h, w), method="bilinear", antialias=True)
    return t.numpy()[0]

def cmvn_per_frequency(mel_01):
    # per-frequency (row-wise) z-score over time axis
    m = mel_01.mean(axis=1, keepdims=True).astype(np.float32)
    s = mel_01.std(axis=1, keepdims=True).astype(np.float32) + 1e-6
    return (mel_01 - m) / s

def per_image_standardize(img_np):
    m = img_np.mean(dtype=np.float32)
    s = img_np.std(dtype=np.float32) + 1e-6
    return (img_np - m) / s

# =============================
# MixUp utilities
# =============================
def sample_beta_distribution(alpha, size):
    return np.random.beta(alpha, alpha, size=size).astype(np.float32)

def mixup(X, y, alpha=MIXUP_ALPHA):
    if alpha <= 0: 
        return X, y
    lam = sample_beta_distribution(alpha, size=(len(X), 1, 1, 1))
    lam_y = lam[:,0,0,0][:, None]
    index = np.random.permutation(len(X))
    X_mixed = lam * X + (1 - lam) * X[index]
    y_mixed = lam_y * y + (1 - lam_y) * y[index]
    return X_mixed.astype(np.float32), y_mixed.astype(np.float32)

# =============================
# Sequence
# =============================
class NpyMelSequence(tf.keras.utils.Sequence):
    def __init__(self, pairs, batch_size, n_classes, shuffle=True, augment=False, mixup_alpha=MIXUP_ALPHA):
        self.pairs = pairs
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.mixup_alpha = mixup_alpha
        self.indexes = np.arange(len(self.pairs))
        self.on_epoch_end()

    def __len__(self): return math.ceil(len(self.pairs) / self.batch_size)

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        X, y = [], []
        aug = SpecAugment() if self.augment else None
        for i in batch_idx:
            fpath, label = self.pairs[i]
            mel = np.load(fpath).astype(np.float32)     # [128, T] dB
            mel = (mel + 80.0) / 80.0                   # -> [0,1]
            mel = cmvn_per_frequency(mel)               # CMVN
            if aug: mel = aug(mel)
            img = np.stack([mel, mel, mel], axis=-1)    # [128,T,3]
            img = center_crop_or_pad_width(img, min(img.shape[1], 1249))
            img = resize_to(img, IMG_H, IMG_W)          # 224x224
            img = per_image_standardize(img)            # global z-score
            X.append(img); y.append(label)
        X = np.stack(X).astype(np.float32)
        y = tf.keras.utils.to_categorical(y, num_classes=self.n_classes, dtype="float32")
        # MixUp only on training batches
        if self.augment and self.mixup_alpha > 0:
            X, y = mixup(X, y, alpha=self.mixup_alpha)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# =============================
# Model
# =============================
def build_model(num_classes, input_shape=(IMG_H, IMG_W, 3), base_trainable=False, dropout=DROPOUT):
    base = MobileNetV2(include_top=False, input_shape=input_shape, pooling="avg", weights="imagenet")
    base.trainable = base_trainable
    x = base.output
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs=base.input, outputs=out)

# =============================
# Train
# =============================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", required=True)
    ap.add_argument("--val_root",   required=True)
    ap.add_argument("--warmup_epochs", type=int, default=8)
    ap.add_argument("--finetune_epochs", type=int, default=24)
    ap.add_argument("--base_lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()

    # Labels
    label_names = list_label_names(args.train_root)
    label_map = {n:i for i,n in enumerate(label_names)}
    num_classes = len(label_names)
    print("Classes:", label_names)

    # Pairs & datasets
    train_pairs = collect_pairs(args.train_root, label_map)
    val_pairs   = collect_pairs(args.val_root,   label_map)

    train_seq = NpyMelSequence(train_pairs, batch_size=args.batch, n_classes=num_classes, shuffle=True,  augment=True,  mixup_alpha=MIXUP_ALPHA)
    val_seq   = NpyMelSequence(val_pairs,   batch_size=args.batch, n_classes=num_classes, shuffle=False, augment=False, mixup_alpha=0.0)

    # Loss + LR schedule + callbacks
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH)

    steps_per_epoch = max(1, len(train_seq))
    total_steps = steps_per_epoch * (args.warmup_epochs + args.finetune_epochs)
    cosine = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=args.base_lr, first_decay_steps=steps_per_epoch*8)

    Path("models/Deep Learning/MobileNetV2").mkdir(parents=True, exist_ok=True)
    ckpt_path = "models/Deep Learning/MobileNetV2/mobilenetv2_harmonized_best.keras"
    cbs = [
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=PATIENCE, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]

    # Warmup (freeze base)
    model = build_model(num_classes=num_classes, base_trainable=False)
    model.compile(optimizer=AdamW(learning_rate=cosine, weight_decay=WEIGHT_DECAY),
                  loss=loss, metrics=["accuracy"])
    print("\n=== Warmup (base frozen) ===")
    model.fit(train_seq, validation_data=val_seq, epochs=args.warmup_epochs,
              callbacks=cbs, workers=NUM_WORKERS, use_multiprocessing=False)

    # Finetune (unfreeze base)
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "mobilenetv2" in layer.name.lower():
            layer.trainable = True
    # lower LR for finetune (same schedule object is fine; we scale initial LR)
    finetune_lr = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=args.base_lr*0.3, first_decay_steps=steps_per_epoch*8)
    model.compile(optimizer=AdamW(learning_rate=finetune_lr, weight_decay=WEIGHT_DECAY),
                  loss=loss, metrics=["accuracy"])
    print("\n=== Finetune (unfreeze base) ===")
    model.fit(train_seq, validation_data=val_seq, epochs=args.finetune_epochs,
              callbacks=cbs, workers=NUM_WORKERS, use_multiprocessing=False)

    # Save final + labels
    final_path = "models/Deep Learning/MobileNetV2/mobilenetv2_harmonized.h5"
    model.save(final_path)
    with open("models/Deep Learning/MobileNetV2/label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    print("Saved:", final_path)
    print("Saved best:", ckpt_path)

if __name__ == "__main__":
    main()
