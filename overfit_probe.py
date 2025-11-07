import os, glob, random, math
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

random.seed(42); np.random.seed(42); tf.random.set_seed(42)

IMG_H = 224
IMG_W = 224
BATCH = 16

def resize_to_224(img_np):
    t = tf.convert_to_tensor(img_np[None, ...])
    t = tf.image.resize(t, (IMG_H, IMG_W), method="bilinear", antialias=True)
    return t.numpy()[0]

def load_subset(root, classes, per_class_train=20, per_class_val=5):
    pairs_tr, pairs_va = [], []
    for c in classes:
        files = glob.glob(os.path.join(root, c, "*.npy"))
        files = sorted(files)
        random.shuffle(files)
        tr = files[:per_class_train]
        va = files[per_class_train:per_class_train+per_class_val]
        pairs_tr += [(f, classes.index(c)) for f in tr]
        pairs_va += [(f, classes.index(c)) for f in va]
    random.shuffle(pairs_tr); random.shuffle(pairs_va)
    return pairs_tr, pairs_va

class Seq(tf.keras.utils.Sequence):
    def __init__(self, pairs, n_classes):
        self.pairs = pairs; self.n_classes = n_classes
    def __len__(self): return math.ceil(len(self.pairs)/BATCH)
    def __getitem__(self, idx):
        batch = self.pairs[idx*BATCH:(idx+1)*BATCH]
        X, y = [], []
        for f, lab in batch:
            mel = np.load(f).astype(np.float32)         # [128,W], dB
            mel = (mel + 80.0)/80.0                     # [0,1]
            img = np.stack([mel,mel,mel], axis=-1)
            img = resize_to_224(img)
            X.append(img); y.append(lab)
        X = np.stack(X).astype(np.float32)
        y = tf.keras.utils.to_categorical(y, num_classes=self.n_classes, dtype="float32")
        return X, y

def main():
    root = "spectrogram_dataset_harmonized/gtzan_split/train"
    classes = sorted([d.name for d in Path(root).iterdir() if d.is_dir()])[:2]  # first 2 classes
    print("Using classes:", classes)

    tr_pairs, va_pairs = load_subset(root, classes, per_class_train=20, per_class_val=5)
    print(f"Train samples: {len(tr_pairs)}, Val samples: {len(va_pairs)}")

    train = Seq(tr_pairs, n_classes=len(classes))
    val   = Seq(va_pairs, n_classes=len(classes))

    base = EfficientNetV2B0(include_top=False, input_shape=(IMG_H, IMG_W, 3), pooling="avg")
    base.trainable = True
    x = layers.Dropout(0.2)(base.output)
    out = layers.Dense(len(classes), activation="softmax")(x)
    model = models.Model(base.input, out)

    model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train, validation_data=val, epochs=20, verbose=1)

if __name__ == "__main__":
    main()
