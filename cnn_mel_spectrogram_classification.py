import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 1. Parameters
# -----------------------------
DATASET_PATH = 'data_wav'
genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

SAMPLES_PER_TRACK = 660000  # approx 30 sec at 22kHz
SPECTROGRAM_SHAPE = (128, 128)

# -----------------------------
# 2. Feature Extraction
# -----------------------------
def extract_mel_spectrogram(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        y, _ = librosa.effects.trim(y)  # Trim silence
        y = librosa.util.fix_length(y, size=SAMPLES_PER_TRACK)

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Normalize
        mel_db = (mel_db - np.mean(mel_db)) / np.std(mel_db)

        if mel_db.shape[1] < SPECTROGRAM_SHAPE[1]:
            mel_db = np.pad(mel_db, ((0, 0), (0, SPECTROGRAM_SHAPE[1] - mel_db.shape[1])), mode='constant')
        else:
            mel_db = mel_db[:, :SPECTROGRAM_SHAPE[1]]

        return mel_db
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# -----------------------------
# 3. Build Dataset
# -----------------------------
X = []
y = []

for genre in genres:
    folder_path = os.path.join(DATASET_PATH, genre)
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            mel = extract_mel_spectrogram(file_path)
            if mel is not None:
                X.append(mel)
                y.append(genre)

X = np.array(X)
X = X[..., np.newaxis]  # add channel dim
y_encoded = LabelEncoder().fit_transform(y)
y_categorical = to_categorical(y_encoded)

# -----------------------------
# 4. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical)

# -----------------------------
# 5. CNN Model
# -----------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(genres), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------
# 6. Train
# -----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=30, batch_size=16,
                    validation_split=0.2, callbacks=[early_stop])

# -----------------------------
# 7. Evaluate
# -----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {test_acc:.2f}")

# -----------------------------
# 8. Confusion Matrix + Report
# -----------------------------
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred_classes, target_names=genres))

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=genres, yticklabels=genres)
plt.title("Confusion Matrix - CNN")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
