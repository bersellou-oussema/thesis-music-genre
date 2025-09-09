import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# -------------------- Parameters --------------------
dataset_path = "data_wav"
output_path = "spectrogram_dataset"
img_size = (224, 224)
num_classes = 10
batch_size = 16
epochs = 30
sr = 22050
duration = 30

# # -------------------- Step 1: Convert audio to spectrogram images --------------------
# def convert_dataset_to_images(dataset_path="data_wav", output_path="spectrogram_dataset", max_duration=30, sr=22050):
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)

#     for genre in os.listdir(dataset_path):
#         genre_path = os.path.join(dataset_path, genre)
#         if not os.path.isdir(genre_path):
#             continue

#         print(f"\n🎧 Genre: {genre}")
#         genre_output_path = os.path.join(output_path, genre)
#         os.makedirs(genre_output_path, exist_ok=True)

#         for filename in os.listdir(genre_path):
#             if filename.endswith(".wav"):
#                 file_path = os.path.join(genre_path, filename)
#                 try:
#                     y, sr = librosa.load(file_path, sr=sr)
#                     original_duration = len(y) / sr

#                     # Trim silence
#                     y, _ = librosa.effects.trim(y)

#                     # Cut to max_duration (30s)
#                     if len(y) > sr * max_duration:
#                         y = y[:sr * max_duration]

#                     # Normalize
#                     y = librosa.util.normalize(y)

#                     # Skip too short files
#                     if len(y) < sr * 3:
#                         print(f"⚠️ Skipping {filename}: Too short ({original_duration:.2f}s)")
#                         continue

#                     # Generate mel-spectrogram
#                     mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
#                     mel_db = librosa.power_to_db(mel, ref=np.max)

#                     # Save image
#                     fig = plt.figure(figsize=(2.56, 2.56), dpi=50)
#                     plt.axis('off')
#                     librosa.display.specshow(mel_db, sr=sr, cmap='magma')
#                     output_file = os.path.join(genre_output_path, filename.replace(".wav", ".png"))
#                     plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
#                     plt.close(fig)

#                     print(f"✅ Saved: {output_file}")

#                 except Exception as e:
#                     print(f"❌ Error with {filename}: {e}")

# # ⚠️ Run once only
# convert_dataset_to_images("data_wav", "spectrogram_dataset")

#-------------------- Step 2: Data Generators --------------------
train_datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_data = train_datagen.flow_from_directory(
    output_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_data = train_datagen.flow_from_directory(
    output_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# -------------------- Step 3: Model --------------------
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=img_size + (3,))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------- Step 4: Training --------------------
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=[early_stop])

# -------------------- Step 5: Evaluation --------------------
val_data.reset()
loss, acc = model.evaluate(val_data)
print(f"\n✅ Final Test Accuracy: {acc:.2f}")

# -------------------- Step 6: Classification Report & Confusion Matrix --------------------
y_pred = model.predict(val_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_data.classes
labels = list(val_data.class_indices.keys())

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=labels))

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.title("Confusion Matrix - ResNet50")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
