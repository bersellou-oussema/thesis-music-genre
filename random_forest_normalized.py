import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------------
# 1. Setup paths and genres
# -----------------------------
DATASET_PATH = 'data_wav'  # Your GTZAN path
genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# -----------------------------
# 2. Feature Extraction
# -----------------------------
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30, mono=True)
        y, _ = librosa.effects.trim(y)  # Trim silence

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

        features = [
            np.mean(mfcc),
            np.mean(chroma),
            np.mean(zcr),
            np.mean(spectral_centroid)
        ]
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# -----------------------------
# 3. Build Dataset
# -----------------------------
data = []

for genre in genres:
    folder_path = os.path.join(DATASET_PATH, genre)
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            features = extract_features(file_path)
            if features:
                features.append(genre)
                data.append(features)

# -----------------------------
# 4. Create DataFrame
# -----------------------------
columns = ['mfcc_mean', 'chroma_mean', 'zcr_mean', 'spectral_centroid_mean', 'label']
df = pd.DataFrame(data, columns=columns)

# -----------------------------
# 5. Prepare Data
# -----------------------------
X = df.drop('label', axis=1)
y = LabelEncoder().fit_transform(df['label'])

scaler = StandardScaler()  # Apply normalization
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -----------------------------
# 6. Train Random Forest
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -----------------------------
# 7. Evaluate
# -----------------------------
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=genres))

report = classification_report(y_test, y_pred, target_names=genres, output_dict=True)

df.to_csv("features_rf_norm.csv", index=False)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=genres, yticklabels=genres, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Random Forest + Normalization')
plt.tight_layout()
plt.savefig("confusion_matrix_rf_norm.png")
plt.show()

print("✅ Experiment completed.")
