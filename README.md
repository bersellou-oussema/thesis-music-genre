# Music Genre Classification – Thesis Project

## 📌 Overview
This project is part of my Master’s thesis **“Data Preprocessing in the Classification of Musical Songs”**.  
The goal is to evaluate how different preprocessing techniques influence the performance of machine learning and deep learning models for music genre classification.

## 📂 Datasets
- **GTZAN** (10 genres, ~100 tracks each)  
- **FMA-small** (8 genres, 8,000 tracks in total)  

Only overlapping genres (*Pop* and *Rock*) were used for some cross-dataset experiments.

## ⚙️ Preprocessing Methods
- Silence trimming (`librosa.effects.trim`)  
- Normalization (`librosa.util.normalize`)  
- Windowing: 1.5s segments with 0.75s hop (50% overlap)  
- Feature extraction for classical ML:
  - MFCC
  - Chroma STFT
  - Zero-Crossing Rate
  - Spectral Centroid  
- Mel-spectrograms for CNNs and transfer learning (EfficientNetV2, ResNet50)

## 🧠 Models Implemented
### Classical Machine Learning
- Support Vector Machine (SVM) → ~46% (GTZAN)
- Random Forest → ~46% (GTZAN)

### Deep Learning
- CNN (Mel-spectrograms) → ~59% (GTZAN)
- ResNet50 (transfer learning) → 63%(GTZAN)

### Transfer Learning + Classical ML (EfficientNetV2 features)
- FMA + RFF + LinearSVC → ~55.9%
- Overlap (GTZAN + FMA merged, Pop & Rock) → ~77.9%
- Cross-dataset GTZAN → FMA → ~55.6%
- Cross-dataset FMA → GTZAN → ~59.2%

## 📊 Results Summary
- Classical ML provided a **baseline** (~46%).  
- CNN improved slightly (~59%).  
- ResNet50 underperformed (~63%).  
- EfficientNetV2 features + SVM achieved the **best performance** (~68% on GTZAN).  
- Cross-dataset experiments highlighted domain shift (55–59%).  

## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/thesis-music-genre.git
   cd thesis-music-genre
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run feature extraction:
   ```bash
   python feature_extraction.py
   ```

5. Train and evaluate (examples):
   ```bash
   python efficientnetv2_train_svm.py
   python efficientnetv2_train_overlap.py
   ```

## 📌 Remarks
- Datasets and generated spectrograms/models are **not included** due to size, but paths are configured in the code.  
- Confusion matrices and full classification reports are available separately.  

---
