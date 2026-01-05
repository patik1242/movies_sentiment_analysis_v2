# 🎬 Sentiment Analysis of Movie Reviews (IMDB)

This project focuses on **sentiment classification of movie reviews** (positive / negative) using various text representations and machine learning models.  
The main goal is to **compare the effectiveness of different text representations** and analyze the impact of **manually engineered linguistic features**.

---

## 📌 Project Scope

The following text representations are compared in this project:

- **Hand-crafted features (custom features)**
- **TF-IDF**
- **SentenceTransformer embeddings (DistilBERT)**
- **TF-IDF + custom features**
- **Embeddings + custom features**

For each representation, multiple classifiers are trained and the best model is selected based on the **F1-score**.

---

## 📂 Project Structure

```text
.
├── data/
│   └── IMDB_Dataset.csv
│
├── main.py
├── load_and_clean_data.py
├── preprocess.py
├── dictionaries_and_extracting_features.py
├── preparation.py
├── compare_representation.py
├── train_with_grid_and_custom_features.py
├── training_and_calculate_metrics.py
├── feature_importance.py
├── requirements.txt
│
├── X_embed_train.npy
├── X_embed_test.npy
├── Best_test_F1_per_representation.png
└── feature_importance_custom.png
```

---

## 📁 Input Data

### `data/IMDB_Dataset.csv`

The project uses the public **IMDB Movie Reviews Dataset**, which contains movie reviews along with sentiment labels.

### 📄 File Description

- **Format:** CSV  
- **Number of records:** 50,000  
- **Number of classes:** 2 (balanced)

### 🧾 Columns

| Column | Type | Description |
|------|------|-------------|
| `review` | string | Movie review text |
| `sentiment` | string | Sentiment label: `positive` / `negative` |

### ⚖️ Class Distribution

- `positive`: ~50%
- `negative`: ~50%

---

## 🧹 Data Cleaning and Preprocessing

Implemented in:
- `load_and_clean_data.py`
- `preprocess.py`

Processing steps:
- removal of duplicate entries
- missing value analysis
- text cleaning
- lowercase normalization
- removal of empty reviews
- label mapping:
  negative -> 0
  positive -> 1


---


---

## 🧠 Hand-Crafted Feature Extraction

File: `dictionaries_and_extracting_features.py`

Extracted features:
- `vader_pos`
- `vader_neg`
- `vader_compound`
- number of exclamation marks
- number of question marks
- number of digits
- number of contrast words (`but`, `however`, `although`, ...)

All features are **standardized** using `StandardScaler`.

---

## 🧾 Text Representations

File: `compare_representation.py`

The following representations are created:
- TF-IDF
- SentenceTransformer embeddings (`distilbert-base-uncased`)
- combinations with custom features

Embeddings are **cached** to disk:
X_embed_train.npy
X_embed_test.npy



---

## 🧠 Hand-Crafted Feature Extraction

File: `dictionaries_and_extracting_features.py`

Extracted features:
- `vader_pos`
- `vader_neg`
- `vader_compound`
- number of exclamation marks
- number of question marks
- number of digits
- number of contrast words (`but`, `however`, `although`, ...)

All features are **standardized** using `StandardScaler`.

---

## 🧾 Text Representations

File: `compare_representation.py`

The following representations are created:
- TF-IDF
- SentenceTransformer embeddings (`distilbert-base-uncased`)
- combinations with custom features

Embeddings are **cached** to disk:

feature_importance_custom.png


---

## 🔍 Feature Importance Analysis

If the best-performing representation consists of **custom features**, feature importance analysis is performed using:
- linear model coefficients
- tree-based feature importances
- permutation importance (fallback)

The result is saved as:


---

## ▶️ Running the Project

1. Install dependencies:
```bash
pip install -r requirements.txt

```
Run the pipeline::
``
python main.py
``
Requirements: 

Listed in requirements.txt

-pandas
-numpy
-scikit-learn
-scipy
-matplotlib
-sentence-transformers
-xgboost
-vaderSentiment

## 🎯 Project Goals

This project enables:

-comparison of classical and modern text representations
-evaluation of the impact of linguistic features on classification quality
-interpretation of NLP model behavio

📚 Data Source

IMDB Movie Reviews Dataset
[IMDB Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)



analizować interpretowalność modeli NLP
