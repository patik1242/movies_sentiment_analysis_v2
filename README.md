# 🎬 Analiza sentymentu recenzji filmowych (IMDB)

Projekt dotyczy **klasyfikacji sentymentu recenzji filmowych** (pozytywny / negatywny) z wykorzystaniem różnych reprezentacji tekstu oraz modeli uczenia maszynowego.  
Celem jest **porównanie skuteczności reprezentacji tekstu** oraz analiza wpływu **ręcznie zaprojektowanych cech lingwistycznych**.

---

## 📌 Zakres projektu

W projekcie porównywane są następujące reprezentacje tekstu:

- **Cechy ręczne (custom features)**
- **TF-IDF**
- **Embeddingi SentenceTransformer (DistilBERT)**
- **TF-IDF + cechy ręczne**
- **Embeddingi + cechy ręczne**

Dla każdej reprezentacji trenowane są różne klasyfikatory, a następnie wybierany jest najlepszy model na podstawie metryki **F1-score**.

---

## 📂 Struktura projektu
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

## 📁 Dane wejściowe

### `data/IMDB_Dataset.csv`

Projekt wykorzystuje publiczny zbiór danych **IMDB Movie Reviews Dataset**, zawierający recenzje filmowe wraz z etykietami sentymentu.

### 📄 Opis pliku

- **Format:** CSV  
- **Liczba rekordów:** 50 000  
- **Liczba klas:** 2 (zbalansowane)

### 🧾 Kolumny

| Kolumna | Typ | Opis |
|-------|----|------|
| `review` | string | Tekst recenzji filmowej |
| `sentiment` | string | Etykieta: `positive` / `negative` |

### ⚖️ Rozkład klas

- `positive`: ~50%
- `negative`: ~50%

---

## 🧹 Czyszczenie i preprocessing danych

Realizowany w plikach:
- `load_and_clean_data.py`
- `preprocess.py`

Wykonywane operacje:
- usunięcie duplikatów
- analiza braków danych
- czyszczenie tekstu
- normalizacja liter
- usunięcie pustych recenzji
- mapowanie etykiet:
  negative -> 0
  positive -> 1


---

## 🧠 Ekstrakcja cech ręcznych

Plik: `dictionaries_and_extracting_features.py`

Wykorzystywane cechy:
- `vader_pos`
- `vader_neg`
- `vader_compound`
- liczba wykrzykników
- liczba znaków zapytania
- liczba cyfr
- liczba słów kontrastujących (`but`, `however`, `although`, ...)

Cechy są **standaryzowane** (`StandardScaler`).

---

## 🧾 Reprezentacje tekstu

Plik: `compare_representation.py`

Tworzone reprezentacje:
- TF-IDF
- embeddingi SentenceTransformer (`distilbert-base-uncased`)
- kombinacje z cechami ręcznymi

Embeddingi są **cache’owane** do plików:
X_embed_train.npy
X_embed_test.npy


---

## 🤖 Modele i trenowanie

Plik: `train_with_grid_and_custom_features.py`

Modele:
- Logistic Regression
- Linear SVM
- Ridge Classifier
- XGBoost (tylko dla cech ręcznych)

Trenowanie:
- GridSearchCV (5-fold CV)
- optymalizacja pod **F1-score**
- balans klas (`class_weight="balanced"`)

---

## 📊 Ewaluacja modeli

Plik: `training_and_calculate_metrics.py`

Obliczane metryki:
- Accuracy
- Precision
- Recall
- F1-score

Dla zbioru testowego zapisywana jest:
- macierz pomyłek (`confusion matrix`)

---

## 📈 Porównanie reprezentacji

Dla każdej reprezentacji wybierany jest najlepszy model (wg F1-score).

Wynik porównania zapisywany jest jako:

Best_test_F1_per_representation.png


---

## 🔍 Analiza ważności cech

Jeżeli najlepszą reprezentacją są **cechy ręczne**, wykonywana jest analiza ważności cech:

- współczynniki modeli liniowych
- feature_importances_ (modele drzewiaste)
- permutation importance (fallback)

Wynik:
feature_importance_custom.png


---

## ▶️ Uruchomienie projektu

1. Instalacja zależności:
```bash
pip install -r requirements.txt
```
Uruchomienie pipeline’u:
``
python main.py
``
Wymagania: 

Plik: requirements.txt

-pandas
-numpy
-scikit-learn
-scipy
-matplotlib
-sentence-transformers
-xgboost
-vaderSentiment

## 🎯 Cel projektu

Projekt umożliwia:

-porównanie klasycznych i nowoczesnych reprezentacji tekstu
-ocenę wpływu cech lingwistycznych na jakość klasyfikacji
-interpretację wyników modeli NLP

📚 Źródło danych

IMDB Movie Reviews Dataset
[IMDB Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)



analizować interpretowalność modeli NLP
