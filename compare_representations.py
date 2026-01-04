import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier

from load_and_clean_data import training_data
from train_with_grid_and_custom_features import train_with_grid_and_custom_features

def comparing_representations(clean_training):
    X_text_train, X_text_test, X_custom_train, X_custom_test, y_train, y_test = training_data(clean_training)

    #TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_text_train)
    X_test_tfidf = vectorizer.transform(X_text_test)

    X_custom_train_sparse = csr_matrix(X_custom_train)
    X_custom_test_sparse = csr_matrix(X_custom_test)

    #Połączenie TF-IDF + custom 
    X_train_custom_tfidf = hstack([X_train_tfidf, X_custom_train_sparse])
    X_test_custom_tfidf = hstack([X_test_tfidf, X_custom_test_sparse])

    #Embedder
    try:
        X_train_embed = np.load("X_embed_train.npy")
        X_test_embed = np.load("X_embed_test.npy")
        print("Loaded cache embedding")
    except FileNotFoundError:
        print("Computing embeddings...")
        embedder = SentenceTransformer('distilbert-base-uncased')
        X_train_embed = embedder.encode(
                X_text_train.tolist(), 
                show_progress_bar=True
        )

        X_test_embed = embedder.encode(
                X_text_test.tolist(),
                show_progress_bar= True
        )

        np.save("X_embed_train.npy", X_train_embed)
        np.save("X_embed_test.npy", X_test_embed)
        

    #Połączenie embedder + custom 
    X_train_custom_embed = np.hstack([X_train_embed, X_custom_train])
    X_test_custom_embed = np.hstack([X_test_embed, X_custom_test])

    data = {"custom": [X_custom_train_sparse, X_custom_test_sparse],
            "tfidf": [X_train_tfidf, X_test_tfidf],
            "embedder": [X_train_embed, X_test_embed],
            "custom_tfidf": [X_train_custom_tfidf, X_test_custom_tfidf], 
            "custom_embedder": [X_train_custom_embed, X_test_custom_embed]}
    
    all_results_imdb = {}
    for rep_model, (X_tr, X_te) in data.items():


        results_imdb = train_with_grid_and_custom_features(
            X_tr, X_te, y_train, y_test)
            
        all_results_imdb[rep_model] = results_imdb
    
    best_f1 = -1
    best_model_name = None
    best_estimator = None
    best_rep = None
    best_f1_per_rep = {}
    for representation, model_dict in all_results_imdb.items():
        for model, results in model_dict.items():
            f1 = results["test"]["f1"]

            if f1 > best_f1:
                best_f1 = f1
                best_estimator = results["estimator"]
                best_rep = representation
                best_model_name = model
            
            if representation not in best_f1_per_rep:
                best_f1_per_rep[representation] = f1
            else:
                best_f1_per_rep[representation] = max(best_f1_per_rep[representation], f1)
    

    if isinstance(best_estimator, XGBClassifier):
        pass

    plt.figure(figsize=(12,6))

    df_plot = pd.DataFrame.from_dict(best_f1_per_rep, orient = "index", columns =["F1"])
    df_plot.plot(kind="bar", legend=False)
    plt.title(f"Best test F1 per representation")
    plt.ylabel("Wartosc metryki")
    plt.ylim(0,1) #oś y w przedziale od 0,1
    plt.xticks(rotation=45) #napis pod kątem
    plt.tight_layout()
    plt.savefig("Best_test_F1_per_representation.png")
    plt.close()