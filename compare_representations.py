import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer

from load_and_clean_data import training_data
from train_with_grid_and_custom_features import train_with_grid_and_custom_features, plot_learning_curve

def comparing_representations(pre_clean_training, clean_training):
    X_text_train, X_text_test, X_custom_train, X_custom_test, y_train, y_test = training_data(pre_clean_training, clean_training)

    #TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_text_train)
    X_test_tfidf = vectorizer.transform(X_text_test)

    #Połączenie TF-IDF + custom 
    X_train_custom_tfidf = hstack([X_train_tfidf, X_custom_train])
    X_test_custom_tfidf = hstack([X_test_tfidf, X_custom_test])

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
        

    X_train_embed = csr_matrix(X_train_embed)
    X_test_embed = csr_matrix(X_test_embed)

    #Połączenie embedder + custom 
    X_train_custom_embed = hstack([X_train_embed, X_custom_train])
    X_test_custom_embed = hstack([X_test_embed, X_custom_test])

    data = {"custom": [X_custom_train, X_custom_test],
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
    plot_data = []
    for representation, model_dict in all_results_imdb.items():
        for model, results in model_dict.items():
            f1 = results["test"]["f1"]

            if f1 > best_f1:
                best_f1 = f1
                best_estimator = results["estimator"]
                best_rep = representation
                best_model_name = model
            
            plot_data.append({
                'Representation': representation,
                'Model': model,  
                'F1': f1
        })

    plot_learning_curve(best_estimator, data[best_rep][0], y_train, f"Learning curve - {best_model_name}")

    plt.figure(figsize=(12,6))

    df_plot = pd.DataFrame(plot_data)
    df_plot.set_index('Representation')["F1"].plot(kind='bar', figsize=(12,6))
    plt.title(f"Porównanie metryk testowych — IMDB")
    plt.ylabel("Wartosc metryki")
    plt.ylim(0,1) #oś y w przedziale od 0,1
    plt.xticks(rotation=45) #napis pod kątem
    plt.tight_layout()
    plt.show()