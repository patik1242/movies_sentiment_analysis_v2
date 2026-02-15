import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import csr_matrix, issparse
from sentence_transformers import SentenceTransformer

from preprocess import preprocess_for_vector
from preparation import training_data
from train_model import train_model, train_with_grid_and_custom_features
from feature_importance import evaluate_feature_importance, mcnemar
from json_files import save_results_to_json, save_best_model


def comparing_representations(clean_training):
    charts_dir = Path("charts")
    charts_dir.mkdir(exist_ok=True)
    
    preds = {}
    X_text_train, X_text_test, X_custom_train, X_custom_test, y_train, y_test, scaler = training_data(clean_training)
    
    X_text_train_vector = X_text_train.apply(preprocess_for_vector)
    X_text_test_vector = X_text_test.apply(preprocess_for_vector)
    
    #TF-IDF
    #vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df = 0.95, max_features=40000, sublinear_tf=True)
    #X_train_tfidf = vectorizer.fit_transform(X_text_train_vector)
    #X_test_tfidf = vectorizer.transform(X_text_test_vector)

    X_custom_train_sparse = csr_matrix(X_custom_train)
    X_custom_test_sparse = csr_matrix(X_custom_test)

    #Bag of Words
    #bow = CountVectorizer()
    #X_train_bow = bow.fit_transform(X_text_train_vector)
    #X_test_bow = bow.transform(X_text_test_vector)

    #Embedder
    try:
        X_train_embed = np.load("X_embed_train.npy")
        X_test_embed = np.load("X_embed_test.npy")
        print("Loaded cache embedding")
    except FileNotFoundError:
        print("Computing embeddings...")
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
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
        
    data = {"custom": [X_custom_train_sparse, X_custom_test_sparse], #sparse
            "tfidf": [X_text_train_vector, X_text_test_vector], #sparse
            "embedder": [X_train_embed, X_test_embed], #dense 
            "bow": [X_text_train_vector, X_text_test_vector]}    
    
    all_results_imdb = {}
    for rep_model, (X_tr, X_te) in data.items():

        print(rep_model,
        "| train:", type(X_tr), "sparse?" , issparse(X_tr),
        "| test:",  type(X_te), "sparse?" , issparse(X_te),
        "| shape:", X_tr.shape)

        if rep_model in ["tfidf", "bow"]:
            results_imdb = train_model(
                X_tr, X_te, y_train, y_test, vectorizer_type = rep_model, texts_test=X_text_test, dataset=rep_model)
        else:
            results_imdb = train_with_grid_and_custom_features(
                X_tr, X_te, y_train, y_test,texts_test=X_text_test, dataset=rep_model)

        for model_name in results_imdb:
            if rep_model in ["tfidf", "bow"]:
                est = results_imdb[model_name]["estimator"]
                results_imdb[model_name]["n_features"] = est.named_steps["kbest"].k
            else:
                results_imdb[model_name]["n_features"] = X_tr.shape[1]

        all_results_imdb[rep_model] = results_imdb
        best_est = max(results_imdb.values(), key=lambda x: x["test"]["f1"])["estimator"]
        preds[rep_model] = best_est.predict(X_te)
   
    best_f1 = -1
    best_model_name = None
    best_rep = None
    best_f1_per_rep = {}

    for representation, model_dict in all_results_imdb.items():
        for model, results in model_dict.items():
            f1 = results["test"]["f1"]

            if f1 > best_f1:
                best_f1 = f1
                best_rep = representation
                best_model_name = model
            
            if representation not in best_f1_per_rep:
                best_f1_per_rep[representation] = f1
            else:
                best_f1_per_rep[representation] = max(best_f1_per_rep[representation], f1)
            
    custom_models = all_results_imdb["custom"]

    best_custom_estimator = max(
        custom_models.values(), key=lambda x: x["test"]["f1"]
    )["estimator"]

    importance_df = evaluate_feature_importance(
        model = best_custom_estimator, 
        X = X_custom_test,
        y = y_test
    )

    importance_df.sort_values("importance").plot.barh(
        x="feature", y = "importance", legend = False, figsize=(8,5))
    plt.tight_layout()
    plt.savefig(charts_dir /"feature_importance_custom.png")
    plt.close()

    best_model_info = {
        "name": best_model_name,
        "representation": best_rep, 
        "f1": best_f1
    }

    rep_sorted = sorted(best_f1_per_rep.items(), key=lambda x: x[1], reverse=True)
    first_rep = rep_sorted[0][0]
    second_rep = rep_sorted[1][0]

    mcnemar_results = mcnemar(preds[first_rep], preds[second_rep], y_test)

    save_results_to_json(all_results_imdb, best_f1_per_rep, best_model_info, second_rep, mcnemar_results)

   
    plt.figure(figsize=(12,6))

    df_plot = pd.DataFrame.from_dict(best_f1_per_rep, orient = "index", columns =["F1"])
    df_plot.plot(kind="bar", legend=False)
    plt.title(f"Best test F1 per representation")
    plt.ylabel("Wartosc metryki")
    plt.ylim(0,1) #oś y w przedziale od 0,1
    plt.xticks(rotation=45) #napis pod kątem
    plt.tight_layout()
    plt.savefig(charts_dir /"Best_test_F1_per_representation.png")
    plt.close()
