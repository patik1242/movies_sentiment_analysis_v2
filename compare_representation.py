import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer

from preparation import training_data
from train_with_grid_and_custom_features import train_with_grid_and_custom_features
from feature_importance import evaluate_feature_importance


def comparing_representations(clean_training):
    X_text_train, X_text_test, X_custom_train, X_custom_test, y_train, y_test, scaler = training_data(clean_training)

    #TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1,3), min_df=2)
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

        if rep_model != "custom":
            allowed_models = ["Logistic Regression", "Linear SVM", "RidgeClassifier"]
        else:
            allowed_models = ["Logistic Regression", "Linear SVM", "RidgeClassifier", "XGBoost"]


        results_imdb = train_with_grid_and_custom_features(
            X_tr, X_te, y_train, y_test, allowed_models=allowed_models)
            
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
    
    best_model_info = {
        "name": best_model_name,
        "representation": best_rep, 
        "f1": best_f1
    }

    save_results_to_json(all_results_imdb, best_f1_per_rep, best_model_info)
    save_best_model(
        best_estimator= best_estimator, 
        best_rep = best_rep, 
        best_model_name = best_model_name, 
        vectorizer = vectorizer if "tfidf" in best_rep else None, 
        scaler = scaler if "custom" in best_rep else None
    )

    X_custom_train_df = X_custom_train.copy()
    X_custom_test_df = X_custom_test.copy()

    if best_rep =="custom":
        importance_df = evaluate_feature_importance(
            model = best_estimator, 
            X = X_custom_test_df, 
            y = y_test
        )

        importance_df.sort_values("importance").plot.barh(
            x="feature", y = "importance", legend = False, figsize=(8,5))
        plt.tight_layout()
        plt.savefig("feature_importance_custom.png")
        plt.close()

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


def save_results_to_json(all_results_imdb, best_f1_per_rep, best_model_info):
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_summary = {
        "timestamp": timestamp, 
        "best_model": {
            "name" : best_model_info["name"],
            "representation": best_model_info["representation"],
            "f1_score": best_model_info["f1"]
        }, 
        "best_f1_per_representation": best_f1_per_rep, 
        "all_models": {}
    }

    for rep, models in all_results_imdb.items():
        results_summary["all_models"][rep] = {}
        for model_name, result in models.items():
            results_summary["all_models"][rep][model_name] = {
                "best_params": result["best_params"], 
                "train_metrics": result["train"],
                "test_metrics": result["test"]
            }
    
    output_path = results_dir / f"results_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n Wyniki zapisane do: {output_path}")

    return output_path


def save_best_model(best_estimator, best_rep, best_model_name, vectorizer = None, scaler = None ):
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    with open(models_dir/"best_model.pkl", "wb") as f:
        pickle.dump(best_estimator, f)
    
    model_info = {
        "model_name": best_model_name,
        "representation": best_rep,
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    }

    with open(models_dir / "best_model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    if vectorizer is not None:
        with open(models_dir/"vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)

    if scaler is not None:
        with open(models_dir/"scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
    
    print(f"Model zapisany w {models_dir}")
