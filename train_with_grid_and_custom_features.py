import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve as learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer



from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import LinearSVC

from training_and_calculate_metrics import train_and_evaluate_model
from load_and_clean_data import training_data

def plot_learning_curve(model, X_train, y_train, title):
    sizes, training_scores, val_scores = learning_curve(model, X_train, y_train, 
                                                        cv =5, 
                                                        train_sizes = np.linspace(0.1, 1.0, 10),
                                                        n_jobs = -1, 
                                                        scoring = 'f1')
    
    mean_training = np.mean(training_scores, axis = 1)
    Standard_Deviation_training = np.std(training_scores, axis=1)

    mean_val = np.mean(val_scores, axis = 1)
    Standard_Deviation_val = np.std(val_scores, axis=1)

    plt.figure(figsize=(10,6))
    plt.plot(sizes,mean_training,label = 'Train', color = 'blue')
    plt.fill_between(sizes, mean_training - Standard_Deviation_training, 
                     mean_training+Standard_Deviation_training, alpha=0.2)
    
    plt.plot(sizes, mean_val, label = "Validation", color = "red")
    plt.fill_between(sizes, mean_val-Standard_Deviation_val, 
                     mean_val+Standard_Deviation_val, alpha = 0.2)
    
    plt.xlabel('Training Set Size')
    plt.ylabel('F1 Score')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def train_with_grid_and_custom_features(pre_clean_training, clean_training):

    X_text_train, X_text_test, X_custom_train, X_custom_test, y_train, y_test = training_data(pre_clean_training, clean_training)

    #TFIDF
    vectorizer = TfidfVectorizer()
    X_text_train_tfdif = vectorizer.fit_transform(X_text_train)
    X_text_test_tfdif = vectorizer.transform(X_text_test)

    #EMBEDDER
    embedder = SentenceTransformer('distilbert-base-uncased-finetuned-sst-2-english')

    X_embed_train = embedder.encode(
            X_text_train.tolist(),
            show_progress_bar = True
            )
    
    X_embed_test = embedder.encode(
            X_text_test.tolist(),
            show_progress_bar = True
            )
    
    X_embed_train = csr_matrix(X_embed_train)
    X_embed_test = csr_matrix(X_embed_test)

    data = {"tfidf": [X_text_train_tfdif, X_text_test_tfdif], 
            "embed": [X_embed_train, X_embed_test]}
    
    all_results_imdb = {}

    for rep_name, (train, test) in data.items():
        #Łączenie TF-IDF/Embeddera + customowych cech 
        X_train_final = hstack([train, X_custom_train])
        X_test_final = hstack([test, X_custom_test])
        
        #Modele z parametrami
        classifiers = {
            "Logistic Regression" : (LogisticRegression(max_iter = 5000, 
                                                    random_state=42, 
                                                    solver="saga",
                                                    class_weight="balanced",
                                                    ),
                                                    {"C": [10, 100],
                                                        "l1_ratio": [0, 0.5, 1.0], 
                                                        }
                                                    ),

            "Linear SVM" : (LinearSVC(max_iter=10000, random_state=42, class_weight="balanced"), 
                    {
                        "C": [0.1,1,10], 
                        "loss": ['hinge', 'squared_hinge']
                    }),

            "RidgeClassifier": (RidgeClassifier(class_weight="balanced"), 
                                {"alpha": [0.01, 0.1, 1, 10], 
                                "solver": ['auto', 'svd', 'lsqr']})
        }

        results_imdb = {}

        print("====WYNIKI DLA IMDB DATASET====")

        for model_name, (model, param_grid) in classifiers.items():
            
            grid = GridSearchCV(estimator = model, 
                                param_grid=param_grid, 
                                scoring={'accuracy': 'accuracy', 
                                         'f1': 'f1', 
                                         'precision': 'precision', 
                                         'recall': 'recall'}, 
                                n_jobs=-1, 
                                verbose=2, 
                                refit='f1')

            grid.fit(X_train_final, y_train)

            cv_results = grid.cv_results_

            print(f"\n{model_name} - CV wyniki: ")

            for metric in ['accuracy', 'f1', 'precision', 'recall']:
                print(f"{metric}: {cv_results[f"mean_test_{metric}"].mean():.4f} (+/- {cv_results[f"std_test_{metric}"].mean():.4f})")
            
            best_model = grid.best_estimator_

            plot_learning_curve(best_model, X_train_final, y_train, f"Learning Curve - {model_name}")

            print(f"Najlepsze parametry: {grid.best_params_}")

            train_metrics, test_metrics = train_and_evaluate_model(
                best_model, X_train_final, X_test_final, y_train, y_test, model_name
            )
            
            results_imdb[model_name] = {"best_params": grid.best_params_, 'train': train_metrics, 'test': test_metrics} 

            train_f1 = train_metrics["f1"]
            test_f1 = test_metrics["f1"]

            print(f"\n{model_name}: ")
            print(f" F1 train: {train_f1:.4f}, test: {test_f1:.4f}")
            if train_f1 - test_f1 > 0.05:
                print("\n Wniosek: Model wykazuje oznaki przetrenowania (overfitting).")
            elif abs(train_f1 - test_f1) < 0.02:
                print("\n Wniosek: Model dobrze generalizuje, brak oznak przetrenowania.")
            else:
                print("\n Wniosek: Model działa poprawnie, z lekką różnicą w generalizacji.")
        
        #Zestawienie wyników w formie tabeli
        summary = []
        for name_metrics, metrics in results_imdb.items():
            summary.append({
                "Model": name_metrics, 
                "Train acc": metrics['train']['accuracy'],
                "Test acc": metrics['test']['accuracy'],
                "Test F1": metrics['test']['f1']
            })

        df_summary = pd.DataFrame(summary)
        print("\nPodsumowanie wyników:\n")
        print(df_summary.to_string(index=False))

        # Wykres porównawczy metryk testowych
        plt.figure(figsize=(12,6))

        plot_data = []
        for name_results in results_imdb:
            plot_data.append({
                'Model': name_results,
                'Accuracy': results_imdb[name_results]['test']['accuracy'],
                'Precision': results_imdb[name_results]['test']['precision'],
                'Recall': results_imdb[name_results]['test']['recall'],
                'F1': results_imdb[name_results]['test']['f1']
            })
            
        
        df_plot = pd.DataFrame(plot_data)
        df_plot.set_index('Model').plot(kind='bar', figsize=(12,6))
        plt.title(f"Porównanie metryk testowych — IMDB ({rep_name})")
        plt.ylabel("Wartosc metryki")
        plt.ylim(0,1) #oś y w przedziale od 0,1
        plt.xticks(rotation=45) #napis pod kątem
        plt.tight_layout()
        plt.show()
        
        all_results_imdb[rep_name] = results_imdb

    return all_results_imdb