
import time 
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier

from training_and_calculate_metrics import train_and_evaluate_model

def train_with_grid_and_custom_features(X_train, X_test, y_train, y_test, allowed_models=None, texts_test=None, dataset=None):    
   
    #Modele z parametrami
    classifiers = {
        "Logistic Regression" : (LogisticRegression(max_iter = 5000, 
                                                random_state=42, 
                                                solver="saga",
                                                class_weight="balanced",
                                                ),
                                                {"C": [10, 100],
                                                    }
                                                ),

        "Linear SVM" : (LinearSVC(max_iter=40000, random_state=42, class_weight="balanced"), 
                {
                    "C": [0.01, 0.1,1], 
                    "loss": ['hinge', 'squared_hinge']
                }),

        "RidgeClassifier": (RidgeClassifier(class_weight="balanced"), 
                            {"alpha": [0.01, 0.1, 1, 10], 
                            "solver": ['auto', 'lsqr']})
    }

    results_imdb = {}

    for model_name, (model, param_grid) in classifiers.items():

        if allowed_models is not None and model_name not in allowed_models:
            continue
        
        X_train_model = X_train
        X_test_model = X_test

        grid = GridSearchCV(estimator = model, 
                            param_grid=param_grid, 
                            scoring = "f1_macro", 
                            n_jobs=-1, 
                            verbose=2, 
                            refit='f1_macro', 
                            cv=5)
        
        start = time.perf_counter()            
        grid.fit(X_train_model, y_train)
        train_time = time.perf_counter() - start 

        best_model = grid.best_estimator_

        train_metrics, test_metrics = train_and_evaluate_model(
            best_model, X_train_model, X_test_model, y_train, y_test, model_name, texts_test=texts_test, dataset=dataset
        )
        
        results_imdb[model_name] = {"best_params": grid.best_params_, 'estimator': grid.best_estimator_, 
                                    'train': train_metrics, 'test': test_metrics, 
                                    "train_time_s": train_time} 
    
    return results_imdb