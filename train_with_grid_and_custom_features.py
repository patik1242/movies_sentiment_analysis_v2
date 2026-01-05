
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from training_and_calculate_metrics import train_and_evaluate_model

def train_with_grid_and_custom_features(X_train, X_test, y_train, y_test, allowed_models=None):    
   
    #Modele z parametrami
    classifiers = {
        "Logistic Regression" : (LogisticRegression(max_iter = 5000, 
                                                random_state=42, 
                                                solver="lbfgs",
                                                class_weight="balanced",
                                                ),
                                                {"C": [10, 100],
                                                    }
                                                ),

        "Linear SVM" : (LinearSVC(max_iter=20000, random_state=42, class_weight="balanced"), 
                {
                    "C": [0.1,1,10], 
                    "loss": ['hinge', 'squared_hinge']
                }),

        "RidgeClassifier": (RidgeClassifier(class_weight="balanced"), 
                            {"alpha": [0.01, 0.1, 1, 10], 
                            "solver": ['auto', 'svd', 'lsqr']}),

        "XGBoost": (XGBClassifier(objective = "binary:logistic", random_state = 42, n_jobs = -1, verbosity = 0, eval_metric = "logloss"), 
                    {
                        "max_depth": [2,3,4], 
                        "n_estimators": [100,200]
                    })
    }

    results_imdb = {}

    for model_name, (model, param_grid) in classifiers.items():

        if allowed_models is not None and model_name not in allowed_models:
            continue
        
        X_train_model = X_train
        X_test_model = X_test

        grid = GridSearchCV(estimator = model, 
                            param_grid=param_grid, 
                            scoring={'accuracy': 'accuracy_weighted', 
                                        'f1': 'f1_weighted', 
                                        'precision': 'precision_weighted', 
                                        'recall': 'recall_weighted'}, 
                            n_jobs=-1, 
                            verbose=2, 
                            refit='f1', 
                            cv=5)
        
            
        grid.fit(X_train_model, y_train)

        best_model = grid.best_estimator_

        train_metrics, test_metrics = train_and_evaluate_model(
            best_model, X_train_model, X_test_model, y_train, y_test, model_name
        )
        
        results_imdb[model_name] = {"best_params": grid.best_params_, 'estimator': grid.best_estimator_, 'train': train_metrics, 'test': test_metrics} 
    
    return results_imdb