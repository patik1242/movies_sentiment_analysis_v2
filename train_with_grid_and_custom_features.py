import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve as learning_curve
from scipy.sparse import issparse

from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from training_and_calculate_metrics import train_and_evaluate_model

def plot_learning_curve(model, X_train, y_train, title):
    sizes, training_scores, val_scores = learning_curve(model, X_train, y_train, 
                                                        cv =5, 
                                                        train_sizes = np.linspace(0.1, 1.0, 10),
                                                        n_jobs = -1, 
                                                        scoring = 'f1_weighted')
    
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
    plt.savefig(title)
    plt.close()

def train_with_grid_and_custom_features(X_train, X_test, y_train, y_test):    
   
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
                        "max_depth": [4,6], 
                        "n_estimators": [200,400], 
                        "learning_rate": [0.05, 0.1],
                        "subsample": [0.8,1.0], 
                        "colsample_bytree": [0.8, 1.0]
                    })
    }

    results_imdb = {}

    for model_name, (model, param_grid) in classifiers.items():
        
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
                            refit='f1')
        
            
        grid.fit(X_train_model, y_train)

        best_model = grid.best_estimator_

        train_metrics, test_metrics = train_and_evaluate_model(
            best_model, X_train_model, X_test_model, y_train, y_test, model_name
        )
        
        results_imdb[model_name] = {"best_params": grid.best_params_, 'estimator': grid.best_estimator_, 'train': train_metrics, 'test': test_metrics} 
    
    return results_imdb