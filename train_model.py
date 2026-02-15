
import time 
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline 
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from training_and_calculate_metrics import train_and_evaluate_model

def train_model(X_train, X_test, y_train, y_test, vectorizer_type = "tfidf", texts_test=None, dataset=None):    

    if vectorizer_type == "tfidf":
        vec = TfidfVectorizer(sublinear_tf=True)
    elif vectorizer_type == "bow":
        vec = CountVectorizer()
    else:
        raise ValueError("Vectorizer must be 'tfidf' or 'bow' ")
    
    pipe = Pipeline([
        ("vec", vec), 
        ("kbest", SelectKBest(chi2)), 
        ("clf", LinearSVC(max_iter=40000, random_state=42, class_weight="balanced"))
    ])

    param_grid = [
        {
            "vec__ngram_range": [(1,2)], 
            "vec__min_df": [2, 5], 
            "vec__max_df": [0.9, 0.95], 
            "vec__max_features": [120000], 
            "kbest__k": [40000, "all"], 
            "clf__loss": ["hinge"], 
            "clf__dual": [True], 
            "clf__C": [0.1, 0.5, 1, 5]
        }, 
        {
            "vec__ngram_range": [(1,2)], 
            "vec__min_df": [2,5], 
            "vec__max_df": [0.9, 0.95], 
            "vec__max_features": [120000], 
            "kbest__k": [40000, "all"], 
            "clf__loss": ["squared_hinge"], 
            "clf__dual": [False], 
            "clf__C": [0.1, 0.5, 1, 5]
        }, 
    ]
    
    grid = GridSearchCV(pipe, param_grid, scoring="f1_macro", cv=5, n_jobs=-1, verbose=2, refit=True)
    start = time.perf_counter()            
    grid.fit(X_train, y_train)
    train_time = time.perf_counter() - start 

    train_metrics, test_metrics = train_and_evaluate_model(
        grid.best_estimator_, X_train, X_test, y_train, y_test, f"Linear SVM ({dataset})", texts_test=texts_test, dataset=dataset
    )
    
    return {"Linear SVM" : {"best_params": grid.best_params_, 'estimator': grid.best_estimator_, 
                                'train': train_metrics, 'test': test_metrics, 
                                "train_time_s": train_time} }

def train_with_grid_and_custom_features(X_train, X_test, y_train, y_test, texts_test=None, dataset=None):    
    

    model = LinearSVC(max_iter=40000, random_state=42, class_weight="balanced")

    param_grid = [
        {   
            "C": [0.1, 0.5, 1, 5],
            "loss": ["hinge"],
            "dual": [True],
        },
        {   
            "C": [0.1, 0.5, 1, 5],
            "loss": ["squared_hinge"],
            "dual": [False],
        },
    ]

    grid = GridSearchCV(estimator = model, 
                        param_grid=param_grid, 
                        scoring = "f1_macro", 
                        n_jobs=-1, 
                        verbose=2, 
                        refit=True, 
                        cv=5)
    
    start = time.perf_counter()            
    grid.fit(X_train, y_train)
    train_time = time.perf_counter() - start 

    best_model = grid.best_estimator_

    train_metrics, test_metrics = train_and_evaluate_model(
        best_model, X_train, X_test, y_train, y_test, f"Linear SVM ({dataset})" , texts_test=texts_test, dataset=dataset
    )
        
    return {
        "Linear SVM": {
            "best_params": grid.best_params_,
            "estimator": best_model,
            "train": train_metrics,
            "test": test_metrics,
            "train_time_s": train_time
        }
    }