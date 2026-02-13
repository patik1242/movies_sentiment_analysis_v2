
import time 
from sklearn.svm import LinearSVC
from training_and_calculate_metrics import train_and_evaluate_model

def train_model(X_train, X_test, y_train, y_test, allowed_models=None, texts_test=None, dataset=None):    
   
    #Modele z parametrami
    classifier = LinearSVC(max_iter=40000, 
                           random_state=42, 
                           class_weight="balanced",
                           C = 1, 
                           loss = "hinge"
                           )
        
    start = time.perf_counter()            
    classifier.fit(X_train, y_train)
    train_time = time.perf_counter() - start 

    train_metrics, test_metrics = train_and_evaluate_model(
        classifier, X_train, X_test, y_train, y_test, "Linear SVM", texts_test=texts_test, dataset=dataset
    )
    
    return {"Linear SVM" : {"best_params": classifier.get_params, 'estimator': classifier, 
                                'train': train_metrics, 'test': test_metrics, 
                                "train_time_s": train_time} }
    