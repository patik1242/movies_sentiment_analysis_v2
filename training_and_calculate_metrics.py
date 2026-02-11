import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import os, time
from pathlib import Path
from feature_importance import false_sentences

def calculate_metrics(y_true, y_pred, model_name, split, dataset):
    charts_dir = Path("charts") / "confusion_matrixses"
    charts_dir.mkdir(parents=True, exist_ok=True)
                     
    """ 
    Obliczanie podstawowych metryk klasyfikacji
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Wyświetlenie wyników dla danego podziału danych
    print(f"\nMetryki dla {model_name} ({split}): \n")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision_macro: {precision_macro:.4f}")
    print(f"Recall_macro: {recall_macro:.4f}")
    print(f"F1-Score_macro: {f1_macro:.4f}")

    title = f"confusion_matrix - {model_name} ({dataset} - {split})"
    # Wizualizacja macierzy pomyłek tylko dla zbioru testowego
    if split == "test":
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
        plt.title(title)
        filename = charts_dir/f"{title}.png"
        i=1
        while filename.exists():
            filename = charts_dir/ f"{title}_{i}.png"
            i+=1
        plt.savefig(filename)
        plt.close()

    # Zwracanie metryk w czytelnym formacie słownika
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1': f1_macro
    }

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name, texts_test, dataset):
    #Predykcje modelu na train i test 
    start = time.perf_counter()
    y_train_pred = model.predict(X_train)
    train_pred_time = time.perf_counter() - start 

    start = time.perf_counter()
    y_test_pred = model.predict(X_test)
    test_pred_time = time.perf_counter() - start 

    #Wyliczanie metryk jakości dla obu zbiorów
    train_metrics = calculate_metrics(y_train, y_train_pred, model_name, "train", dataset=dataset)
    test_metrics = calculate_metrics(y_test, y_test_pred, model_name, "test", dataset=dataset)
    
    train_metrics["predict_time_s"] = train_pred_time
    test_metrics["predict_time_s"] = test_pred_time
    
    false_sentences(texts_test, y_test_pred, y_test, mname = f"{model_name}_{dataset}")

    return train_metrics, test_metrics


