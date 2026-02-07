import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import os 
from pathlib import Path
from feature_importance import false_sentences

def calculate_metrics(y_true, y_pred, model_name, split):
    charts_dir = Path("charts") / "confusion_maxtrises"
    charts_dir.mkdir(parents=True, exist_ok=True)
                     
    """ 
    Obliczanie podstawowych metryk klasyfikacji
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Wyświetlenie wyników dla danego podziału danych
    print(f"\nMetryki dla {model_name} ({split}): \n")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision_macro: {precision_macro:.4f}")
    print(f"Recall_macro: {recall_macro:.4f}")
    print(f"Precision_weighted: {precision_weighted:.4f}")
    print(f"Recall_weighted: {recall_weighted:.4f}")
    print(f"F1-Score: {f1_macro:.4f}")
    print(f"F1-Score_weighted: {f1_weighted:.4f}")

    title = f"confusion_matrix - {model_name} ({split})"
    # Wizualizacja macierzy pomyłek tylko dla zbioru testowego
    if split == "test":
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
        plt.title(title)
        filename = f"{title}.png"
        i=1
        while os.path.exists(filename):
            filename = charts_dir/ f"{title}_{i}.png"
            i+=1
        plt.savefig(filename)
        plt.close()

    # Zwracanie metryk w czytelnym formacie słownika
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name, texts_test):
    #Predykcje modelu na train i test 
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    #Wyliczanie metryk jakości dla obu zbiorów
    train_metrics = calculate_metrics(y_train, y_train_pred, model_name, "train")
    test_metrics = calculate_metrics(y_test, y_test_pred, model_name, "test")
    
    false_sentences(texts_test, y_test_pred, y_test, model_name)

    return train_metrics, test_metrics


