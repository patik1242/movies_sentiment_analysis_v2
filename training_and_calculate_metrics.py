import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay


def calculate_metrics(y_true, y_pred, model_name, split):
    """ 
    Obliczanie podstawowych metryk klasyfikacji
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted")

    # Wyświetlenie wyników dla danego podziału danych
    print(f"\nMetryki dla {model_name} ({split}): \n")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Wizualizacja macierzy pomyłek tylko dla zbioru testowego
    if split == "test":
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
        plt.title(f"Macierz pomyłek - {model_name} ({split})")
        plt.show()

    # Zwracanie metryk w czytelnym formacie słownika
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    #Trening modelu na zbiorze treningowym
    model.fit(X_train, y_train)

    #Predykcje modelu na train i test 
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    #Wyliczanie metryk jakości dla obu zbiorów
    train_metrics = calculate_metrics(y_train, y_train_pred, model_name, "train")
    test_metrics = calculate_metrics(y_test, y_test_pred, model_name, "test")

    return train_metrics, test_metrics


