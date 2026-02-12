import json, pickle
from pathlib import Path
from datetime import datetime

def save_results_to_json(all_results_imdb, best_f1_per_rep, best_model_info, second_rep, mcnemar_result):
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
                "test_metrics": result["test"],
                "n_features": result.get("n_features"),
                "train_time_s": result.get("train_time_s")
            }
            
    results_summary["mcnemar"] = {
        "rep_a": best_model_info["representation"],
        "rep_b": second_rep, 
        **mcnemar_result
    }

    output_path = results_dir / f"results_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n Wyniki zapisane do: {output_path}")

    return output_path


def save_best_model(best_estimator, best_rep, best_model_name, vectorizer = None, scaler = None):
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    with open(models_dir/"best_model.pkl", "wb") as f:
        pickle.dump(best_estimator, f)
    
    model_info = {
        "model_name": best_model_name,
        "representation": best_rep,
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        "embedder_name": "all-MiniLM-L6-v2"
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