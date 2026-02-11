import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import binom, chi2


def evaluate_feature_importance(model, X, y=None, method="auto"):
    
    if not hasattr(X, "columns"):
        raise ValueError("X must be a DataFrame with column names")

    if method=="auto":
        if hasattr(model, "coef_"):
            method = "linear"
        else:
            method = "permutation"

    if method=="linear":
        importance = np.mean(np.abs(model.coef_), axis=0)

    elif method=="permutation":
        if y is None:
            raise ValueError("Permutation importances requires y.")
        
        result = permutation_importance(model, X,y,n_repeats = 10, random_state=42)
        importance = result.importances_mean
    
    else:
        raise ValueError("Unknown method")


    return (
        pd.DataFrame({
            "feature": X.columns, 
            "importance": importance
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

def false_sentences(texts_test, y_pred, y_test, name="model"):
    charts_dir = Path("charts")
    charts_dir.mkdir(exist_ok=True)

    fp_mask = (y_pred==1) & (y_test==0)
    fn_mask = (y_pred==0) & (y_test==1)
    tp = np.sum((y_pred==1) & (y_test==1))
    tn = np.sum((y_pred==0) & (y_test==0))

    fp = np.sum(fp_mask)
    fn = np.sum(fn_mask)

    df = pd.DataFrame({
        "text": texts_test,
        "true": y_test, 
        "pred": y_pred
    })

    df[fp_mask].to_csv(charts_dir/f"{name}_false_positive.csv", index=False)
    df[fn_mask].to_csv(charts_dir/f"{name}_false_negative.csv", index=False)

    labels = ["True Positive", "True Negative", "False Positive", "False Negative"]
    values = [tp, tn, fp, fn]

    plt.figure(figsize=(6,6))
    plt.pie(values, labels=labels, autopct='%1.1f%%')
    plt.title("Types of errors")
    plt.savefig(charts_dir /f"{name}_error_pie.png")
    plt.close()

def mcnemar(y_pred_a, y_pred_b, y_test):
    correct_a = (y_pred_a== y_test)
    correct_b = (y_pred_b==y_test)

    b = int(np.sum(correct_a & (~correct_b)))
    c = int(np.sum((~correct_a) & correct_b))
    
    n = b+c
    if n==0:
        return {"b": b, "c": c, "n": n, "method": "none", "p_value": 1.0}

    n_min, n_max = sorted([b,c])

    if n < 25:
        pvalue = 2* binom.cdf(n_min, n, 0.5) - binom.pmf(n_min, n, 0.5)
        return {"b": b, "c": c, "n": n,"method": "exact","p_value": float(min(1.0, pvalue))}

    chi2_statistic = (abs(b-c)-1)**2/n
    pvalue = chi2.sf(chi2_statistic, 1)

    return {"b": b, "c": c, "n": n, "method": "chi2", "chi2": float(chi2_statistic), "p_value": float(pvalue)}