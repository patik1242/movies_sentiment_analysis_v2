import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

def evaluate_feature_importance(model, X, y=None, method="auto"):
    
    if not hasattr(X, "columns"):
        raise ValueError("X must be a DataFrame with column names")

    if method=="auto":
        if hasattr(model, "coef_"):
            method = "linear"
        elif hasattr(model, "feature_importances_"):
            method = "tree"
        else:
            method = "permutation"

    if method=="linear":
        importance = np.abs(model.coef_[0])
    
    elif method=="tree":
        importance = model.feature_importances_
    
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
