import pandas as pd
import numpy as np

from dictionaries_and_extracting_features import extract_features

def add_features(clean_training):
    features_series = clean_training["review"].apply(extract_features)
    features_df = pd.DataFrame(features_series.tolist())

    #Dołączanie cech do oryginalnego datasetu
    clean_training = pd.concat([clean_training, features_df], axis = 1)

    clean_training["has_pos_word"] = (
        (clean_training["pos"]>0) | 
        (clean_training["negated_neg_count"]>0)
    )

    clean_training["has_neg_word"] = (
        (clean_training["neg"]>0) | 
        (clean_training["negated_pos_count"]>0)
    )

    print("Reviews WITHOUT any positive words:", (clean_training["has_pos_word"] == 0).mean())
    print("Reviews WITHOUT any negative words:", (clean_training["has_neg_word"] == 0).mean())

    return clean_training