import pandas as pd
import numpy as np

from dictionaries_and_extracting_features import extract_features

def add_features(clean_training):
    features_series = clean_training["review"].apply(extract_features)
    features_df = pd.DataFrame(features_series.tolist())

    #Dołączanie cech do oryginalnego datasetu
    clean_training = pd.concat([clean_training, features_df], axis = 1)

    clean_training["has_pos_sentiment"] = clean_training["vader_pos"]>0

    clean_training["has_neg_sentiment"] = clean_training["vader_neg"]>0
    
    print("Reviews WITHOUT any positive sentiment:", (clean_training["has_pos_sentiment"] == 0).mean())
    print("Reviews WITHOUT any negative sentiment:", (clean_training["has_neg_sentiment"] == 0).mean())

    return clean_training