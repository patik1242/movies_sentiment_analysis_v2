import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#słownik
contrast_words = {
    "but", "however", "although", "though", "yet", "despite",
    "nevertheless", "nonetheless", "still", "whereas"
}

analyzer = SentimentIntensityAnalyzer()

def extract_features(text):
    exclamation_count = digit_count = question_count = 0 
    contrast_count = 0 

    words = text.split()
    contrast_count = sum(1 for w in words if w in contrast_words)
    digit_count = sum(1 for w in words if w.isdigit())
    exclamation_count = text.count("!")
    question_count = text.count("?")

    scores = analyzer.polarity_scores(text)
    pos = scores["pos"]
    neg = scores["neg"]
    compound = scores["compound"]

    return {
        "vader_pos": pos, 
        "vader_neg": neg,
        "vader_compound": compound,
        "exclamation_count": exclamation_count, 
        "digit_count": digit_count, 
        "question_count": question_count, 
        "contrast_count": contrast_count, 
        "has_exclamation": int(exclamation_count)>0,
        "has_question": int(question_count)>0, 
        "has_contrast": int(contrast_count)>0, 
        "has_digits": int(digit_count)>0, 
        "neg_dominates": int(neg>pos),
        "pos_dominates": int(pos>neg), 
        "high_positive": int(pos>0.5),
        "high_negative": int(neg>0.5),
        "pos_neg_diff": pos-neg,
        "mixed_sentiment": int((pos>0.1) and (neg>0.1)),
        "neg_to_pos_ratio": neg / pos if pos > 0 else neg, 
        "compound_mismatch": abs(compound- (pos-neg))
    }