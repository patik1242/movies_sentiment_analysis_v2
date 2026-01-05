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

    return {
        "vader_pos": scores["pos"], 
        "vader_neg": scores["neg"],
        "vader_compound": scores["compound"],
        "exclamation_count": exclamation_count, 
        "digit_count": digit_count, 
        "question_count": question_count, 
        "contrast_count": contrast_count, 
    }