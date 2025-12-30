import numpy as np
#słowniki 

positive_words = {
    # Podstawowe
    "good", "great", "excellent", "amazing", "wonderful", "fantastic", 
    "brilliant", "superb", "outstanding", "perfect", "best",
    
    # Formy czasownikowe
    "love", "loved", "loving", "loves", "enjoy", "enjoyed", "enjoying",
    "recommend", "recommended", "recommending",
    
    # Filmowe pozytywne
    "masterpiece", "compelling", "captivating", "engaging", "entertaining",
    "impressive", "stunning", "beautiful", "powerful", "moving",
    "memorable", "unforgettable", "extraordinary", "remarkable",
    "touching", "heartwarming", "thrilling", "suspenseful",
    "hilarious", "funny", "witty", "clever", "smart", "intelligent",
    
    # Oceny
    "better", "improved", "superior", "favorite", "favourite",
    
    # Emocje pozytywne
    "happy", "glad", "pleased", "satisfied", "delighted",
    "surprised", "amazed", "impressed", "touched"
}

negative_words = {
    # Podstawowe
    "bad", "terrible", "awful", "horrible", "poor", "worst",
    "disappointing", "disappointed", "disappointment",
    
    # Krytyka
    "boring", "dull", "slow", "tedious",  "waste",
    "wasted", "pointless", "useless", "meaningless",
    
    # Filmowe negatywne
    "predictable", "cliché", "cliche", "generic", "formulaic",
    "unconvincing", "unbelievable", "implausible", "ridiculous",
    "confusing", "confused", "messy", "incoherent",
    "overrated", "overhyped", "mediocre", "forgettable",
    
    # Problemy techniczne
    "poorly", "weak", "lacking", "fails", "failed", "failure",
    "flaw", "flawed", "problem", "problems", "issue", "issues",
    
    # Emocje negatywne
    "hate", "hated", "dislike", "disliked", "annoyed", "annoying",
    "frustrated", "frustrating", "bored"
}

negation_words = {
    "not", "no", "never", "none", "neither", "nor",
    "n't", "dont", "don't", "doesnt", "doesn't",
    "isnt", "isn't", "wasnt", "wasn't",
    "cant", "can't", "couldnt", "couldn't",
    "shouldnt", "shouldn't", "won't", "wont",
    "wouldnt", "wouldn't", "nobody", "nothing", "nowhere", "without"
}

intensifiers = {
    "very", "really", "extremely", "absolutely", "completely",
    "totally", "utterly", "highly", "incredibly", "exceptionally",
    "particularly", "especially", "quite", "rather", "fairly"
}

contrast_words = {
    "but", "however", "although", "though", "yet", "despite",
    "nevertheless", "nonetheless", "still", "whereas"
}


def extract_features(text):
    pos=neg = 0 
    pos_ratio = neg_ratio = 0 
    negated_pos_count = negated_neg_count = 0 
    exclamation_count = digit_count = question_count = 0 
    negation_count = intensifier_count = contrast_count = 0 
    pos_end = neg_end = 0 

    words = text.split()
    text_len = len(words) if len(words)>0 else 1

    pos = sum(1 for w in words if w in positive_words)
    neg = sum(1 for w in words if w in negative_words)
    negation_count = sum(1 for w in words if w in negation_words)
    intensifier_count = sum(1 for w in words if w in intensifiers)
    contrast_count = sum(1 for w in words if w in contrast_words)
    digit_count = sum(1 for w in words if w.isdigit())
    exclamation_count = text.count("!")
    question_count = text.count("?")

    for i, w in enumerate(words):
        if w in negation_words:
            for j in range(1, min(4, len(words)-i)):
                next_word = words[i+j]

                if next_word in positive_words:
                    negated_pos_count+=1
                    break

                elif next_word in negative_words:
                    negated_neg_count+=1
                    break

    effective_pos = pos - negated_pos_count
    effective_neg = neg - negated_neg_count

    pos_ratio = effective_pos/text_len
    neg_ratio = effective_neg/text_len

    last_quarter_start = int(len(words)*0.75)
    last_quarter = words[last_quarter_start:]

    pos_end = sum(1 for w in last_quarter if w in positive_words)
    neg_end = sum(1 for w in last_quarter if w in negative_words)

    return {
        "pos": pos, 
        "neg": neg, 
        "pos_ratio": pos_ratio, 
        "neg_ratio": neg_ratio,
        "negated_pos_count": negated_pos_count, 
        "negated_neg_count": negated_neg_count, 
        "exclamation_count": exclamation_count, 
        "digit_count": digit_count, 
        "question_count": question_count, 
        "negation_count": negation_count, 
        "intensifier_count": intensifier_count, 
        "contrast_count": contrast_count, 
        "pos_end": pos_end, 
        "neg_end": neg_end
    }