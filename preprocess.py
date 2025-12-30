import re

def preprocess(text):
    if not isinstance(text, str):
        return ""
    
    #Separacja interpunkcji: zamiast great! jest great !
    text = re.sub(r"([!?.;,])", r" \1 ", text)

    #Usuwamy dziwne znaki, ale zachowujemy interpunkcję
    text = re.sub(r"[^a-zA-Z0-9!?'.,; ]+", "", text)

    #Tokenizacja i usunięcie wielokrotnych spacji
    text = re.sub(r"\s+", " ", text)

    #Zmiana wielkości liter
    text = text.lower()

    #Usunięcie spacji na początku/końcu
    return text.strip()