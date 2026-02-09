import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk_ready = False
stops = None
wnl = WordNetLemmatizer()

def ensure_nltk():
    global nltk_ready, stops
    if nltk_ready:
        return
    nltk.download("wordnet", quiet=True)
    #nltk.download("stopwords", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    #stops = set(stopwords.words('english'))

    #stops -= {"not", "no", "nor", "never",
    #"none", "nobody", "nothing", "nowhere",
    #"neither", "without", "hardly", "barely", "scarcely"}

    nltk_ready=True

def preprocess_base(text):
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

def preprocess_for_vector(text):
    ensure_nltk()

    text = preprocess_base(text)

    #usunięcie wielokrotnych znaków interpunkcyjnych
    text = re.sub(r"([!?])\1{1,}", r"\1", text)

    words = text.split()
    #words = [w for w in words if w not in stops]
    words_l = [wnl.lemmatize(w) for w in words]

    return " ".join(words_l) 