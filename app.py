import flask, pickle, json, os
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer
from preprocess import preprocess_base, preprocess_for_vector
from dictionaries_and_extracting_features import extract_features

app = flask.Flask(__name__, template_folder='templates')

vectorizer = scaler = embedder = None

with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/best_model_info.json", "r") as f:
    model_info = json.load(f)

rep = model_info["representation"].lower()

preprocess_func = preprocess_for_vector if ("tfidf" in rep or "bow" in rep) else preprocess_base

if os.path.exists("models/vectorizer.pkl"):
    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

if os.path.exists("models/scaler.pkl"):
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

if "embedder" in rep: 
    embedder = SentenceTransformer(model_info["embedder_name"])
        
def custom_features(text):
    if not scaler:
        raise ValueError("There is no scaler")
    features = extract_features(text)
    X = pd.DataFrame([features])
    X_scaled = scaler.transform(X)
    return X_scaled

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None

    if flask.request.method =="POST":
        text = flask.request.form["sentiment"]
        clean_text = preprocess_func(text)

        combination = []

        if "custom" in rep:
            X_cus = custom_features(clean_text)
            combination.append(csr_matrix(X_cus))
                
        if "embedder" in rep:
            X_emb = embedder.encode([clean_text])
            combination.append(csr_matrix(X_emb))
        
        if ("tfidf" in rep) or ("bow" in rep):
            X_vec = vectorizer.transform([clean_text])
            combination.append(X_vec)

        if len(combination)==0:
            raise RuntimeError("There is no part of rep in model_info")

        X = combination[0] if len(combination) == 1 else hstack(combination)

        pred = model.predict(X)[0]
        sentiment = "Positive" if pred==1 else "Negative"

    return flask.render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)