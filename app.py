import flask, pickle, json, os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from preprocess import preprocess_base, preprocess_for_vector
from dictionaries_and_extracting_features import extract_features

app = flask.Flask(__name__, template_folder='templates')

vectorizer = scaler = embedder = None

with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/best_model_info.json", "r") as f:
    model_info = json.load(f)

rep = model_info["representation"].lower()

preprocess_func = preprocess_for_vector if "tfidf" in rep else preprocess_base

if os.path.exists("models/vectorizer.pkl"):
    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

if os.path.exists("models/scaler.pkl"):
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

if "embedder" in rep: 
    embedder = SentenceTransformer(model_info["embedder_name"])
        
def custom_features(text):
    features = extract_features(text)
    X = pd.DataFrame([features])
    X_scaled = pd.DataFrame(
        scaler.transform(X), 
        columns = X.columns, 
        index = X.index
    )
    return X_scaled.values

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None

    if flask.request.method =="POST":
        text = flask.request.form["sentiment"]

        clean_text = preprocess_func(text)

        if "custom" in model_info["representation"]:
            X = custom_features(clean_text)
        
        if "embedder" in model_info["representation"]:
            X = embedder.encode([clean_text])
        else: 
            X = vectorizer.transform([clean_text])

        pred = model.predict(X)[0]

        sentiment = "Positive" if pred==1 else "Negative"

    return flask.render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)