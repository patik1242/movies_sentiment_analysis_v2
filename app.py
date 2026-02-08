import flask, pickle, json, os
from sentence_transformers import SentenceTransformer
from preprocess import preprocess_base, preprocess_for_tfidf

app = flask.Flask(__name__, template_folder='templates')

with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

if os.path.exists("models/vectorizer.pkl"):
    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

if os.path.exists("models/scaler.pkl"):
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

with open("models/best_model_info.json", "r") as f:
    model_info = json.load(f)

if "tfidf" in model_info["representation"]:
    preprocess_func = preprocess_for_tfidf
else:
    preprocess_func = preprocess_base


@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None

    if flask.request.method =="POST":
        text = flask.request.form["sentiment"]

        clean_text = preprocess_func(text)
        if "custom" in model_info["representation"]:
            X = scaler.transform([clean_text])
        elif "embedder" in model_info["representation"]:
            embedder = SentenceTransformer(model_info["representation"])
            X = embedder.encode([clean_text])
        else: 
            X = vectorizer.transform("representation")

        pred = model.predict(X)[0]

        sentiment = "Positive" if pred==1 else "Negative"

    return flask.render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)