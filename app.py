import flask
import pickle

from preprocess import preprocess

app = flask.Flask(__name__, template_folder='templates')

with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None

    if flask.request.method =="POST":
        text = flask.request.form["sentiment"]

        clean_text = preprocess(text)
        X = vectorizer.transform([clean_text])

        pred = model.predict(X)[0]

        sentiment = "Positive" if pred==1 else "Negative"

    return flask.render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)