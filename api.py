import pandas as pd
import re
import pickle

import io
from io import BytesIO

from flask import Flask, Response, request, jsonify, send_file, render_template, make_response
from flask_cors import CORS

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# Downloading stopwords for text preprocessing
STOPWORDS = set(stopwords.words("english"))

# Initializing Flask app
app = Flask(__name__)
CORS(app)

# Declaring data as a global variable
data = pd.DataFrame()

# Route for testing the service
@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."

# Route for the landing page
@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

# Route for prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # Loading the pre-trained models and data preprocessing tools
    predictor = pickle.load(open(r"Models/model_xgb.pkl", "rb"))
    scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
    cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))
    try:
        # Handling file input for bulk prediction
        if "file" in request.files:
            file = request.files["file"]
            global data
            data = pd.read_csv(file)
            predictions = bulk_prediction(predictor, scaler, cv, data)
            response = make_response(send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            ))
            return response
        # Handling text input for single prediction
        elif "text" in request.json:
            text_input = request.json["text"]
            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
            return jsonify({"prediction": predicted_sentiment})

    except Exception as e:
        return jsonify({"error": str(e)})

# Function for single prediction
def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]
    print(y_predictions)
    return "Positive" if y_predictions == 1 else "Negative"

# Function for bulk prediction
def bulk_prediction(predictor, scaler, cv, data):
    corpus = []
    stemmer = PorterStemmer()
    for i in range(0, data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"])
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)
    y_predictions = list(map(sentiment_mapping, y_predictions))

    data["Predicted sentiment"] = y_predictions
    predictions_csv = BytesIO()

    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)

    return predictions_csv

# Function to map predicted sentiment to readable labels
def sentiment_mapping(x):
    if x == 1:
        return "Positive"
    else:
        return "Negative"

# Route for generating and serving pie chart image
@app.route('/plot')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

# Function to create pie chart
def create_figure():
    fig = Figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.01, 0.01)

    ax.pie(
        tags,
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode
    )
    
    ax.set_title("Sentiment Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("")

    return fig

# Running the Flask app
if __name__ == "__main__":
    app.run(port=5000, debug=True)
