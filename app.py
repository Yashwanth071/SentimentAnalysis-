from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

import json
import re
import string
import os

# -----------------------------
# Configuration
# -----------------------------

# Path 
MODEL_PATH = "sentiment_model\sentiment_model.h5"
TOKENIZER_PATH = "sentiment_model\tokenizer.json"

# This value should match training value
MAX_SEQ_LEN = 200

# -----------------------------
# Text cleaning 
# -----------------------------

def clean_text(text: str) -> str:
    text = str(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"<.*?>", " ", text)           # It remove the HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # It remove the URLs
    text = re.sub(r"\d+", " ", text)            # It remove the numbers
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# Loading the model and tokenizing
# -----------------------------

# Loading Keras model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file does not found: {MODEL_PATH}")

model = load_model(MODEL_PATH)

# Loading tokenizer from JSON file
if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"Tokenizer file does not found: {TOKENIZER_PATH}")

with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
    tokenizer_json = f.read()        # reading as string rather than json

tokenizer = tokenizer_from_json(tokenizer_json)


# -----------------------------
# Prediction helper functions
# -----------------------------

def preprocess_single_review(review_text: str):
    cleaned = clean_text(review_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
    return pad

def predict_sentiment(review_text: str, threshold: float = 0.5):
    pad = preprocess_single_review(review_text)
    prob = model.predict(pad)[0][0]
    label = "Positive review" if prob >= threshold else "Negative review"
    return label, float(prob)
# -----------------------------
# Flask application
# -----------------------------

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("review_text", "")
        if input_text.strip():
            label, prob = predict_sentiment(input_text)
            prediction = label
            probability = prob

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        input_text=input_text
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
