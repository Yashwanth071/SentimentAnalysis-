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
