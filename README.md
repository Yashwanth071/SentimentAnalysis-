________________________________________
SentimentAnalysis — ISY503 Final Project (NLP Sentiment Analysis)
Project: ISY503 — Final Project (Assessment 3)
Group: Yashwanth, Tharun, Vani, Annudogu
Repository: https://github.com/Yashwanth071/SentimentAnalysis-.git
________________________________________
Overview
This repository contains an end-to-end NLP sentiment analysis project built for ISY503. The project:

●	Uses the JHU multi-domain Amazon reviews dataset (books, dvd, electronics, kitchen & housewares).

●	Implements a BiLSTM-based sentiment classifier (binary positive/negative).

●	Provides preprocessing, training and evaluation notebooks/scripts.

●	Provides a simple Flask web app (app.py + templates/index.html) to demo the model with a text box that outputs "Positive review" or "Negative review" and a confidence score.

The final deliverable includes code, a saved model and tokenizer, presentation slides, and individual reports.
________________________________________
Repo structure (key files)
SentimentAnalysis-/
├─ app.py                     # Flask web app (backend)
├─ templates/
│  └─ index.html              # Frontend UI 
├─ sentiment_model.h5         # Saved Keras model 
├─ tokenizer.json             # Keras Tokenizer serialized
├─ nlp_isy503.py              # Training / data-prep script 
├─ README.md                  


________________________________________
Quick results (from local run)

●	Example inference outputs:

○	This product was amazing, I loved it and will buy again! → Positive review (score ≈ 0.9887)

○	Terrible quality, completely useless and a waste of money. → Negative review (score ≈ 0.0)

○	It was okay. Not great, not terrible, just average. → Negative review (score ≈ 0.002)

________________________________________
Requirements
Create and use a virtual environment (recommended).
# Create venv
python -m venv .venv
# Activate
.venv\Scripts\Activate.ps1   # PowerShell
# or
.venv\Scripts\activate.bat   # cmd.exe

Install packages (example):
pip install tensorflow bs4 lxml numpy pandas scikit-learn matplotlib jupyter

Dataset
This project uses the JHU domain sentiment dataset. 

If you need to download the dataset manually, you can find it from the JHU NLP distributions (search for "domain sentiment dataset JHU" online) and extract it.

This project uses the JHU domain sentiment dataset. 
Total samples loaded: 8000 (2000 per domain; balanced positive/negative)
Train / Val / Test split: 5599 / 1200 / 1200
dataset folder contains books, dvd, electronics, kitchen_&_housewares subfolders each containing positive.review and negative.review.
________________________________________

________________________________________
**How to run the Flask demo locally**
1. Prepare dataset: ensure domain_sentiment_data or the dataset folder is present and contains books, dvd, electronics, kitchen_&_housewares subfolders each containing positive.review and negative.review.
2. Ensure sentiment_model.h5 and tokenizer.json are in the same folder as app.py (update the paths in app.py & nlp_isy503.py).
3. python run nlp_isy503.py
4. python run app.py
5. click on http://127.0.0.1:5000 or open it in browser
6. Enter review text.
7. Click Analyze to get result
________________________________________

Team Contributions (for Assessment / Report)

●	Yashwanth — UI, index.html, commit ,fixes minor issues & repo maintenance — 25%

●	Tharun — Data parsing, cleaning, preprocessing, tokenizer — 25%

●	Vani — Model architecture (BiLSTM), training, evaluation — 25%

●	Annudogu — Flask backend finalization, integration, deployment testing — 25%
________________________________________
References (select):
●	Blitzer, J., Dredze, M., & Pereira, F. (2007). Biographies and multi-domain sentiment dataset (JHU domain sentiment dataset).

●	Gebru, T., et al. (2018). Datasheets for Datasets.

●	Mitchell, M., et al. (2019). Model Cards for Model Reporting.



