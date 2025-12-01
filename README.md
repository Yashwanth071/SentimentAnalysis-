
# **SentimentAnalysis — ISY503 Final Project (NLP Sentiment Analysis)**

**Project:** ISY503 — Final Project (Assessment 3)
**Group:** Yashwanth, Tharun, Vani, Annudogu

**Repository:** [https://github.com/Yashwanth071/SentimentAnalysis-.git](https://github.com/Yashwanth071/SentimentAnalysis-.git)

---

## **📌 Overview**

This repository contains an end-to-end **NLP Sentiment Analysis** project developed for ISY503. The project:

* Uses the **JHU multi-domain Amazon reviews dataset** (Books, DVD, Electronics, Kitchen & Housewares).
* Implements a **BiLSTM-based sentiment classifier** for binary classification (Positive/Negative).
* Includes **data preprocessing**, **model training**, **evaluation**, and **inference scripts**.
* Features a **Flask web app** (`app.py`) with a simple input interface (`index.html`) to classify user-provided reviews with sentiment label and confidence score.

The final deliverables include:
✔ Complete codebase
✔ Trained model (`sentiment_model.h5`) and tokenizer
✔ Flask web demo
✔ Presentation slides
✔ Individual reports

---

## **📁 Repository Structure**

```
SentimentAnalysis-/
├─ app.py                     # Flask web application (backend)
├─ templates/
│  └─ index.html              # Frontend UI template
├─ sentiment_model.h5         # Saved trained Keras model
├─ tokenizer.json             # Serialized Keras tokenizer
├─ nlp_isy503.py              # Data preprocessing & model training script
├─ README.md                  # Project documentation
```

---

## **🧪 Sample Prediction Results** (From Local Execution)

| Input Review Example                                         | Predicted Sentiment | Confidence Score |
| ------------------------------------------------------------ | ------------------- | ---------------- |
| *This product was amazing, I loved it and will buy again!*   | Positive            | 0.9887           |
| *Terrible quality, completely useless and a waste of money.* | Negative            | 0.0004           |
| *It was okay. Not great, not terrible, just average.*        | Negative            | 0.0020           |

---

## **🛠 Requirements**

### **▶ Create and activate virtual environment (Recommended)**

```bash
# Create virtual environment
python -m venv .venv

# Activate (PowerShell)
.venv\Scripts\Activate.ps1

# Activate (CMD)
.venv\Scripts\activate.bat
```

### **▶ Install dependencies**

```bash
pip install tensorflow bs4 lxml numpy pandas scikit-learn matplotlib jupyter flask
```

---

## **📂 Dataset Details**

This project uses the **JHU Domain Sentiment Dataset**, containing balanced positive and negative reviews across **four domains**:

📌 Domains included:
📘 Books | 💿 DVD | 💻 Electronics | 🍽 Kitchen & Housewares

**Dataset Structure** (inside `dataset/` or `domain_sentiment_data/`):

```
books/
 ├─ positive.review
 └─ negative.review
dvd/
 ├─ positive.review
 └─ negative.review
electronics/
 ├─ positive.review
 └─ negative.review
kitchen_&_housewares/
 ├─ positive.review
 └─ negative.review
```

**Data Summary:**

* Total Samples: 8000 (2000 per domain)
* Balanced labels: Positive / Negative
* Train / Validation / Test Split: **5599 / 1200 / 1200**

---

## **🚀 How to Run the Flask Demo Locally**

1️⃣ Place the dataset folder (`domain_sentiment_data/`) properly with all domain folders.
2️⃣ Ensure `sentiment_model.h5` and `tokenizer.json` are in the same directory as `app.py`.
3️⃣ Train or load the model:

```bash
python nlp_isy503.py
```

4️⃣ Run the Flask app:

```bash
python app.py
```

5️⃣ Open in browser:
🔗 [http://127.0.0.1:5000](http://127.0.0.1:5000)
6️⃣ Enter a review text
7️⃣ Click **Analyze** to see sentiment and confidence score

---

## **👥 Team Contributions (Assessment Purpose)**

| Team Member | Contribution                                                    | Percentage |
| ----------- | --------------------------------------------------------------- | ---------- |
| Yashwanth   | UI Design, `index.html`, commits, fixes, repository management  | 25%        |
| Tharun      | Data parsing, cleaning, preprocessing, tokenizer implementation | 25%        |
| Vani        | Model architecture (BiLSTM), training, evaluation               | 25%        |
| Annudogu    | Flask backend, integration, deployment testing                  | 25%        |

---

## **📚 References**

* Blitzer, J., Dredze, M., & Pereira, F. (2007). *Biographies and multi-domain sentiment dataset (JHU dataset).*
* Gebru, T., et al. (2018). *Datasheets for Datasets.*
* Mitchell, M., et al. (2019). *Model Cards for Model Reporting.*

---

