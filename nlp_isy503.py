#installing necessary python packages and importing 

!pip install bs4 lxml 
!pip install tensorflow
!pip install numpy 
!pip install pandas

import os
import re
import string
from collections import Counter

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models

print("TensorFlow version:", tf._version_)

ROOT_DIR = "domain_sentiment_data/sorted_data_acl"  
print("Root exists:", os.path.exists(ROOT_DIR))
print("Subdirectories:", os.listdir(ROOT_DIR))

#gathering domains and positive/negative.review files

domain_files = {}

for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
    if not filenames:
        continue
    filenames_lower = [f.lower() for f in filenames]
    
    if "positive.review" in filenames_lower and "negative.review" in filenames_lower:
        # Map back to original case
        name_map = {f.lower(): f for f in filenames}
        pos_file = name_map["positive.review"]
        neg_file = name_map["negative.review"]
        
        domain_name = os.path.basename(dirpath)  # books, dvd, electronics, kitchen_&_housewares
        domain_files[domain_name] = {
            "dir": dirpath,
            "positive": os.path.join(dirpath, pos_file),
            "negative": os.path.join(dirpath, neg_file),
        }

print("Found domains and their files:\n")
for d, info in domain_files.items():
    print(f"Domain: {d}")
    print("  Positive:", info["positive"])
    print("  Negative:", info["negative"])
    print("-" * 60)

print("Available domains:", list(domain_files.keys()))

#loading reviews from an XML .review file

def load_xml_reviews(file_path):
    """Parse a .review XML file and return a list of review texts."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    soup = BeautifulSoup(raw, "lxml")
    
    review_nodes = soup.find_all("review_text")
    reviews = [node.get_text(strip=True) for node in review_nodes]
    
    #if no <review_text>, try generic <review> tags
    if not reviews:
        review_nodes = soup.find_all("review")
        reviews = [node.get_text(strip=True) for node in review_nodes]
    
    return reviews

# Quick check on one domain
sample_domain = next(iter(domain_files))
print("Testing load_xml_reviews on domain:", sample_domain)
test_pos = load_xml_reviews(domain_files[sample_domain]["positive"])
test_neg = load_xml_reviews(domain_files[sample_domain]["negative"])

print("Sample domain:", sample_domain)
print("  Positive reviews loaded:", len(test_pos))
print("  Negative reviews loaded:", len(test_neg))
print("Example positive review:\n", test_pos[0][:500], "\n")
print("Example negative review:\n", test_neg[0][:500])

#Loading all domains and build a combined DataFrame

records = []

for domain, paths in domain_files.items():
    # Positive
    pos_reviews = load_xml_reviews(paths["positive"])
    for txt in pos_reviews:
        records.append({
            "domain": domain,
            "label": 1,  # positive
            "label_text": "positive",
            "raw_text": txt
        })
    # Negative
    neg_reviews = load_xml_reviews(paths["negative"])
    for txt in neg_reviews:
        records.append({
            "domain": domain,
            "label": 0,  # negative
            "label_text": "negative",
            "raw_text": txt
        })

df = pd.DataFrame(records)
print("Total reviews loaded:", len(df))
print(df.head())

# data cleaning and basic dataset analysis

def clean_text(text):
    text = str(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"<.*?>", " ", text)           # removing the HTML tags from text
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # remove the URLs from text
    text = re.sub(r"\d+", " ", text)            # removeing numbers from the text
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text"] = df["raw_text"].apply(clean_text)
df["char_len"] = df["text"].apply(len)
df["word_count"] = df["text"].apply(lambda t: len(t.split()))

print("Overall information:\n")
print(df[["domain", "label", "char_len", "word_count"]].describe(include="all"))

print("\nClass balance:")
print(df["label"].value_counts())

print("\nPer domain counts:")
print(df.groupby("domain")["label"].value_counts().unstack(fill_value=0))

#Outlier removal – removing very short reviews

print("Before outlier removal:", df.shape)

MIN_WORDS = 3
df = df[df["word_count"] >= MIN_WORDS].reset_index(drop=True)

print("After outlier removal:", df.shape)

df["word_count"].describe()

#Training and validating dataset and test train split

X = df["text"].values
y = df["label"].values

# 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("Train size:", len(X_train))
print("Val size:", len(X_val))
print("Test size:", len(X_test))

#Tokenization and padding

MAX_VOCAB_SIZE = 20000
MAX_SEQ_LEN = 200

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq   = tokenizer.texts_to_sequences(X_val)
X_test_seq  = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
X_val_pad   = pad_sequences(X_val_seq,   maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
X_test_pad  = pad_sequences(X_test_seq,  maxlen=MAX_SEQ_LEN, padding="post", truncating="post")

y_train = np.array(y_train)
y_val   = np.array(y_val)
y_test  = np.array(y_test)

print("Padded train data shape:", X_train_pad.shape)
print("Padded val data shape:", X_val_pad.shape)
print("Padded test data shape:", X_test_pad.shape)

