import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from pathlib import Path
import streamlit as st

st.title("Researchly Research Abstract Classifier")
st.write("This app classifies research abstracts as either AI-related or classical research")

np.random.seed(42)

#Read dataset
# Locate the main project directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Read dataset
df = pd.read_json(BASE_DIR / "Datasets" / "research_abstracts.json")

#Drop missing values
df = df.dropna()

nltk.download('punkt_tab')
stop_words = set(stopwords.words("english"))

#Preprocess function
def clean_text(text):
    #Remove punctuation, special characters, numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    #Lowercase
    text = text.lower()

    #Tokenize
    tokens = word_tokenize(text)

    #Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    text = " ".join(tokens)
    
    return text

df["Clean Abstract"] = df["Abstract"].apply(clean_text)

#Clean dataset
clean_df = df[["Clean Abstract", "Label"]].copy()

X = clean_df["Clean Abstract"]
y = clean_df["Label"]

tfidf = TfidfVectorizer()
#Train test split 80-20
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#Feature selection on train and test set
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

#Initialise model
model = LinearSVC(C=1.0)

#Train model
model.fit(X_train_tfidf, y_train)

#For results output
label_mapping = {
    "ai": "artificial intelligence",
    "classical": "classical research"
}



# Abstract input
abstract_in = st.text_area("Enter research abstract:")

# Classification
if st.button("Classify Abstract"):

    if abstract_in.strip():

        # Preprocess abstract
        clean_abstract = clean_text(abstract_in)

        # Apply TF-IDF transformation
        abstract_tfidf = tfidf.transform([clean_abstract])

        # Predict using SVM
        prediction = model.predict(abstract_tfidf)

        predicted_label = label_mapping.get(prediction[0], prediction[0])

        st.success(
            f"The abstract is classified as: {predicted_label}"
        )

    else:
        st.warning("Please enter a research abstract.")
