import streamlit as st
import joblib
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Ensure NLTK data is available
nltk.download('stopwords')
nltk.download('punkt')

# Load pre-trained models
vectorizer = joblib.load("tfidf_model.pkl")
model = joblib.load("tf_logistic_model.pkl")

# Function for preprocessing text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r"[^a-z\s]", "", text)
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word not in stop_words]
    # Join words back into a single string
    return " ".join(filtered_words)

# App title
st.title("Sentiment Analysis App")
st.write("This app uses TF-IDF and Logistic Regression to classify reviews as positive or negative.")

# Input text area
input_text = st.text_area("Enter a review:", "")

# Prediction
if st.button("Analyze Sentiment"):
    if input_text.strip():
        # Preprocess the input text
        processed_text = preprocess_text(input_text)
        
        # Preprocess and vectorize input
        input_vectorized = vectorizer.transform([processed_text])

        # Predict sentiment
        prediction = model.predict(input_vectorized)

        # Display result
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        st.subheader(f"Sentiment: {sentiment}")
    else:
        st.error("Please enter some text to analyze.")
