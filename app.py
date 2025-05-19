# app.py

import streamlit as st
import re
import numpy as np
import joblib

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load models
with open("w2v_model.model", "rb") as f:
    model = joblib.load(f)

with open("rf_model.joblib", "rb") as f:
    reg = joblib.load(f)

# NLP Preprocessing
ps = PorterStemmer()

def prep(message):
    clean_message = re.sub('[^a-zA-Z0-9]', ' ', message)
    lower_sms = clean_message.lower()
    splited = lower_sms.split()
    new_message = [ps.stem(word) for word in splited if word not in stopwords.words('english')]
    return new_message

def sentence_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# Streamlit UI
st.title("📚 Kindle Review Sentiment Prediction")
user_input = st.text_area("Enter a Kindle review here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        tokens = prep(user_input)
        vector = sentence_vector(tokens, model).reshape(1, -1)
        prediction = reg.predict(vector)[0]

        if prediction == 1:
            st.success("✅ Positive review")
        else:
            st.error("❌ Negative review")
