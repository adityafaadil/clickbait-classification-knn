import streamlit as st
import joblib
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model
knn = joblib.load('model/model.joblib')

# Function to preprocess text data
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub('\d+', '', text)
    # Remove whitespace
    text = text.strip()
    return text

# definisikan vectorizer
vectorizer = TfidfVectorizer()

# Define function to classify text
def classify_text(text):
    text = preprocess_text(text)
    text = vectorizer.transform([text])
    prediction = knn.predict(text)[0]
    return prediction

# Define Streamlit app
def app():
    st.title('Klasifikasi Berita Clickbait')
    st.write('Gunakan model ini untuk menentukan apakah suatu artikel adalah clickbait atau tidak.')
    
    # Get user input
    text = st.text_input('Masukkan teks artikel:')
    
    # Convert the preprocessed input into a feature vector
    feature_vector = vectorizer.transform([text])
    
    # Classify text on button click
    if st.button('Klasifikasi'):
        if text:
            prediction = classify_text(text)
            st.write(f'Prediksi: {prediction}')
        else:
            st.warning('Masukkan teks artikel terlebih dahulu.')

# Run Streamlit app
app()
