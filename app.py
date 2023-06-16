import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import re
import string

# Load the pre-trained model and vectorizer
model = joblib.load('model/model.joblib')
vectorizer = joblib.load('model/vectorizer.joblib')

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

# Define the Streamlit app
def app():
    st.set_page_config(page_title='Clickbait Classification', page_icon=':newspaper:', layout='wide')
    st.title('Clickbait Classification')

    # Create a dropdown menu with page selection options
    page = st.sidebar.selectbox('Page', ['Halaman Utama', 'Klasifikasi', 'Dashboard'])

    if page == 'Halaman Utama':
        # Display the description of the program
        st.write('This is a clickbait classification program. It uses a pre-trained model to classify whether a given news headline is clickbait or not.')

    elif page == 'Klasifikasi':
        st.write('Ini adalah program klasifikasi clickbait. Program ini menggunakan model KNN untuk mengklasifikasikan apakah judul berita tersebut clickbait atau tidak.')
        # Create a text input box for the user to enter a headline
        user_input = st.text_input('Judul berita:')
        
        # Classify the headline if the user presses the 'Klasifikasi' button
        if st.button('Klasifikasi'):
            # Preprocess the user input
            preprocessed_input = preprocess_text(user_input)
            # Convert the preprocessed input into a feature vector
            feature_vector = vectorizer.transform([preprocessed_input])
            # Make a prediction using the pre-trained model
            prediction = model.predict(feature_vector)[0]
            # Display the prediction to the user
            if prediction == 1:
                st.write('Judul berita ini clickbait.')
            else:
                st.write('Judul berita ini bukan clickbait.')

    elif page == 'Dashboard':
        st.write('This is the About page. Here you can provide information about the project.')

# Run the Streamlit app
if __name__ == '__main__':
    app()
