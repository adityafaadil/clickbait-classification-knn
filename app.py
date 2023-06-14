import streamlit as st
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
    st.write('Masukkan judul berita untuk diklasifikasi apakah itu clickbait atau tidak.')
    
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

    # Show additional dashboard elements here
    st.subheader('Dashboard')
    
    # Example: Display a table of previously classified headlines
    previous_headlines = ['Judul 1', 'Judul 2', 'Judul 3']
    previous_predictions = [1, 0, 1]
    df = pd.DataFrame({'Judul Berita': previous_headlines, 'Prediksi Clickbait': previous_predictions})
    st.dataframe(df)

    # Example: Show a bar chart of clickbait vs. non-clickbait headlines
    labels = ['Clickbait', 'Non-Clickbait']
    values = [2, 1]
    chart_data = pd.DataFrame({'Label': labels, 'Value': values})
    st.bar_chart(chart_data['Value'], labels=chart_data['Label'])

    # Example: Display the most common words in the previous headlines
    word_frequency = {'clickbait': 10, 'berita': 8, 'judul': 5, 'bukan': 4, 'ini': 3}
    st.subheader('Word Frequency')
    st.write(word_frequency)

# Run the Streamlit app
if __name__ == '__main__':
    app()
