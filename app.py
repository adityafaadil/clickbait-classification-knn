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

# Read the data from CSV
data = pd.read_csv('/dataset/data_bersih.csv')

# Set the page title and layout
st.set_page_config(page_title='Clickbait Dashboard', layout='wide')

# Display the dataset
st.title('Clickbait Dashboard')
st.subheader('Data')
st.dataframe(data)

# Show statistics
st.subheader('Statistics')
st.write('Total number of headlines:', len(data))
st.write('Number of clickbait headlines:', len(data[data['label'] == 1]))
st.write('Number of non-clickbait headlines:', len(data[data['label'] == 0]))

# Show bar chart of label distribution
st.subheader('Label Distribution')
label_counts = data['label'].value_counts()
plt.bar(label_counts.index, label_counts.values)
plt.xlabel('Label')
plt.ylabel('Count')
st.pyplot(plt)

# Show word cloud of headlines
from wordcloud import WordCloud
clickbait_text = ' '.join(data[data['label'] == 1]['headline'])
non_clickbait_text = ' '.join(data[data['label'] == 0]['headline'])

st.subheader('Word Cloud')
st.write('Clickbait Headlines')
wordcloud = WordCloud(width=800, height=400).generate(clickbait_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)

st.write('Non-Clickbait Headlines')
wordcloud = WordCloud(width=800, height=400).generate(non_clickbait_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)

# Run the Streamlit app
if __name__ == '__main__':
    app()
