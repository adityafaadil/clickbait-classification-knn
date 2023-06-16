import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud

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

    # Create a dropdown menu with page selection options
    page = st.sidebar.selectbox('Page', ['Halaman Utama', 'Klasifikasi', 'Dashboard'])

    if page == 'Halaman Utama':
        st.title('Clickbait Classification')
        # Display the description of the program
        st.write('Ini adalah program klasifikasi judul berita clickbait. Clickbait sendiri adalah praktik yang dilakukan untuk menarik perhatian pengguna internet dengan judul, gambar, atau deskripsi yang menjanjikan sesuatu yang menarik, mengejutkan, atau kontroversial, tetapi tidak selalu memberikan informasi yang relevan atau berkualitas saat pengguna mengkliknya. Tujuan utama clickbait adalah untuk mendapatkan sebanyak mungkin klik dan lalu lintas ke situs web tertentu dengan menggunakan metode manipulatif.')
        st.write('Program ini menggunakan data *CLICK-ID: A Novel Dataset for Indonesian Clickbait Headlines* dari situs Mendeley Data https://data.mendeley.com yang dikumpulkan dari 12 portal berita online lokal. Data ini berjumlah 15000 sample judul berita, dengan jumlah 6290 untuk judul clickbait dan 8710 untuk judul non-clickbait.')
        st.write('Algoritma yang digunakan dalam pemodelan ini yaitu _K-Nearest Neighbor_(KNN), dengan nilai k terbaik yaitu k=11 yang memiliki akurasi 72% dengan pembagian 80% data training dan 20% data testing.')
        
    elif page == 'Klasifikasi':
        st.title('Pengujian Klasifikasi')
        st.write('Ini adalah Pengujian Klasifikasi. Pengujian ini menggunakan model _K-Nearest Neighbor_(KNN) dengan menggunakan matrix _Euclidean Distance_ untuk mengklasifikasikan apakah judul berita tersebut clickbait atau tidak.')
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
        st.title('Tampilan Dashboard')
        df = pd.read_csv('dataset/data_bersih.csv')
        
        # top-level filters
        title_filter = st.selectbox("Select title", pd.unique(df["label"]))
        
        # dataframe filter
        df = df[df["label"] == title_filter]
        st.markdown("### Detailed Data View")
        st.dataframe(df)
        
        # Word Cloud
        st.markdown("### Word Cloud")
        wordcloud = WordCloud().generate(' '.join(df['title']))
        fig_wordcloud = px.imshow(wordcloud)
        fig_wordcloud.update_layout(title='Word Cloud')
        st.plotly_chart(fig_wordcloud)
    
        # Bar Chart
        st.markdown("### Bar Chart")
        bar_data = df['label'].value_counts()
        fig_bar = go.Figure(data=go.Bar(x=df['title'].index, y=df['label']))
        fig_bar.update_layout(title='Bar Chart')
        st.plotly_chart(fig_bar)
    
        # Pie Chart
        st.markdown("### Pie Chart")
        pie_data = df['label'].value_counts()
        fig_pie = go.Figure(data=go.Pie(labels=df['label'].index, values=pie_data))
        fig_pie.update_layout(title='Pie Chart')
        st.plotly_chart(fig_pie)
        
# Run the Streamlit app
if __name__ == '__main__':
    app()
