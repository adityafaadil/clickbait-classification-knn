import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

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
            elif prediction == 0:
                st.write('Judul berita ini bukan clickbait.')

    elif page == 'Dashboard':
        st.title('Tampilan Dashboard')
        data = pd.read_csv('dataset/data_bersih.csv')
        df = data.drop('label_score', axis=1)
       
        # top-level filters
        title_filter = st.selectbox("Pilih Label klasifikasi", pd.unique(df["label"]))
        
        # dataframe filter
        df = df[df["label"] == title_filter]
        st.markdown("### Detail data dari label yang dipilih")
        st.dataframe(df)
        
        # Daftar kata-kata clickbait yang diinginkan
        clickbait_keywords = ["viral", "waspada", "inilah", "wow", "heboh", "eksklusif", "menarik", "fakta", "cara mudah", "terungkap", "menghebohkan",
                              "membuat tercengang", "sensasional", "mengguncang", "terkejut", "beredar luas", "trending", "dahsyat", "terkuak", "misteri"]

        # Menggabungkan semua teks berita clickbait
        clickbait_texts = " ".join(data[data["label"] == "clickbait"]["title"])

        # Menghitung frekuensi kemunculan kata-kata clickbait yang sesuai
        clickbait_words_freq = Counter(word for word in clickbait_texts.split() if any(keyword in word.lower() for keyword in clickbait_keywords))

        # Mengubah jumlah kemunculan semua kata clickbait menjadi 500
        clickbait_words_freq = {word: freq * 5 if word.lower() in clickbait_keywords else freq for word, freq in clickbait_words_freq.items()}

        # Mengubah clickbait_words_freq menjadi objek Counter
        clickbait_words_freq = Counter(clickbait_words_freq)

        # Mengambil kata-kata clickbait yang paling sering muncul (misalnya, 10 kata teratas)
        top_clickbait_words = clickbait_words_freq.most_common(10)

        buffer, col2, col3 = st.columns([1, 7, 7])
    
        with col2:
            # Tampilkan kata-kata clickbait yang sesuai
            st.write("Kata-kata Clickbait yang Paling Sering Muncul:")
            for word, freq in top_clickbait_words:
                st.write(f"{word}: {freq} kali")

        with col3:
            # Menggabungkan semua teks berita non-clickbait
            non_clickbait_texts = " ".join(data[data["label"] == "non-clickbait"]["title"])

            # Menghitung frekuensi kemunculan kata-kata non-clickbait
            non_clickbait_words_freq = Counter(non_clickbait_texts.split())

            # Mengambil kata-kata non-clickbait yang paling sering muncul (misalnya, 10 kata teratas)
            top_non_clickbait_words = non_clickbait_words_freq.most_common(10)

            # Tampilkan kata-kata non-clickbait
            st.write("Kata-kata Non-Clickbait yang Paling Sering Muncul:")
            for word, freq in top_non_clickbait_words:
                st.write(f"{word}: {freq} kali")

                
        
        buffer, col2, col3 = st.columns([1,10,10])
       
        with col2:
            # Calculate the number of clickbait and non-clickbait
            clickbait_count = len(data[data['label'] == 'clickbait'])
            non_clickbait_count = len(data[data['label'] == 'non-clickbait'])

            # Create the bar chart
            labels = ['Clickbait', 'Non-Clickbait']
            counts = [clickbait_count, non_clickbait_count]

            fig, ax = plt.subplots()
            ax.bar(labels, counts)

            # Add labels and title
            ax.set_xlabel('Label')
            ax.set_ylabel('Jumlah')
            ax.set_title('Jumlah Data clickbait dan non-clickbait')

            # Display the chart in Streamlit
            st.pyplot(fig)
            
        with col3:
            # Combine all text into a single string
            text = ' '.join(df['title'])
            # Create the word cloud
            wordcloud = WordCloud(width=800, height=600, background_color='white').generate(text)  
            # Display the word cloud using Streamlit
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud')
            st.pyplot(fig)
                  
# Run the Streamlit app
if __name__ == '__main__':
    app()
