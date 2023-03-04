import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load model
model = joblib.load('model/clickbait_knn_model.joblib')

# Define TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Define Standard Scaler
scaler = StandardScaler()

# Set app title
st.title('Klasifikasi Berita Clickbait')

# Get input from user
input_text = st.text_input('Masukkan judul berita')

# Make prediction
if st.button('Predict'):
    # Process input text
    input_text = [input_text]
    input_vec = vectorizer.transform(input_text)
    input_scaled = scaler.transform(input_vec)
    
    # Make prediction using model
    prediction = model.predict(input_scaled)
    
    # Display prediction
    if prediction == 1:
        st.write('Judul berita ini termasuk clickbait')
    else:
        st.write('Judul berita ini tidak termasuk clickbait')
