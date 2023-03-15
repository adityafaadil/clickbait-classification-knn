import streamlit as st
import joblib

# Load model
knn = joblib.load('model/model.joblib')

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
    
    # Classify text on button click
    if st.button('Klasifikasi'):
        if text:
            prediction = classify_text(text)
            st.write(f'Prediksi: {prediction}')
        else:
            st.warning('Masukkan teks artikel terlebih dahulu.')

# Run Streamlit app
app()
