import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import string


# Load model
@st.cache
def load_model():
    with open('model/knn_model.pkl', 'rb') as file:
        model = joblib.load(file)
    return model

# Main function
def main():
    # Set page title and favicon
    st.set_page_config(page_title='Klasifikasi Berita Clickbait', page_icon=':newspaper:', layout='wide')

    # Load model
    model = load_model()

    # Display title and subtitle
    st.title('Klasifikasi Berita Clickbait')
    st.markdown('Ini adalah aplikasi untuk melakukan klasifikasi berita clickbait')

    # Display input form
    input_text = st.text_input('Masukkan judul berita:')

    # Predict class and display result
    if input_text:
        input_data = [[input_text]]
        prediction = model.predict(input_data)[0]

        # Display result
        st.markdown('---')
        st.subheader('Hasil klasifikasi')
        if prediction == 0:
            st.error('Berita ini **bukan** clickbait')
        else:
            st.success('Berita ini **adalah** clickbait')

# Run main function
if __name__ == '__main__':
    main()
