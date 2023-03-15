import pickle
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

# Load model
with open('model/knn_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load CountVectorizer
with open('model/vectorizer.pkl', 'rb') as file:
    cv = pickle.load(file)

# Create Streamlit app
st.title("Clickbait News Classification")

# Create input form
input_text = st.text_input("Input news title")

# Create predict button
if st.button("Predict"):
    # Transform input_text into Bag of Words
    input_text_bow = cv.transform([input_text])
    # Predict
    prediction = model.predict(input_text_bow)[0]
    if prediction == 1:
        st.write("This news title is clickbait")
    else:
        st.write("This news title is not clickbait")
