import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# load the KNN model and other artifacts
model = joblib.load('/model/knn_model.pkl')
vectorizer = joblib.load('/model/tfidf_vectorizer.pkl')
scaler = joblib.load('/model/scaler.pkl')

# create a function for prediction
def predict_clickbait(text_input):
    text_features = vectorizer.transform([text_input])
    text_features = scaler.transform(text_features.toarray())
    prediction = model.predict(text_features)

    return prediction[0]

# set up the app interface
st.title('Clickbait Classification with KNN')

# get user input and make prediction
text_input = st.text_input('Enter a clickbait title:')
if text_input != '':
    prediction = predict_clickbait(text_input)

    # display prediction result
    if prediction == 0:
        st.write('Not clickbait')
    else:
        st.write('Clickbait')
