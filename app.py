import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

# load model
with open('model/clickbait_knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# load TF-IDF vectorizer
with open('model/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# define function for predicting clickbait
def predict_clickbait(title):
    # transform text using tf-idf vectorizer
    title_tfidf = tfidf_vectorizer.fit_transform([title])
    # predict clickbait using trained model
    prediction = model.predict(title_tfidf)
    # return prediction
    return prediction[0]

# set up streamlit app
st.title('Clickbait Classifier')

# get user input
title = st.text_input('Enter the title of the article:')
if st.button('Predict'):
    # predict clickbait
    prediction = predict_clickbait(title)
    # display prediction
    if prediction == 1:
        st.write('This article is likely to be clickbait.')
    else:
        st.write('This article is not likely to be clickbait.')
