# import library yang diperlukan
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

# load dataset clickbait
data = pd.read_csv('data_bersih.csv')

# memisahkan dataset menjadi data training dan data testing
train_data, test_data, train_label, test_label = train_test_split(data['title'], data['label'], test_size=0.2, random_state=42)

# melakukan vektorisasi teks pada dataset
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(train_data)
test_features = vectorizer.transform(test_data)

# melakukan standarisasi fitur
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features.toarray())
test_features = scaler.transform(test_features.toarray())

# melakukan cross validation untuk menentukan parameter k
k_range = range(1, 20)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, train_features, train_label, cv=5)
    k_scores.append(scores.mean())

# menentukan parameter k yang paling baik
best_k = k_range[k_scores.index(max(k_scores))]

# melatih model KNN dengan parameter k yang paling baik
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(train_features, train_label)

# mengevaluasi performa model
accuracy = knn.score(test_features, test_label)

# Define the title and subtitle of the app
st.title('Clickbait Classification with KNN')
st.write('Enter a clickbait title to classify')

# Create a text input for users to enter a title
text_input = st.text_input('Title')

# Create a button to classify the title
classify_button = st.button('Classify')

if classify_button:
    # Transform the user input using the vectorizer and scaler
    text_features = vectorizer.transform([text_input])
    text_features = scaler.transform(text_features.toarray())

    # Predict whether the title is clickbait or not
    prediction = knn.predict(text_features)

    # Display the prediction
    if prediction == 0:
        st.write('Not clickbait')
    else:
        st.write('Clickbait')
