import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pickle
import streamlit as st

# Load dataset
df = pd.read_csv("data/data_bersih.csv")

# Split dataset into X and y
X = df["title"]
y = df["label_score"]

# Create Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

# Split dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create model KNN dengan cross-validation nilai k nya dari 1 sampai 11
scores = []
for k in range(1, 12):
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train, y_train, cv=10)
    scores.append(score.mean())

# Find the best k
best_k = np.argmax(scores) + 1
st.write("Best k: ", best_k)

# Create model KNN dengan nilai k terbaik
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)

# Save model
with open('clickbait_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Load model
with open('clickbait_model.pkl', 'rb') as file:
    model = pickle.load(file)

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

# Show accuracy
accuracy = model.score(X_test, y_test)
st.write("Accuracy: ", accuracy)
