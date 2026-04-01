import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

model = MultinomialNB()
model.fit(X, y)

# UI
st.title("📩 Spam Message Classifier")

msg = st.text_input("Enter your message")

if st.button("Check"):
    if msg.strip() != "":
        msg_vec = vectorizer.transform([msg])
        result = model.predict(msg_vec)

        if result[0] == 1:
            st.error("Spam 🚨")
        else:
            st.success("Not Spam ✅")
    else:
        st.warning("Please enter a message")