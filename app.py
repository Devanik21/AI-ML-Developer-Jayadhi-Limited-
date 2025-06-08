import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

nltk.download('stopwords')

# 💖 Page setup
st.set_page_config(page_title="Emotion Detector", page_icon="🔍")
st.title("🧠 Real-time Emotion Classifier using Random Forest")

# 📂 Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Emotion_classify_Data.csv")
    df = df.rename(columns=lambda x: x.strip())
    return df

df = load_data()
st.subheader("📄 Dataset Sample")
st.write(df.head())

# 🧼 Preprocess
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = re.sub(r"http\S+|@\S+|#\S+|[^A-Za-z0-9\s]", "", str(text))
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

df['text'] = df['text'].apply(clean_text)

# 🔢 Features & Labels
X = df['text']
y = df['emotion'].astype(str)

# ✨ TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# 🎓 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# 🌲 Random Forest Training
st.subheader("🌲 Training Random Forest Classifier")
with st.spinner("Training... Please wait ⏳"):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

# 📈 Accuracy
accuracy = accuracy_score(y_test, y_pred)
st.success(f"✅ Model Accuracy: {accuracy:.2f}")

# 📊 Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', xticklabels=clf.classes_, yticklabels=clf.classes_, cmap='Purples', ax=ax)
st.subheader("📌 Confusion Matrix")
st.pyplot(fig)

# ✍️ User Prediction
st.subheader("💬 Try Your Own Text")
user_input = st.text_input("Enter a sentence to analyze emotion:")

if user_input:
    user_clean = clean_text(user_input)
    user_vec = vectorizer.transform([user_clean])
    pred = clf.predict(user_vec)[0]
    st.success(f"🔮 Predicted Emotion: **{pred}** 💖")
