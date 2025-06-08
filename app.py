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
    try:
        df = pd.read_csv("Emotion_classify_Data.csv")
    except FileNotFoundError:
        st.error("Error: 'Emotion_classify_Data.csv' not found. Please ensure the file is in the correct directory.")
        return pd.DataFrame(columns=['text', 'emotion']) # Return empty DataFrame with expected schema
    except pd.errors.EmptyDataError:
        st.error("Error: 'Emotion_classify_Data.csv' is empty.")
        return pd.DataFrame(columns=['text', 'emotion'])
    except Exception as e: # Catch other potential pandas read_csv errors
        st.error(f"Error reading 'Emotion_classify_Data.csv': {e}")
        return pd.DataFrame(columns=['text', 'emotion'])

    if df.empty:
        st.warning("'Emotion_classify_Data.csv' is empty or could not be parsed correctly.")
        return pd.DataFrame(columns=['text', 'emotion'])

    # Strip whitespace and convert column names to lowercase for consistency
    df.columns = [col.strip().lower() for col in df.columns]

    # Assume the first column is text and the second is emotion.
    # Rename them to 'text' and 'emotion' respectively.
    if len(df.columns) >= 2:
        df = df.rename(columns={df.columns[0]: 'text', df.columns[1]: 'emotion'})
    elif len(df.columns) == 1:
        # If only one column, assume it's 'text'. 'emotion' will be missing.
        df = df.rename(columns={df.columns[0]: 'text'})
        st.warning("CSV has only one column, assumed as 'text'. 'emotion' column is missing.")
        df['emotion'] = pd.NA # Add placeholder for 'emotion'
    else:
        st.error("CSV has no columns. Please check the file.")
        return pd.DataFrame(columns=['text', 'emotion'])

    return df

df = load_data()
st.subheader("📄 Dataset Sample")
st.write(df.head())

# 🧼 Preprocess
# Ensure 'text' column exists and handle potential missing values before applying clean_text
if 'text' not in df.columns:
    st.error("Critical error: 'text' column is missing from the DataFrame. Cannot proceed with preprocessing.")
    st.stop() # Stop execution if 'text' column isn't there

stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = re.sub(r"http\S+|@\S+|#\S+|[^A-Za-z0-9\s]", "", str(text))
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

df['text'] = df['text'].apply(clean_text)

# 🔢 Features & Labels
# Ensure 'emotion' column exists
if 'emotion' not in df.columns:
    st.error("Critical error: 'emotion' column is missing from the DataFrame. Cannot proceed with model training.")
    st.stop() # Stop execution

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
