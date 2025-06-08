# 🧠 Emotion Detection using Text Classification

## 📊 Dataset Used
This project uses the **Emotion Dataset from Kaggle**, containing labeled text samples across the following emotion categories:

- 😠 Anger  
- 🤢 Disgust  
- 😨 Fear  
- 😊 Joy  
- 😢 Sadness  
- 😲 Surprise  
- 😐 Neutral  

## 🛠️ Approach Summary
The notebook applies a deep learning approach to detect emotions in text, involving the following steps:

1. **Data Preprocessing**
   - Tokenization and stopword removal  
   - Text normalization and sequence padding  

2. **Model Architecture**
   - **Embedding Layer**: Converts words to dense vectors  
   - **LSTM Layer**: Captures temporal dependencies  
   - **Dense Layer**: Outputs emotion probabilities using softmax  

3. **Training & Evaluation**
   - Loss Function: `categorical_crossentropy`  
   - Optimizer: `adam`  
   - Evaluation Metric: `accuracy`  

## 📦 Dependencies

Make sure the following Python libraries are installed:

```bash
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
keras
nltk

```

Install them using pip:

```bash
Copy
Edit
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras nltk
