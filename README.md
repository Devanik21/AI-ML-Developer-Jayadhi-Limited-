# AI/ML Developer(Jayadhi Limited)

Emotion Detection using Text Classification
Dataset Used
The model is trained and evaluated using the Emotion Dataset from Kaggle, which contains text samples labeled with seven distinct emotion categories:

Anger

Disgust

Fear

Joy

Sadness

Surprise

Neutral

Approach Summary
This project implements a text-based emotion detection model. The main steps include:

Data Preprocessing:

Tokenization and stopword removal

Text normalization and padding for input sequences

Model Architecture:

An Embedding layer for text representation

LSTM (Long Short-Term Memory) layers for sequential modeling

Dense output layer with softmax activation for multi-class emotion classification

Training and Evaluation:

The model is trained using categorical cross-entropy loss

Accuracy is used as the primary evaluation metric

Dependencies
To run this notebook, the following libraries are required:

bash
Copy
Edit
numpy  
pandas  
matplotlib  
seaborn  
scikit-learn  
tensorflow  
keras  
nltk  
Ensure that all dependencies are installed using pip or conda before execution.

