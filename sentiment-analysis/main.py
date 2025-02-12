# Non-Deep Learning Approach: TF-IDF + Logistic Regression
import tensorflow_datasets as tfds
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Load IMDb dataset
dataset, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
train_data, test_data = dataset['train'], dataset['test']
train_texts, train_labels = zip(*[(str(text.numpy()), label.numpy()) for text, label in train_data])
test_texts, test_labels = zip(*[(str(text.numpy()), label.numpy()) for text, label in test_data])
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# TF-IDF Vectorizer and Logistic Regression
model = make_pipeline(TfidfVectorizer(max_features=10000), LogisticRegression())
model.fit(train_texts, train_labels)

# Evaluate Model
test_predictions = model.predict(test_texts)
accuracy = accuracy_score(test_labels, test_predictions)
print(f'Logistic Regression Accuracy: {accuracy:.4f}')

# Example Prediction
sample_review = test_texts[0]
tfidf_prediction = model.predict([sample_review])[0]
print(f'Sample Review: {sample_review}')
print(f'TF-IDF Sentiment Prediction: {"Positive" if tfidf_prediction == 1 else "Negative"}')
