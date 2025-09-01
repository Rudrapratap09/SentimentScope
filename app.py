from flask import Flask, render_template, request
import joblib
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Disables TensorFlow, uses PyTorch

from transformers import pipeline
bert_sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", revision="714eb0f")

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Set template_folder to the templates directory
app = Flask(__name__, template_folder='templates')

# Load model and Word2Vec
model = joblib.load('sentiment_model.pkl')
word2vec_model = Word2Vec.load('word2vec.model')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9 ]+', '', text)
    text = re.sub(r'(http|https|ftp|ssh)://\S+', '', text)
    text = BeautifulSoup(text, 'lxml').get_text()
    text = " ".join(text.split())
    text = " ".join([word for word in text.split() if word not in stop_words])
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return word_tokenize(text)

def avg_word2vec(tokens, model, vector_size=100):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

def get_bert_sentiment(text):
    max_length = 512
    result = bert_sentiment(text[:max_length])[0]
    return result['label'], result['score']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['review'].strip()
    if not text:
        print("Error: Empty review")
        return render_template('index.html', error="Please enter a review.")

    # Current model prediction
    tokens = preprocess(text)
    avg_vec = avg_word2vec(tokens, word2vec_model, 100).reshape(1, -1)
    current_prediction = model.predict(avg_vec)[0]
    current_sentiment = "Positive 😊" if current_prediction == 1 else "Negative 😞"

    # BERT prediction
    bert_label, bert_score = get_bert_sentiment(text)
    bert_sentiment_result = f"Positive 😊 (Confidence: {bert_score:.2f})" if bert_label == 'POSITIVE' else f"Negative 😞 (Confidence: {bert_score:.2f})"

    print(f"Input: {text}")
    print(f"Custom Model: {current_sentiment}")
    print(f"BERT: {bert_sentiment_result}")

    return render_template('index.html',
                          input_text=text,
                          model_prediction=current_sentiment,
                          bert_prediction=bert_sentiment_result)

if __name__ == '__main__':
    app.run(debug=True)
