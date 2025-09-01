
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

#  Load and clean data
df = pd.read_csv('all_kindle_review .csv')[['reviewText', 'rating']]
df.dropna(inplace=True)

# Binary sentiment: 1 for positive, 0 for negative
df['rating'] = df['rating'].apply(lambda x: 0 if x < 3 else 1)

# Lowercase text
df['reviewText'] = df['reviewText'].str.lower()

# Clean ext
df['reviewText'] = df['reviewText'].apply(lambda x: re.sub('[^a-zA-Z0-9 ]+', '', str(x)))
df['reviewText'] = df['reviewText'].apply(lambda x: re.sub(r'(http|https|ftp|ssh)://\S+', '', x))
df['reviewText'] = df['reviewText'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x.split()))

# Remove stopwords
stop_words = set(stopwords.words('english'))
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join([word for word in x.split() if word not in stop_words]))

# Lemmatize
lemmatizer = WordNetLemmatizer()
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['reviewText'], df['rating'], test_size=0.20, random_state=42)

# 4. Word2Vec training
X_train_tokens = [word_tokenize(review) for review in X_train]
X_test_tokens = [word_tokenize(review) for review in X_test]

word2vec_model = Word2Vec(sentences=X_train_tokens, vector_size=100, window=5, min_count=2, workers=4, sg=1)

def avg_word2vec(tokens, model, vector_size=100):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

X_train_avg = np.array([avg_word2vec(tokens, word2vec_model) for tokens in X_train_tokens])
X_test_avg = np.array([avg_word2vec(tokens, word2vec_model) for tokens in X_test_tokens])

# 5. Train with Logistic Regression + GridSearch
param_grid = {'C': [0.01, 0.1, 1, 10]}
grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train_avg, y_train)

best_model = grid_search.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test_avg)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# 6. Save the model and Word2Vec
joblib.dump(best_model, 'sentiment_model.pkl')
word2vec_model.save('word2vec.model')
print("✅ Model and Word2Vec saved!")
