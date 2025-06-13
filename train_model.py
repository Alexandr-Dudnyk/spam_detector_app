import streamlit as st
import sklearn
import pickle
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.stem import PorterStemmer

port_stemmer = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Завантаження
df = pd.read_csv("ukr_sms_spam.csv")
X = df["text"]
y = df["label"]

# Train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Модель
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Оцінка
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Збереження
joblib.dump(model, "spam_detector_ukr.pkl")
print("Модель збережено у spam_detector_ukr.pkl")
