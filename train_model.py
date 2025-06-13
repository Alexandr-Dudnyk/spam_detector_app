import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
  
import nltk
from nltk import word_tokenize
import string, re
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
nltk.download('wordnet')

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
