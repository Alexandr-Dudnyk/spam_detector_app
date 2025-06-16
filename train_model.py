import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import joblib

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
