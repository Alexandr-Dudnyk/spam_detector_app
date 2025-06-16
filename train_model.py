import streamlit as st
import joblib

# Завантаження моделі
@st.cache_resource
def load_model():
    return joblib.load("spam_detector_ukr.pkl")

model = load_model()

# Інтерфейс
st.set_page_config(page_title="UkrSpamDetector", page_icon="📩")
st.title("📩 Виявлення СПАМу (українська мова)")
st.markdown("Введіть текст повідомлення, щоб перевірити, чи є воно спамом.")

# Ввід користувача
user_input = st.text_area("✍️ Введіть повідомлення:", height=150)

if user_input:
    prediction = model.predict([user_input])[0]
    probas = model.predict_proba([user_input])[0]
    spam_prob = round(probas[model.classes_.tolist().index("spam")], 2)

    st.subheader("🔍 Результат:")
    if prediction == "spam":
        st.error(f"❌ Це СПАМ (ймовірність: {spam_prob})")
    else:
        st.success(f"✅ Це не спам (ймовірність спаму: {spam_prob})")

    with st.expander("📊 Деталі"):
        st.json({
            "Класи": list(model.classes_),
            "Ймовірності": {cls: round(prob, 3) for cls, prob in zip(model.classes_, probas)}
        })
