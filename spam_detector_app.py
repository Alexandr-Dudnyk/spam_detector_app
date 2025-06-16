import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Завантаження моделі BERT (багатомовна, підтримує українську)
@st.cache_resource
def load_model():
    model_name = "cointegrated/rubert-tiny-toxicity"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Інтерфейс
st.set_page_config(page_title="BERT Spam/Toxic Detector", page_icon="🤖")
st.title("🤖 Виявлення спаму/токсичності (BERT, українська мова)")
st.markdown("Введіть повідомлення українською для класифікації за допомогою BERT-моделі.")

user_input = st.text_area("✍️ Введіть повідомлення:", height=150)

def classify(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.sigmoid(outputs.logits).squeeze().numpy()
    labels = model.config.id2label
    return {labels[i]: float(scores[i]) for i in range(len(scores))}

if user_input:
    results = classify(user_input)
    st.subheader("🔍 Результати класифікації:")
    for label, score in results.items():
        st.write(f"**{label}**: {score:.2f}")
        st.progress(score)

    if results.get("spam", 0) > 0.5 or results.get("toxic", 0) > 0.5 or results.get("obscene", 0) > 0.5:
        st.error("⚠️ Повідомлення може бути спамом або токсичним.")
    else:
        st.success("✅ Повідомлення виглядає безпечним.")
