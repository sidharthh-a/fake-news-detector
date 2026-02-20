import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector for Students")

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained("fake_news_tokenizer")
    model = DistilBertForSequenceClassification.from_pretrained("fake_news_model")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

text_input = st.text_area("Paste news article or headline:", height=200)

if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("Enter some text.")
    else:
        inputs = tokenizer(
            text_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence = torch.max(probs).item() * 100
        prediction = torch.argmax(probs).item()

        if prediction == 1:
            st.success("✅ Real News")
        else:
            st.error("⚠️ Fake News")

        st.write(f"Confidence: {confidence:.2f}%")
        st.progress(confidence / 100)