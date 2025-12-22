import streamlit as st
import numpy as np
import pickle
import os
import google.generativeai as genai
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load model and artifacts
model = load_model("ticket_classifier.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Gemini setup
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# Text cleaning
def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Prediction
def predict_queue(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
    pred = model.predict(pad)
    label = encoder.inverse_transform([np.argmax(pred)])
    return label[0]

# Gemini reply generation
def generate_reply(ticket_text, predicted_queue):
    prompt = f"""
    A customer submitted a support ticket: "{ticket_text}"
    It was classified under the '{predicted_queue}' department.
    Write a polite, empathetic reply acknowledging the issue and assuring support.
    """
    response = gemini_model.generate_content(prompt)
    return response.text

# Streamlit UI
st.title("Ticket Classifier + Gemini Reply Generator")

ticket_input = st.text_area("Paste customer ticket text here:", height=200)

if st.button("Classify and Generate Reply"):
    if ticket_input.strip():
        queue = predict_queue(ticket_input)
        reply = generate_reply(ticket_input, queue)

        st.subheader("Predicted Queue:")
        st.success(queue)

        st.subheader("Generated Reply:")
        st.write(reply)
    else:
        st.warning("Please enter a ticket message to proceed.")