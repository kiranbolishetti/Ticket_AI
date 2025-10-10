import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import os
from google import genai
from google.genai.errors import APIError

#STREAMLIT CONFIGURATION

st.set_page_config(
    page_title="Ticket.AI: Automated Support System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Ticket.AI: Automated Support System")
st.markdown("Automate ticket classification and response generation using AI.")

#CONSTANTS & PATHS

MAX_LEN = 100
MODEL_PATH = "ticket_classification_rnn_model.h5"
TOKENIZER_PATH = "tokenizer.pickle"
LABEL_ENCODER_PATH = "label_encoder.pickle"
GEMINI_MODEL = "gemini-2.5-flash"

#LOAD ASSETS (MODEL, TOKENIZER, LABEL ENCODER)

@st.cache_resource
def load_assets():
    try:
        # Load Model
        with tf.device("/cpu:0"):
            model = load_model(MODEL_PATH)

        # Load Tokenizer
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)

        # Load Label Encoder
        with open(LABEL_ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)

        # Handle both dict and scikit-learn LabelEncoder
        try:
            index_to_label = {v: k for k, v in label_encoder.items()}
        except AttributeError:
            if hasattr(label_encoder, 'classes_'):
                index_to_label = {i: label for i, label in enumerate(label_encoder.classes_)}
            else:
                raise Exception("Loaded Label Encoder object is in an unsupported format.")
            
        return model, tokenizer, index_to_label, None

    except FileNotFoundError as e:
        return None, None, None, f"File missing: {e.filename}"
    except Exception as e:
        return None, None, None, f"Unexpected error during asset loading: {e}"

model, tokenizer, index_to_label, asset_error = load_assets()

#INITIALIZE GEMINI CLIENT

@st.cache_resource
def initialize_gemini():
    # Use st.secrets or os.getenv, based on your setup. Sticking to os.getenv as per original code.
    api_key = os.getenv("GEMINI_API_KEY") 
    if not api_key:
        return None, " Missing API key. Please set environment variable 'GEMINI_API_KEY'."

    try:
        client = genai.Client(api_key=api_key)
        # Model validation call removed in previous fix
        return client, None
    except APIError as e:
        return None, f"Gemini API error: {e}"
    except Exception as e:
        return None, f"Failed to initialize Gemini client: {e}"

gemini_client, client_error = initialize_gemini()

#CLASSIFICATION FUNCTION

def predict_queue(text, model, tokenizer, max_len):
    """Classify ticket text into a queue label."""
    text = str(text).strip().lower()
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

    probabilities = model.predict(padded, verbose=0)[0]
    pred_index = int(np.argmax(probabilities))
    queue = index_to_label.get(pred_index, "Unknown Queue")
    confidence = float(probabilities[pred_index])
    return queue, confidence

#GEMINI REPLY GENERATION

def generate_reply(original_text, predicted_queue, client):
    """Generate a polite, professional AI reply."""
    system_prompt = (
        "You are an AI assistant for a customer support organization. "
        "Generate a professional, polite, and empathetic initial reply. "
        "Acknowledge the customer's issue and mention that it has been routed "
        f"to the correct department: **{predicted_queue}**. "
        "Keep the reply under 5 sentences, start with 'Dear Customer,' and use Markdown format."
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                {"role": "user", "parts": [{"text": f"Customer Ticket:\n---\n{original_text}"}]},
            ],
            config={"system_instruction": system_prompt, "temperature": 0.3}
        )
        return response.text
    except Exception as e:
        return f" Unable to generate reply: {e}"

#HANDLE CRITICAL ERRORS
if asset_error or client_error:
    st.error(f"**CRITICAL ERROR:** {asset_error or client_error}")
    st.stop()

#MAIN APPLICATION UI
st.subheader(" Enter Customer Ticket")
default_text = (
    "I can't log into my account after the password reset. "
    "I need to access my billing information immediately to pay an overdue invoice."
)
ticket_text = st.text_area("Ticket Body:", height=200, value=default_text)

if st.button(" Classify & Generate Reply", use_container_width=True):
    if not ticket_text.strip():
        st.warning("Please enter some text before proceeding.")
    else:
        with st.spinner(" Analyzing ticket..."):
            queue, confidence = predict_queue(ticket_text, model, tokenizer, MAX_LEN)
            reply = generate_reply(ticket_text, queue, gemini_client)

        st.success("Processing Complete!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(" RNN Classification Result")
            st.info(f"Predicted Queue: **{queue}**")
            st.metric("Model Confidence", f"{confidence*100:.2f}%")

        with col2:
            st.subheader(" Gemini LLM Draft Reply")
            st.markdown("---")
            st.markdown(reply)

        st.markdown("---")
        st.caption("Pipeline: Text ➜ RNN Classification ➜ Queue ➜ Gemini Reply")

#SIDEBAR INFO - FIX APPLIED HERE

with st.sidebar:
    st.markdown("---")
    st.caption(
        " **Project:** Ticket.AI Automation\n"
        " **Built with:** Keras / TensorFlow / Google Gemini\n"
        " **Developer:** Kiran Kumar Bolishetti"
    )