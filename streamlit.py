import streamlit as st
import requests
from transformers import pipeline

# Hugging Face API URL and headers
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": "Bearer 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"}  # Replace with your actual API key

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def summarize_text(text):
    payload = {"inputs": text}
    output = query(payload)

    # Check if the API response is valid
    if isinstance(output, list) and len(output) > 0:
        return output[0]['summary_text']
    else:
        return "Error: Unable to get a summary."

# Specify the model explicitly
model_name = "sshleifer/distilbart-cnn-12-6"  
try:
    summarizer = pipeline("summarization", model=model_name)
except Exception as e:
    summarizer = None

def summarize(text):
    if summarizer is not None:
        summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
        return summary[0]['summary_text']
    else:
        return summarize_text(text)

# Streamlit app
st.title("Text Summarizer with Hugging Face API")

text = st.text_area("Enter text to summarize")

if st.button("Summarize with Streamlit"):
    if text:
        summary = summarize(text)
        st.write("Summary:")
        st.write(summary)
        
        # Evaluation Metrics
        st.write("Evaluation:")
        st.write(f"Original Length: {len(text.split())} words")
        st.write(f"Summary Length: {len(summary.split())} words")
        st.write(f"Compression Ratio: {len(summary.split()) / len(text.split()):.2f}")
    else:
        st.write("Please enter some text to summarize.")
