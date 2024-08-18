import streamlit as st
import requests
from transformers import pipeline

# Hugging Face API URL and headers
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": "Bearer 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"}  # Replace with your actual API key

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def summarize_text(text, min_length, max_length):
    payload = {
        "inputs": text,
        "parameters": {
            "min_length": min_length,
            "max_length": max_length
        }
    }
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

def summarize(text, min_length, max_length):
    if summarizer is not None:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    else:
        return summarize_text(text, min_length, max_length)

# Streamlit app
st.title("MMD Summarizer")

text = st.text_area("Enter text to summarize")

# Input fields for min and max length
min_length = st.number_input("Minimum summary length (words):", min_value=5, max_value=100, value=25)
max_length = st.number_input("Maximum summary length (words):", min_value=5, max_value=200, value=50)

if st.button("Summarize with Streamlit"):
    if text:
        summary = summarize(text, min_length=min_length, max_length=max_length)
        st.write("Summary:")
        st.write(summary)
        
        # Evaluation Metrics
        st.write("Evaluation:")
        st.write(f"Original Length: {len(text.split())} words")
        st.write(f"Summary Length: {len(summary.split())} words")
        st.write(f"Compression Ratio: {len(summary.split()) / len(text.split()):.2f}")
    else:
        st.write("Please enter some text to summarize.")
