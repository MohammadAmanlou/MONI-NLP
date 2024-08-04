import gradio as gr
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

# Gradio interface
iface = gr.Interface(fn=summarize, inputs="textbox", outputs="textbox", title="Text Summarizer - Gradio")
iface.launch(share=True)
