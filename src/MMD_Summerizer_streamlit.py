import streamlit as st
import requests
from transformers import pipeline
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bert_score
import nltk
nltk.download('punkt')

# Hugging Face API URL and headers
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": "Bearer YOUR_API_KEY_HERE"}  # Replace with your actual API key

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
    max_input_length = 1024  # Maximum input length for BART model

    # Truncate input text if necessary
    if len(text.split()) > max_input_length:
        text = ' '.join(text.split()[:max_input_length])
        st.warning("Input text was too long and has been truncated.")

    if summarizer is not None:
        try:
            summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            st.error("An error occurred during summarization.")
            return None
    else:
        return summarize_text(text, min_length, max_length)

def evaluate_summary(reference, summary):
    metrics = {}

    # ROUGE scores
    rouge = Rouge()
    rouge_scores = rouge.get_scores(summary, reference, avg=True)
    metrics['ROUGE-1'] = rouge_scores['rouge-1']['f']
    metrics['ROUGE-2'] = rouge_scores['rouge-2']['f']
    metrics['ROUGE-L'] = rouge_scores['rouge-l']['f']

    # BLEU score
    reference_tokens = [nltk.word_tokenize(reference)]
    summary_tokens = nltk.word_tokenize(summary)
    metrics['BLEU'] = sentence_bleu(reference_tokens, summary_tokens)

    # BERTScore
    P, R, F1 = bert_score([summary], [reference], lang="en")
    metrics['BERTScore (F1)'] = F1.mean().item()

    return metrics

# Streamlit app
st.title("MMD Summarizer")

text = st.text_area("Enter text to summarize")

# Input fields for min and max length
min_length = st.number_input("Minimum summary length (words):", min_value=5, max_value=100, value=25)
max_length = st.number_input("Maximum summary length (words):", min_value=5, max_value=200, value=50)

if st.button("Summarize with Streamlit"):
    if text:
        summary = summarize(text, min_length=min_length, max_length=max_length)
        if summary:
            st.write("Summary:")
            st.write(summary)
            
            # Evaluation Metrics
            st.write("Evaluation:")
            st.write(f"Original Length: {len(text.split())} words")
            st.write(f"Summary Length: {len(summary.split())} words")
            st.write(f"Compression Ratio: {len(summary.split()) / len(text.split()):.2f}")
            
            # Calculate and display automatic metrics
            metrics = evaluate_summary(text, summary)
            st.write("ROUGE-1 Score:", metrics['ROUGE-1'])
            st.write("ROUGE-2 Score:", metrics['ROUGE-2'])
            st.write("ROUGE-L Score:", metrics['ROUGE-L'])
            st.write("BLEU Score:", metrics['BLEU'])
            st.write("BERTScore (F1):", metrics['BERTScore (F1)'])
    else:
        st.write("Please enter some text to summarize.")
