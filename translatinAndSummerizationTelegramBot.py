# -*- coding: utf-8 -*-
import os
import langid
import torch
import nltk
from transformers import (SeamlessM4TModel, AutoProcessor, MarianMTModel, MarianTokenizer, T5ForConditionalGeneration, T5Tokenizer, pipeline)
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bert_score
from telegram import Update, ForceReply
from telegram.ext import Application, CommandHandler, MessageHandler, filters

nltk.download('punkt')

# Hugging Face token for authentication (remove the token for security purposes when sharing code)
HUGGINGFACE_TOKEN = "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"

# Models for summarization
persian_model = "HooshvareLab/bert-fa-base-uncased"
english_model = "facebook/bart-large-cnn"

# Initialize the models globally
summarizers = {
    'fa': pipeline("summarization", model=persian_model, tokenizer=persian_model),
    'en': pipeline("summarization", model=english_model, tokenizer=english_model)
}

# Function to get the appropriate summarizer based on language
def get_summarizer(language):
    if language == 'fa':
        return summarizers['fa']
    else:
        return summarizers['en']

def summarize(text, min_length, max_length):
    max_input_length = 512  # Set to the model's maximum input length
    cleaned_text = text.strip()

    if len(cleaned_text.split()) > max_input_length:
        cleaned_text = ' '.join(cleaned_text.split()[:max_input_length])

    try:
        language, _ = langid.classify(cleaned_text)
        summarizer = get_summarizer(language)

        # Ensure max_length does not exceed the model's output size limit
        if max_length > 150:
            max_length = 150
        
        summary = summarizer(cleaned_text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"An error occurred during summarization: {str(e)}"

def evaluate_summary(reference, summary):
    metrics = {}
    rouge = Rouge()
    rouge_scores = rouge.get_scores(summary, reference, avg=True)
    metrics['ROUGE-1'] = rouge_scores['rouge-1']['f']
    metrics['ROUGE-2'] = rouge_scores['rouge-2']['f']
    metrics['ROUGE-L'] = rouge_scores['rouge-l']['f']

    reference_tokens = [nltk.word_tokenize(reference)]
    summary_tokens = nltk.word_tokenize(summary)
    metrics['BLEU'] = sentence_bleu(reference_tokens, summary_tokens)

    P, R, F1 = bert_score([summary], [reference], lang="en")
    metrics['BERTScore (F1)'] = F1.mean().item()

    return metrics

def translate_text(input_text, output_language="pes", model_index=0):
    try:
        input_language, _ = langid.classify(input_text)

        model_choices = [
            "facebook/hf-seamless-m4t-medium",
            "Helsinki-NLP/opus-mt-en-roa",
            "google/madlad400-10b-mt"
        ]

        model_name = model_choices[model_index]

        if model_index == 0:
            model = SeamlessM4TModel.from_pretrained(model_name)
            processor = AutoProcessor.from_pretrained(model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            inputs = processor(text=input_text, src_lang=input_language, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, tgt_lang=output_language, generate_speech=False)
            translated_text = processor.decode(outputs[0].tolist()[0], skip_special_tokens=True)

        elif model_index == 1:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            input_tokens = tokenizer(input_text, return_tensors="pt", padding=True).to(device)
            translated = model.generate(input_ids=input_tokens["input_ids"], attention_mask=input_tokens["attention_mask"])
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

        elif model_index == 2:
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            text = f"<2{output_language}>{input_text}"
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            outputs = model.generate(input_ids=input_ids)
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return translated_text

    except Exception as e:
        return f"An error occurred during translation: {str(e)}"

# Telegram Bot Handlers
async def start(update: Update, context) -> None:
    user = update.effective_user
    await update.message.reply_html(f"Hi {user.mention_html()}! Welcome to the NLP bot. You can ask for summarization or translation.")

async def summarize_command(update: Update, context) -> None:
    user_input = ' '.join(context.args)
    if not user_input:
        await update.message.reply_text("Please provide some text for summarization.")
        return
    summary = summarize(user_input, min_length=50, max_length=150)
    await update.message.reply_text(f"Summary: {summary}")

async def translate_command(update: Update, context) -> None:
    user_input = ' '.join(context.args)
    if not user_input:
        await update.message.reply_text("Please provide some text for translation.")
        return
    translated_text = translate_text(user_input, output_language="pes", model_index=0)
    await update.message.reply_text(f"Translated Text: {translated_text}")

async def help_command(update: Update, context) -> None:
    await update.message.reply_text("You can use /summarize <text> to summarize and /translate <text> to translate the text.")

# Main function to set up the bot
def main():
    # Add your bot's token here
    TOKEN = "6832279323:AAESNQQOlazT9fUx8JIKlz592RcA17fMcqo"

    # Create the Application and pass it your bot's token
    application = Application.builder().token(TOKEN).build()

    # Register command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("summarize", summarize_command))
    application.add_handler(CommandHandler("translate", translate_command))
    application.add_handler(CommandHandler("help", help_command))

    # Start the bot
    application.run_polling()

if __name__ == "__main__":
    main()
