# -*- coding: utf-8 -*-
import langid
import torch
import nltk
from transformers import (SeamlessM4TModel, AutoProcessor, MarianMTModel, MarianTokenizer, T5ForConditionalGeneration, T5Tokenizer, pipeline)
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, ConversationHandler, filters
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()  
API_KEY = os.getenv('API_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

# Log in to Hugging Face with your token
login(token=API_KEY, add_to_git_credential=True)

nltk.download('punkt')

# Model settings
persian_model = "HooshvareLab/bert-fa-base-uncased"
english_model = "facebook/bart-large-cnn"
spanish_model = "facebook/bart-large-cnn"
french_model = "facebook/bart-large-cnn"

summarizers = {
    'pes': pipeline("summarization", model=persian_model, tokenizer=persian_model),
    'eng': pipeline("summarization", model=english_model, tokenizer=english_model),
    'spa': pipeline("summarization", model=spanish_model, tokenizer=spanish_model),
    'fra': pipeline("summarization", model=french_model, tokenizer=french_model)
}

# Conversation states
CHOOSE_OPERATION, GET_TEXT, GET_MIN_LENGTH, GET_MAX_LENGTH, GET_TARGET_LANGUAGE, GET_SUMMARIZED_TEXT_LANGUAGE, FINAL_MESSAGE = range(7)

# Emojis
SUMMARIZE_EMOJI = "📝"
TRANSLATE_EMOJI = "🌍"
WELCOME_EMOJI = "👋"
CONFIRM_EMOJI = "✅"
ERROR_EMOJI = "❗"

# Language options
LANGUAGE_OPTIONS = {
    "Persian": "pes",
    "Spanish": "spa",
    "French": "fra",
    "English": "eng"
}

# Helper functions
def get_summarizer(language):
    if language in summarizers:
        return summarizers[language]
    else:
        return summarizers['eng']  # Default to English if language not supported

def summarize(text, min_length, max_length):
    max_input_length = 512
    cleaned_text = text.strip()

    if len(cleaned_text.split()) > max_input_length:
        cleaned_text = ' '.join(cleaned_text.split()[:max_input_length])

    try:
        language, _ = langid.classify(cleaned_text)
        summarizer = get_summarizer(language)

        summary = summarizer(cleaned_text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"{ERROR_EMOJI} An error occurred during summarization: {str(e)}"

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
        return f"{ERROR_EMOJI} An error occurred during translation: {str(e)}"

# Telegram bot handlers
async def start(update: Update, context) -> None:
    user = update.effective_user
    welcome_message = (
        f"{WELCOME_EMOJI} Hi {user.first_name}! Welcome to the NLP bot.\n\n"
        "Please choose an operation to get started:\n"
        f"{SUMMARIZE_EMOJI} Summarization only\n"
        f"{TRANSLATE_EMOJI} Translation only\n"
        f"{SUMMARIZE_EMOJI}{TRANSLATE_EMOJI} Summarization before Translation\n"
        f"{TRANSLATE_EMOJI}{SUMMARIZE_EMOJI} Translation before Summarization\n"
    )

    # Show options for the user to choose from
    options = [["Summarization only", "Translation only"], ["Summarization before Translation", "Translation before Summarization"]]
    reply_markup = ReplyKeyboardMarkup(options, one_time_keyboard=True)

    await update.message.reply_text(welcome_message, reply_markup=reply_markup)
    return CHOOSE_OPERATION

async def choose_operation(update: Update, context) -> int:
    user_choice = update.message.text
    context.user_data['operation'] = user_choice

    if user_choice == "Summarization only":
        await update.message.reply_text(f"{SUMMARIZE_EMOJI} Please enter the text you want to summarize:")
        return GET_TEXT
    elif user_choice == "Translation only":
        await update.message.reply_text(f"{TRANSLATE_EMOJI} Please enter the text you want to translate:")
        return GET_TEXT 
    elif user_choice == "Summarization before Translation":
        await update.message.reply_text(f"{SUMMARIZE_EMOJI} Please enter the text you want to summarize:")
        return GET_TEXT
    elif user_choice == "Translation before Summarization":
        await update.message.reply_text(f"{TRANSLATE_EMOJI} Please enter the text you want to translate before summarizing:")
        return GET_TEXT 
    else:
        await update.message.reply_text(f"{ERROR_EMOJI} Invalid choice, please try again.")
        return CHOOSE_OPERATION


async def get_text(update: Update, context) -> int:
    user_text = update.message.text
    context.user_data['text'] = user_text  # ذخیره متن دریافت شده

    if context.user_data['operation'] == "Translation only":
        await update.message.reply_text(f"{CONFIRM_EMOJI} Text received! Please choose the target language for translation:")
        
        # ایجاد کیبورد از زبان‌های موجود در LANGUAGE_OPTIONS
        language_options = [[language] for language in LANGUAGE_OPTIONS.keys()]
        reply_markup = ReplyKeyboardMarkup(language_options, one_time_keyboard=True)
        await update.message.reply_text("Select a language:", reply_markup=reply_markup)
        return GET_TARGET_LANGUAGE
    
    elif context.user_data['operation'] == "Translation before Summarization":
        await update.message.reply_text(f"{CONFIRM_EMOJI} Text received! Please choose the target language for translation:")
        
        # نمایش گزینه‌های زبان مقصد
        language_options = [[language] for language in LANGUAGE_OPTIONS.keys()]
        reply_markup = ReplyKeyboardMarkup(language_options, one_time_keyboard=True)
        await update.message.reply_text("Select a language:", reply_markup=reply_markup)
        return GET_TARGET_LANGUAGE



async def get_target_language(update: Update, context) -> int:
    user_language_choice = update.message.text

    # بررسی اینکه آیا زبان انتخاب شده معتبر است یا خیر
    if user_language_choice in LANGUAGE_OPTIONS:
        # ذخیره زبان مقصد در context
        context.user_data['target_language'] = LANGUAGE_OPTIONS[user_language_choice]
        await update.message.reply_text(f"{CONFIRM_EMOJI} Target language selected: {user_language_choice}.\nNow please enter the minimum length for the summary:")
        return GET_MIN_LENGTH  # رفتن به مرحله دریافت حداقل طول
    else:
        await update.message.reply_text(f"{ERROR_EMOJI} Invalid language choice, please try again.")
        return GET_TARGET_LANGUAGE



async def get_min_length(update: Update, context) -> int:
    try:
        min_length = int(update.message.text)
        context.user_data['min_length'] = min_length
        await update.message.reply_text("Please enter the maximum length for the summary:")
        return GET_MAX_LENGTH
    except ValueError:
        await update.message.reply_text("Please enter a valid integer.")
        return GET_MIN_LENGTH

async def get_max_length(update: Update, context) -> int:
    try:
        max_length = int(update.message.text)
        context.user_data['max_length'] = max_length

        # اجرای ترجمه بر روی متن اصلی
        user_text = context.user_data['text']
        target_language = context.user_data['target_language']

        await update.message.reply_text(f"{CONFIRM_EMOJI} Processing translation...")
        translated_text = translate_text(user_text, output_language=target_language, model_index=0)
        context.user_data['translated_text'] = translated_text  # ذخیره متن ترجمه شده

        # خلاصه‌سازی متن ترجمه‌شده
        await update.message.reply_text(f"{CONFIRM_EMOJI} Processing summarization...")
        summary = summarize(translated_text, min_length=context.user_data['min_length'], max_length=max_length)
        context.user_data['summarized_text'] = summary  # ذخیره خلاصه نهایی

        await update.message.reply_text(f"Translated and Summarized Text: {summary}")
        return ConversationHandler.END

    except ValueError:
        await update.message.reply_text("Please enter a valid integer.")
        return GET_MAX_LENGTH


# Main function to run the bot
def main():
    # Add your bot token here
    TOKEN = TELEGRAM_TOKEN

    # Create the Application and pass it your bot's token
    application = Application.builder().token(TOKEN).build()

    # Conversation handler to manage the flow
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            CHOOSE_OPERATION: [MessageHandler(filters.TEXT & ~filters.COMMAND, choose_operation)],
            GET_TEXT: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_text)],
            GET_MIN_LENGTH: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_min_length)],
            GET_MAX_LENGTH: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_max_length)],
            GET_TARGET_LANGUAGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_target_language)],
        },
        fallbacks=[]
    )

    application.add_handler(conv_handler)

    # Start the bot
    application.run_polling()

if __name__ == "__main__":
    main()
