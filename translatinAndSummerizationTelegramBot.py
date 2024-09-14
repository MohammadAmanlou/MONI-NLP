# -*- coding: utf-8 -*-
import langid
import torch
import nltk
from transformers import (SeamlessM4TModel, AutoProcessor, MarianMTModel, MarianTokenizer, T5ForConditionalGeneration, T5Tokenizer, pipeline)
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bert_score
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, ConversationHandler, filters

nltk.download('punkt')

# Model settings
persian_model = "HooshvareLab/bert-fa-base-uncased"
english_model = "facebook/bart-large-cnn"

summarizers = {
    'fa': pipeline("summarization", model=persian_model, tokenizer=persian_model),
    'en': pipeline("summarization", model=english_model, tokenizer=english_model)
}

# Conversation states
CHOOSE_OPERATION, GET_TEXT, SUMMARIZE_ONLY, TRANSLATE_ONLY, TRANSLATION_AFTER_SUMMARY, SUMMARIZATION_AFTER_TRANSLATION = range(6)

# Emojis
SUMMARIZE_EMOJI = "📝"
TRANSLATE_EMOJI = "🌍"
WELCOME_EMOJI = "👋"
CONFIRM_EMOJI = "✅"
ERROR_EMOJI = "❗"

# Helper functions
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

        if max_length > 150:
            max_length = 150

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

    if user_choice in ["Summarization only", "Summarization before Translation", "Translation before Summarization"]:
        await update.message.reply_text(f"{SUMMARIZE_EMOJI} Please enter the text for summarization:")
        return GET_TEXT
    elif user_choice == "Translation only":
        await update.message.reply_text(f"{TRANSLATE_EMOJI} Please enter the text for translation:")
        return GET_TEXT
    else:
        await update.message.reply_text(f"{ERROR_EMOJI} Invalid choice, please try again.")
        return CHOOSE_OPERATION

async def get_text(update: Update, context) -> int:
    user_text = update.message.text
    context.user_data['text'] = user_text

    if context.user_data['operation'] == "Summarization only":
        await update.message.reply_text(f"{CONFIRM_EMOJI} Processing summarization...")
        summary = summarize(user_text, min_length=50, max_length=150)
        await update.message.reply_text(f"Summary: {summary}")
        return ConversationHandler.END

    elif context.user_data['operation'] == "Translation only":
        await update.message.reply_text(f"{CONFIRM_EMOJI} Processing translation...")
        translation = translate_text(user_text, output_language="pes", model_index=0)
        await update.message.reply_text(f"Translated Text: {translation}")
        return ConversationHandler.END

    elif context.user_data['operation'] == "Summarization before Translation":
        await update.message.reply_text(f"{CONFIRM_EMOJI} Processing summarization first...")
        summary = summarize(user_text, min_length=50, max_length=150)
        context.user_data['summary'] = summary
        await update.message.reply_text(f"Summary done! Now translating...\nSummary: {summary}")
        translation = translate_text(summary, output_language="pes", model_index=0)
        await update.message.reply_text(f"Translated Summary: {translation}")
        return ConversationHandler.END

    elif context.user_data['operation'] == "Translation before Summarization":
        await update.message.reply_text(f"{CONFIRM_EMOJI} Processing translation first...")
        translation = translate_text(user_text, output_language="pes", model_index=0)
        context.user_data['translation'] = translation
        await update.message.reply_text(f"Translation done! Now summarizing...\nTranslated Text: {translation}")
        summary = summarize(translation, min_length=50, max_length=150)
        await update.message.reply_text(f"Summarized Translation: {summary}")
        return ConversationHandler.END

# Main function to run the bot
def main():
    # Add your bot token here
    TOKEN = "6832279323:AAESNQQOlazT9fUx8JIKlz592RcA17fMcqo"

    # Create the Application and pass it your bot's token
    application = Application.builder().token(TOKEN).build()

    # Conversation handler to manage the flow
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            CHOOSE_OPERATION: [MessageHandler(filters.TEXT & ~filters.COMMAND, choose_operation)],
            GET_TEXT: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_text)]
        },
        fallbacks=[]
    )

    application.add_handler(conv_handler)

    # Start the bot
    application.run_polling()

if __name__ == "__main__":
    main()
