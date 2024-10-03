# MONI_NLP

**MONI_NLP** is a comprehensive repository designed to simplify and enhance the processes of text summarization and translation using state-of-the-art Natural Language Processing (NLP) models. This project integrates various models and user interfaces, such as Gradio and Streamlit, to provide a flexible, intuitive platform for handling text-related tasks. Additionally, it includes a Telegram bot to facilitate on-the-go summarization and translation.

## Features

### 1. **Text Summarization using Transformers**
- **Model:** The repository uses the `facebook/bart-large-cnn` model from Hugging Face to generate concise, coherent summaries of input text.
- **Interfaces:** 
  - **Gradio Interface**: Provides a straightforward user interface for quick summarization tasks.
  - **Streamlit Interface**: A more detailed application that allows for customization of summaries, including options like summary length, model selection, and evaluation metrics.

### 2. **Translation using MarianMT**
- **Models:** Translation tasks are handled by MarianMT (`Helsinki-NLP/opus-mt-en-roa`), with support for other models.
- **Multilingual Support:** Translate text between multiple languages using models fine-tuned on various language pairs.
- **Interfaces:**
  - **Streamlit Application**: An advanced translation interface with flexible options for task management and language pair selection.

### 3. **Unified Summarization and Translation App**
- A combined Streamlit application integrates both summarization and translation workflows, with full customization options.
- **Support for Models:** Choose between summarization and translation or a combination of both using models like SeamlessM4T, Opus-MT, and MADLAD400.
- **Customizable Settings:** Adjust parameters like minimum/maximum summary length, target language for translation, and more.

### 4. **Evaluation Metrics**
- **Summarization:** Evaluate the quality of generated summaries using well-known metrics such as:
  - **ROUGE**
  - **BLEU**
  - **BERTScore**
  
### 5. **Telegram Bot Integration**
- **Bot Name:** [MONI_NLP_bot](https://t.me/MONI_NLP_bot)
- This Telegram bot provides quick access to the repository’s summarization and translation capabilities, making it easy to handle text tasks on mobile devices.

## File Descriptions

1. **`MMD_Summarizer_Gradio.py`**  
   - Gradio interface for text summarization using the Hugging Face model.

2. **`MMD_Summarizer_Streamlit.py`**  
   - Streamlit interface for advanced text summarization with evaluation options.

3. **`translate_and_summarize_Streamlit.py`**  
   - A combined Streamlit application that integrates summarization and translation workflows.

4. **`Analysis_MarianMT_Translator_model.ipynb`**  
   - A Jupyter notebook analyzing the performance of MarianMT for multilingual translations.

5. **`Summarizing_Metrics.ipynb`**  
   - A notebook to evaluate summarization results using different metrics like ROUGE, BLEU, and BERTScore.

6. **`Translation_example_with_MarianMT.ipynb`**  
   - Example notebook demonstrating the use of MarianMT for translation with test cases.

7. **`translation_function_for_use_in_big_project.ipynb`**  
   - Explains how to integrate the translation function into larger projects, offering practical usage guidance.

8. **`translate_and_summarize_TelegramBot.py`**  
   - Code for the Telegram bot which performs both summarization and translation tasks via the bot interface.

9. **`README.md`**  
   - This file contains a detailed guide for repository usage.

## Usage

### **Summarization:**
1. Gradio:  
   ```bash
   python MMD_Summarizer_Gradio.py
   ```
2. Streamlit:  
   ```bash
   streamlit run MMD_Summarizer_Streamlit.py
   ```

### **Translation:**
1. Streamlit:  
   ```bash
   streamlit run translate_and_summarize_Streamlit.py
   ```

### **Telegram Bot:**
To use the Telegram bot for summarization or translation, search for **[MONI_NLP_bot](https://t.me/MONI_NLP_bot)** on Telegram and follow the instructions.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MONI_NLP.git
   ```
2. Navigate to the repository:
   ```bash
   cd MONI_NLP
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies
- `gradio`
- `streamlit`
- `transformers`
- `torch`
- `requests`
- `rouge`
- `nltk`
- `bert_score`
- `langid`

Ensure these dependencies are installed before running the scripts.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributors
- **Niloofar Hoseinzadeh:** Contributed to the integration of translation and summarization functions.
- **Mohammad Amanlou:** Maintainer and lead developer.

## Contact
For questions or issues, please raise an issue in this repository or reach out at **[MohammadAmanlou2@gmail.com](mailto:MohammadAmanlou2@gmail.com)**.
