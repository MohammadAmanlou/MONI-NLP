# MONI_NLP

## Overview

`MONI_NLP` is a repository that provides tools for text summarization and translation using state-of-the-art natural language processing models. It offers multiple interfaces for summarization, translation, and evaluation, including a unified Streamlit application for flexible workflows.

## Features

- **Text Summarization using Transformers**: Utilizes models like `facebook/bart-large-cnn` from Hugging Face for generating concise and coherent summaries.
- **Translation using MarianMT and other models**: Supports translation between multiple languages using models like `Helsinki-NLP/opus-mt-en-roa`.
- **Gradio Interface**: A simple UI for quick text summarization tasks.
- **Streamlit Interface**: A more detailed interface that allows for customization of tasks like summary length, translation models, and evaluation metrics.
- **Unified Streamlit App**: A combined application for both summarization and translation with customizable workflows.
- **Evaluation Metrics**: Tools to evaluate summaries using metrics like ROUGE, BLEU, and BERTScore.

## File Descriptions

### 1. `app_gradio.py`

Provides a Gradio interface for text summarization using Hugging Face’s summarization pipeline.

### 2. `app_streamlit.py`

Offers an advanced interface with Streamlit for summarization with evaluation options using multiple metrics.

### 3. `Translate.py`

A new Streamlit-based translation interface using the MarianMT model for translating text between various languages.

**Usage:**
```bash
streamlit run app_streamlit.py
```

### 4. `Translation_example_with_MarianMT.ipynb`

Demonstrates the use of MarianMT for translation with examples.

### 5. `Analysis_MarianMT_Translator_model.ipynb`

A notebook that analyzes the MarianMT translation model and its performance.

### 6. `translation_function_for_use_in_big_project.ipynb`

Explains how to integrate the translation function into larger projects.

### 7. `combined_summarize_translate.py`

This is a comprehensive Streamlit application that integrates both summarization and translation capabilities, with options for different models and evaluation metrics.

**Key Features:**
- Choose between summarization, translation, or a combination of both.
- Support for multiple translation models including SeamlessM4T, Opus-MT, and MADLAD400.
- Customizable summarization settings including minimum and maximum lengths.
- Evaluation of generated summaries using ROUGE, BLEU, and BERTScore metrics.

**Usage:**
```bash
streamlit run combined_summarize_translate.py
```

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
4. Run the applications:
    - For Gradio: `python app_gradio.py`
    - For Streamlit Summarizer: `streamlit run app_streamlit.py`
    - For Streamlit Combined App: `streamlit run combined_summarize_translate.py`

## Dependencies

- `gradio`
- `streamlit`
- `transformers`
- `requests`
- `rouge`
- `nltk`
- `bert_score`
- `langid`
- `torch`

Ensure you have these dependencies installed before running the scripts.

## Collaborators

- **Niloofar Hoseinzadeh**: Contributed to the integration of translation and summarization functions.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For any queries or issues, please raise an issue in the repository or contact the maintainer at [MohammadAmanlou2@gmail.com].
