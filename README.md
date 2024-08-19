# MMD_Summarizer

## Overview

`MMD_Summarizer` is a repository that provides tools for text summarization using state-of-the-art natural language processing models. It offers two main interfaces: one built with Gradio for quick summarization tasks, and another built with Streamlit for more detailed summaries and evaluation using multiple metrics.

## Features

- **Text Summarization using Transformers**: Utilizes the `facebook/bart-large-cnn` model from Hugging Face for generating concise and coherent summaries.
- **Gradio Interface**: A simple UI for quick text summarization tasks.
- **Streamlit Interface**: A more detailed interface that allows for custom summary lengths and evaluates the quality of the summaries using various metrics.
- **Evaluation Metrics**: The repository includes a detailed notebook that explains and implements various automatic and content-based evaluation metrics like ROUGE, BLEU, and BERTScore.

## File Descriptions

### 1. `app_gradio.py`

This script provides a Gradio interface for text summarization. It utilizes the Hugging Face `pipeline` for summarization and falls back to the `facebook/bart-large-cnn` model if needed.

**Key Features:**
- Simple and user-friendly text box for input.
- Outputs a summarized version of the input text.
- Option to launch the interface with a shareable link.

**Usage:**
```bash
python app_gradio.py
```

### 2. `app_streamlit.py`

This script offers a more advanced interface using Streamlit. It allows users to specify the desired length of the summary and evaluate the summary using various metrics.

**Key Features:**
- Allows customization of minimum and maximum summary length.
- Provides a detailed evaluation of the summary including metrics like ROUGE, BLEU, and BERTScore.
- Displays warnings if the input text exceeds the model's input length limits.

**Usage:**
```bash
streamlit run app_streamlit.py
```

### 3. `evaluation_metrics_notebook.ipynb`

This Jupyter notebook provides a comprehensive overview of text summarization evaluation metrics. It covers both automatic metrics (like ROUGE, BLEU, BERTScore) and content-based metrics, as well as readability metrics and human evaluation criteria.

**Key Sections:**
- **Introduction to Evaluation Metrics**
- **Automatic Metrics**: ROUGE, BLEU, METEOR, etc.
- **Content-Based Metrics**: Information Content, Coverage, Redundancy.
- **Readability Metrics**: Flesch-Kincaid, Gunning Fog Index.
- **Human Evaluation**: Fluency, Coherence, Relevance, Conciseness.

**Usage:**
Open the notebook in Jupyter and execute the cells to learn about and implement these metrics.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/MMD_Summarizer.git
    ```
2. Navigate to the repository:
    ```bash
    cd MMD_Summarizer
    ```
3. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the applications:
    - For Gradio: `python app_gradio.py`
    - For Streamlit: `streamlit run app_streamlit.py`

## Dependencies

- `gradio`
- `streamlit`
- `requests`
- `transformers`
- `rouge`
- `nltk`
- `bert_score`

Ensure you have these dependencies installed before running the scripts.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For any queries or issues, please raise an issue in the repository or contact the maintainer at [MohammadAmanlou2@gmail.com].

