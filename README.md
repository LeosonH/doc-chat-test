# docChat

docChat is a Streamlit-based application that allows you to upload PDF documents and interactively ask questions about their content using OpenAI's language models. It leverages the [embedchain](https://github.com/embedchain/embedchain) framework for document ingestion and retrieval.

## Features

- Upload one or more PDF files.
- Ask questions about the uploaded documents.
- Uses OpenAI's GPT models for answering queries.
- Secure: Your OpenAI API key is not stored.

## Getting Started

### Prerequisites

- Python 3.8+
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Installation

1. Clone this repository:

    ```sh
    git clone <your-repo-url>
    cd docGPT
    ```

2. Install dependencies:

    ```sh
    pip install -r requirements.txt
    ```

### Running the App

Start the Streamlit app:

```sh
streamlit run app.py
```

Open your browser and go to [http://localhost:8501](http://localhost:8501).

### Usage

1. Enter your OpenAI API key in the sidebar.
2. Upload your PDF files.
3. Ask questions in the chat interface.

## File Structure

- `app.py`: Main Streamlit application.
- `kwab.py`: Chainlit-based chatbot (alternative interface).
- `requirements.txt`: Python dependencies.
- `embedchain.json`: Embedchain configuration.
- `.streamlit/secrets.toml`: (Optional) Store your OpenAI API key for local development.

## Notes

- Your OpenAI API key is required for the app to function.
- Uploaded files are processed temporarily and not stored permanently.

## License

MIT License

---

*This project uses [embedchain](https://github.com/embedchain/embedchain) and [Streamlit](https://streamlit.io/).*
