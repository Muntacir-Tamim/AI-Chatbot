# 📚 PDF AI Chatbot

> Upload your PDFs and chat with them. Get accurate answers with page references, in English or Bangla.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)
![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-8b5cf6)
![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-green)

## Overview

PDF AI Chatbot is a Retrieval-Augmented Generation (RAG) application. It reads one or more PDF files, indexes their content, and answers your questions using **only the information found in those documents**. Every answer shows which file and page it came from.

## Features

- **Multi-PDF chat:** upload and query several PDFs together.
- **Scanned PDF support:** pages without selectable text are read with Gemini OCR (Bangla supported).
- **Grounded answers:** the model answers only from the document and says so when the answer is missing.
- **Source references:** file name and page number are shown for each answer.
- **Smart retrieval:** MMR search gives relevant and diverse chunks; summaries use chunks from the whole document.
- **Quick actions:** one-click Summary, Key Points, Important Numbers and Dates, and Explain Simply.
- **Answer styles:** choose Concise or Detailed.
- **Language aware:** replies in the language you ask in.
- **Reliable:** automatic fallback to another Gemini model if one is busy (503).
- **Modern UI:** clean chat interface, document stats, and chat download.

## How it works

```
PDF ──► Extract text (PyMuPDF / Gemini OCR)
    ──► Split into chunks (with page numbers)
    ──► Embeddings (gemini-embedding-001)
    ──► FAISS vector index
Question ──► Embed ──► MMR search ──► Top chunks ──► Gemini ──► Streamed answer + sources
```

## Tech Stack

| Layer | Technology |
|---|---|
| Interface | Streamlit |
| PDF reading | PyMuPDF, Gemini OCR |
| Embeddings | `gemini-embedding-001` |
| Vector search | FAISS |
| Answer model | Gemini 3.x Flash (with fallback list) |
| Language | Python |

## Project Structure

```
AI-Chatbot-main/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── .env                # Your API key (not committed)
└── README.md
```

## Getting Started

### 1. Clone the project

```bash
git clone <your-repo-url>
cd AI-Chatbot-main
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
streamlit>=1.40
google-genai>=1.0
faiss-cpu
numpy
pymupdf
python-dotenv
```

### 4. Add your API key

Get a free key from [Google AI Studio](https://aistudio.google.com/apikey) and create a `.env` file:

```
GOOGLE_API_KEY=your_api_key_here
```

You can also paste the key in the sidebar when the app starts.

### 5. Run the app

```bash
streamlit run app.py
```

## Usage

1. Upload one or more PDFs from the sidebar.
2. Click **Process PDFs** and wait until the stats appear.
3. Ask a question, or use a quick action.
4. Open **Sources** under an answer to see the pages used.

## Configuration

Edit these values at the top of `app.py`:

| Setting | Default | Meaning |
|---|---|---|
| `CHAT_MODELS` | Gemini 3.x list | Models tried in order |
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `TOP_K` | 8 | Chunks used per question |
| `MAX_OCR_PAGES` | 40 | Max scanned pages to OCR |

## Troubleshooting

| Problem | Fix |
|---|---|
| `404 NOT_FOUND` model error | Update model names in `CHAT_MODELS` |
| `503 UNAVAILABLE` | Model is busy; the app retries other models automatically |
| No answer appears | Wait for the timeout (60 s) and ask again; check the error shown |
| "No readable text found" | The PDF may be empty or corrupted |
| Rate limit errors | Free API keys have limits; wait a minute and retry |

## Limitations

- Index is kept in memory, so it resets when the page is refreshed.
- OCR is limited to the first 40 scanned pages per file.
- Answer quality depends on the PDF text quality.

## Future Improvements

- Save chat history and indexes per user
- Voice input in Bangla and English
- Support for Word and text files
- Online deployment

## License

This project is for educational use. Add a license of your choice (for example MIT) before publishing.