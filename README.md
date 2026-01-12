# 🎙️ HeadOn GPT

A local RAG-powered chatbot that lets you analyze conversations from YouTube videos. Upload any YouTube debate, interview, or discussion and ask questions about what the speakers said.

![Python](https://img.shields.io/badge/python-3.12-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.32+-red)
![License](https://img.shields.io/badge/license-Apache%202.0-green)

## ✨ Features

- **YouTube Integration** — Paste any YouTube URL to automatically download and transcribe with speaker diarization
- **Speaker Analysis** — See stats for each speaker (talk time, word count, WPM) and AI-generated stance analysis
- **Local RAG** — All processing happens locally using ChromaDB for retrieval and Ollama for generation
- **Chat Interface** — Ask natural language questions about the conversation
- **Evidence Citations** — Responses include relevant transcript snippets with timestamps

## 🖼️ Preview

```
┌─────────────────────────────────────────────────────┐
│  🎙️ HeadOn GPT                                      │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ## The Conversation                                │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐                │
│  │ 🩷 Adam      │  │ 💙 Tal       │                │
│  │ PARTICIPANT  │  │ PARTICIPANT  │                │
│  │              │  │              │                │
│  │ 22.9 min     │  │ 18.3 min     │                │
│  │ 4,409 words  │  │ 3,201 words  │                │
│  └──────────────┘  └──────────────┘                │
│                                                     │
│  💬 Ask about the conversation...                  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.12** (recommended)
- **Ollama** with a model installed (e.g., `llama3`)
- **AssemblyAI API Key** (for transcription) — [Get one free](https://www.assemblyai.com/)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/PhilbertChristian/sentiment_evaluator_bot.git
   cd sentiment_evaluator_bot
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama and a model**
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull llama3
   ```

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

6. **Open in browser**
   
   Navigate to `http://localhost:8501`

## 📖 Usage

### Loading a YouTube Video

1. Expand the **"Load a different YouTube video"** section at the bottom
2. Paste the YouTube URL
3. Enter the number of speakers (e.g., 2 for a debate)
4. Enter your AssemblyAI API key
5. Click **"Create New Chatbot"**

The app will:
- Download the audio from YouTube
- Transcribe it with speaker diarization via AssemblyAI
- Build a local vector index with ChromaDB
- Display speaker profiles

### Asking Questions

Simply type your question in the chat input at the bottom:

- *"What did they disagree about?"*
- *"What was Adam's main argument?"*
- *"Summarize Tal's position on X"*
- *"Did they find any common ground?"*

The AI will retrieve relevant transcript snippets and generate an answer with citations.

### Settings

Click **⚙️ Settings** in the top-left to:
- Adjust **Context depth (k)** — number of snippets to retrieve (3-10)
- View transcript info
- Clear chat history
- Export conversation log

## 🗂️ Project Structure

```
headon_gpt/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/
│   ├── video_transcripts.jsonl   # Default transcript file
│   ├── session.jsonl             # Current session transcript
│   ├── chat_log.jsonl            # Conversation history
│   └── chroma/                   # Vector database
└── scripts/
    ├── local_rag.py              # RAG index building & retrieval
    ├── youtube_to_jsonl.py       # YouTube download & transcription
    └── analyze_transcript.py     # Speaker analysis utilities
```

## ⚙️ Configuration

### Environment Variables (optional)

```bash
export ASSEMBLYAI_API_KEY="your_key_here"  # Skip entering in UI
```

### Customizing Speaker Names

Edit the `SPEAKER_NAMES` dictionary in `app.py`:

```python
SPEAKER_NAMES = {
    "A": "Adam",
    "B": "Tal",
    "speaker_1": "Adam",
    "speaker_2": "Tal",
}
```

### Changing the LLM Model

Edit `OLLAMA_MODEL` in `app.py`:

```python
OLLAMA_MODEL = "llama3"  # or "mistral", "phi3", etc.
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | [Streamlit](https://streamlit.io/) |
| Vector DB | [ChromaDB](https://www.trychroma.com/) |
| Embeddings | [Sentence Transformers](https://www.sbert.net/) (all-MiniLM-L6-v2) |
| LLM | [Ollama](https://ollama.ai/) (local) |
| Transcription | [AssemblyAI](https://www.assemblyai.com/) |
| YouTube Download | [yt-dlp](https://github.com/yt-dlp/yt-dlp) |

## 🐛 Troubleshooting

### "Ollama not found"
Make sure Ollama is installed and running:
```bash
ollama serve  # Start the Ollama server
ollama list   # Check installed models
```

### "ChromaDB inconsistent state"
Delete the chroma directory and restart:
```bash
rm -rf data/chroma
streamlit run app.py
```

### Transcription taking too long
AssemblyAI transcription typically takes 20-40% of the video length. A 30-minute video takes ~10 minutes to process.

### pip install hanging
Try installing torch separately first:
```bash
pip install torch==2.3.1
pip install -r requirements.txt
```

## 📜 License

This project is licensed under the Apache 2.0 License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing UI framework
- [AssemblyAI](https://www.assemblyai.com/) for accurate transcription with diarization
- [Ollama](https://ollama.ai/) for making local LLMs accessible

---

**Made with ❤️ by [Philbert Christian](https://github.com/PhilbertChristian)**
