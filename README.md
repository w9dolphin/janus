# Local Voice AI Agent

A real-time voice chat application powered by local AI models. This project allows you to have voice conversations with AI models like Llama running locally on your machine, with support for multiple languages (English, Spanish, Mandarin, Japanese) and RAG (Retrieval-Augmented Generation) capabilities.

## Features

- **Real-time speech-to-text conversion** using OpenAI Whisper
- **Local LLM inference** using Ollama (supports Llama 3.2, Gemma, and other models)
- **Text-to-speech response generation** using Kokoro TTS
- **Multi-language support** (English, Spanish, Mandarin, Japanese)
- **Wake word detection** ("hello", "hola", "你好", "こんにちは")
- **RAG (Retrieval-Augmented Generation)** for context-aware responses using ChromaDB
- **Web interface** built with Gradio for easy interaction
- **Text input option** for typing questions instead of voice
- **WebRTC communication** via FastRTC for low-latency audio streaming

## Project Structure

- `local_voice_chat.py` - Main application file with voice chat interface
- `get_embedding_function.py` - Embedding function for RAG using Ollama embeddings
- `populate_database.py` - Script to populate the RAG database from PDF documents
- `fastrtc/` - Local FastRTC package (WebRTC communication library)

## Prerequisites

- **Python 3.12+**
- **MacOS** (or Linux/Windows with appropriate adjustments)
- [Ollama](https://ollama.ai/) - Run LLMs locally
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver (recommended)
  OR
- **pip** - Standard Python package manager

## Installation

### Option 1: Using uv (Recommended)

#### 1. Install prerequisites with Homebrew

```bash
brew install ollama
brew install uv
```

#### 2. Clone the repository

```bash
git clone https://github.com/jesuscopado/local-voice-ai-agent.git
cd local-voice-ai-agent
```

#### 3. Set up Python environment and install dependencies

```bash
uv venv
source .venv/bin/activate
uv sync
```

### Option 2: Using pip

#### 1. Install prerequisites

```bash
# Install Ollama (macOS)
brew install ollama

# Or download from https://ollama.ai/
```

#### 2. Clone the repository

```bash
git clone https://github.com/jesuscopado/local-voice-ai-agent.git
cd local-voice-ai-agent
```

#### 3. Set up Python environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Download required models in Ollama

```bash
# Download the LLM model (default: llama3.2:3b)
ollama pull llama3.2:3b

# Alternative models you can use:
ollama pull gemma3:1b
ollama pull gemma3:4b

# Download the embedding model for RAG
ollama pull nomic-embed-text
```

### 5. Set up RAG database (Optional)

If you want to use RAG (Retrieval-Augmented Generation) for context-aware responses:

1. Place PDF documents in the `data/` directory
2. Run the database population script:

```bash
python populate_database.py

# To reset the database:
python populate_database.py --reset
```

## Usage

### Basic Voice Chat

```bash
python local_voice_chat.py
```

The application will:
1. Start a Gradio web interface
2. Generate a shareable URL (if using `share=True`)
3. Create a QR code for easy mobile access

### Using the Interface

1. **Voice Input**: Click the microphone button and say the wake word ("hello" for English, "hola" for Spanish, "你好" for Mandarin, "こんにちは" for Japanese)
2. **Text Input**: Type your question in the text box and click Submit
3. **Language Selection**: Use the dropdown to switch between languages
4. The AI will respond with both audio and text output

## How it works

The application uses:
- **FastRTC** for WebRTC communication and audio streaming
- **OpenAI Whisper** for local speech-to-text conversion
- **Kokoro TTS** for text-to-speech synthesis
- **Ollama** for running local LLM inference (default: Llama 3.2 3B)
- **ChromaDB** for RAG (Retrieval-Augmented Generation) with document embeddings
- **LangChain** for document processing and RAG orchestration
- **Gradio** for the web interface

### Workflow

When you speak:
1. Audio is captured via WebRTC
2. Speech is transcribed to text using Whisper
3. Wake word detection filters out non-commands
4. User query is enhanced with relevant context from RAG database (if available)
5. Query is sent to local LLM via Ollama for processing
6. LLM response is truncated based on language-specific limits
7. Response is converted to speech using Kokoro TTS
8. Audio is streamed back via FastRTC

## Configuration

### LLM Model

Edit `local_voice_chat.py` to change the LLM model:

```python
LLM_MODEL = "llama3.2:3b"  # Default
# LLM_MODEL = "gemma3:1b"   # Alternative
```

### Response Limits

- **Word-based languages** (English, Spanish): Maximum 30 words per response
- **Sentence-based languages** (Mandarin, Japanese): Maximum 2-3 sentences per response

### RAG Configuration

- `CHROMA_PATH`: Directory for ChromaDB storage (default: "chroma")
- `DATA_PATH`: Directory containing PDF documents (default: "data")
- `RAG_K`: Number of documents to retrieve (default: 1)
- `MAX_CONTEXT_LENGTH`: Maximum context length in characters (default: 2000)

## Dependencies

See `requirements.txt` or `pyproject.toml` for the complete list of dependencies.

Key dependencies:
- `gradio` - Web interface
- `fastrtc` - WebRTC communication (local package)
- `ollama` - LLM inference
- `openai-whisper` - Speech-to-text
- `kokoro-onnx` - Text-to-speech
- `langchain-chroma` - RAG database
- `langchain-ollama` - Embeddings
- `librosa` - Audio processing
- `numpy` - Numerical operations
- `loguru` - Logging
- `qrcode` - QR code generation
