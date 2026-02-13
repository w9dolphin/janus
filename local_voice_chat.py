import sys
import re
import os
import qrcode
import threading
import time
import numpy as np
import tempfile
import wave
import webbrowser
import http.server
import socketserver
import socket
import gradio as gr
from concurrent.futures import ThreadPoolExecutor
from fastrtc import ReplyOnPause, Stream, get_tts_model, AdditionalOutputs
from fastrtc.reply_on_pause import AlgoOptions
from fastrtc.text_to_speech.tts import KokoroTTSOptions
from loguru import logger
from ollama import chat
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
import whisper
import librosa

# Initialize Whisper model for STT
try:
    whisper_model = whisper.load_model("base")
    logger.info("✅ Whisper model loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load Whisper model: {e}")
    raise

tts_model = get_tts_model()  # kokoro

# Configure logging to see what's happening
try:
    logger.remove(0)  # Remove default handler if it exists
except ValueError:
    pass  # Handler doesn't exist, which is fine
logger.add(sys.stderr, level="DEBUG")

# Suppress uvicorn access logs (the "content_type application/json" messages)
import logging
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Wake word state - tracks whether wake word has been detected
wake_word_detected = False

# Language state - tracks current language preference (English, Spanish, Mandarin, or Japanese)
current_language = "English"

# Language configuration dictionary
LANGUAGE_CONFIG = {
    "English": {
        "wake_word": "hello",
        "wake_word_variations": ["hello", "ello", "hello ", " hello", "hello,", "hello.", "hello!"],
        "system_prompt_template": "You are a helpful AI assistant. Your responses will be converted to audio, so do not include emojis, special characters, or markdown formatting in your answers. Speak naturally and clearly. Make your answers {limit} words or less.",
        "tts_voice": "af_heart",
        "tts_lang": "en-us",
        "whisper_lang": "en",
        "context_instruction": "Use the following context to answer the question. If the context doesn't contain relevant information, answer based on your general knowledge:\n\nContext:\n",
        "word_limit_reminder_template": "\n\nRemember: Make your answer {limit} words or less.",
        "echo_message_template": "You said: {message}. Let me find the answer for you.",
        "error_message": "Sorry, I encountered an error processing your request.",
        "greeting": "Hello! How can I help you?",
        "empty_response_message": "I'm sorry, I couldn't generate a proper response.",
        "uses_word_limit": True,
        "max_sentences": None
    },
    "Spanish": {
        "wake_word": "hola",
        "wake_word_variations": ["hola", "ola", "holá", "óla", "ola ", " hola", "hola,", "hola.", "hola!"],
        "system_prompt_template": "Eres un asistente de IA útil. Tus respuestas se convertirán en audio, así que no incluyas emojis, caracteres especiales o formato markdown en tus respuestas. Habla de forma natural y clara. Haz que tus respuestas tengan {limit} palabras o menos.",
        "tts_voice": "em_alex",
        "tts_lang": "es",
        "whisper_lang": "es",
        "context_instruction": "Usa el siguiente contexto (que puede estar en inglés) para responder la pregunta en español. Si el contexto no contiene información relevante, responde basándote en tu conocimiento general:\n\nContexto:\n",
        "word_limit_reminder_template": "\n\nRecuerda: Haz que tu respuesta tenga {limit} palabras o menos.",
        "echo_message_template": "Dijiste: {message}. Dejame buscar la respuesta para ti.",
        "error_message": "Lo siento, encontré un error al procesar tu solicitud.",
        "greeting": "¡Hola! ¿Cómo puedo ayudarte?",
        "empty_response_message": "Lo siento, no pude generar una respuesta adecuada.",
        "uses_word_limit": True,
        "max_sentences": None
    },
    "Mandarin": {
        "wake_word": "你好",
        "wake_word_variations": ["你好", "你好 ", " 你好", "你好，", "你好。", "你好！", "你好?", "你好？"],
        "system_prompt_template": "你是一个有用的AI助手。你的回答将被转换为音频，所以不要在回答中包含表情符号、特殊字符或markdown格式。自然清晰地说话。让你的回答最多2-3句话，对于更复杂的问题也是如此。",
        "tts_voice": "zm_yunxi",  # May need adjustment based on available voices
        "tts_lang": "cmn",  # Mandarin Chinese (using 'cmn' for espeak compatibility instead of 'z')
        "whisper_lang": "zh",
        "context_instruction": "使用以下上下文（可能为英文）来用中文回答问题。如果上下文不包含相关信息，请根据你的常识回答：\n\n上下文：\n",
        "word_limit_reminder_template": "\n\n记住：让你的回答最多2-3句话。",
        "echo_message_template": "你说：{message}。让我为你找到答案。",
        "error_message": "抱歉，我在处理你的请求时遇到了错误。",
        "greeting": "你好！我能为你做什么？",
        "empty_response_message": "抱歉，我无法生成合适的回答。",
        "uses_word_limit": False,
        "max_sentences": 3
    },
    "Japanese": {
        "wake_word": "こんにちは",
        "wake_word_variations": ["こんにちは", "こんにちは ", " こんにちは", "こんにちは、", "こんにちは。", "こんにちは！", "こんにちは?", "こんにちは？"],
        "system_prompt_template": "あなたは役立つAIアシスタントです。あなたの回答は音声に変換されるため、絵文字、特殊文字、またはmarkdown形式を含めないでください。自然で明確に話してください。あなたの回答は最大2-3文にしてください。より複雑な質問でも同様です。",
        "tts_voice": "jf_alpha",  # May need adjustment based on available voices
        "tts_lang": "ja",  # Japanese (using 'ja' for espeak compatibility instead of 'j')
        "whisper_lang": "ja",
        "context_instruction": "次のコンテキスト（英語の可能性があります）を使用して、日本語で質問に答えてください。コンテキストに関連情報が含まれていない場合は、一般的な知識に基づいて答えてください：\n\nコンテキスト：\n",
        "word_limit_reminder_template": "\n\n覚えておいてください：あなたの回答は最大2-3文にしてください。",
        "echo_message_template": "あなたは言いました：{message}。答えを見つけてあげます。",
        "error_message": "申し訳ございませんが、リクエストの処理中にエラーが発生しました。",
        "greeting": "こんにちは！どのようにお手伝いできますか？",
        "empty_response_message": "申し訳ございませんが、適切な回答を生成できませんでした。",
        "uses_word_limit": False,
        "max_sentences": 3
    }
}

# Wake word function - language-aware
def get_wake_word(language="English"):
    """Get wake word in the specified language."""
    return LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["English"])["wake_word"]

def get_wake_word_variations(language="English"):
    """Get list of possible wake word variations for fuzzy matching."""
    return LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["English"])["wake_word_variations"]

# Default wake word (for backward compatibility)
WAKE_WORD = get_wake_word()

def normalize_text_for_matching(text):
    """
    Normalize text for wake word matching by removing punctuation and extra spaces.
    Handles Unicode characters including Chinese and Japanese.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Remove punctuation and special characters, but keep Unicode word characters
    # \w in Python 3 includes Unicode word characters (Chinese, Japanese, etc.)
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Convert to lowercase and strip (lowercase doesn't affect Chinese/Japanese)
    return text.lower().strip()

def check_wake_word_in_transcript(transcript, wake_word, language="English"):
    """
    Check if wake word appears in transcript using multiple matching strategies.
    
    Args:
        transcript: The transcribed text
        wake_word: The expected wake word
        language: Current language setting
        
    Returns:
        Tuple of (found: bool, normalized_transcript: str)
    """
    if not transcript or not transcript.strip():
        return False, ""
    
    # Strategy 1: Direct substring match (case-insensitive)
    transcript_lower = transcript.lower().strip()
    if wake_word.lower() in transcript_lower:
        return True, transcript_lower
    
    # Strategy 2: Normalized matching (remove punctuation)
    normalized_transcript = normalize_text_for_matching(transcript)
    normalized_wake_word = normalize_text_for_matching(wake_word)
    if normalized_wake_word in normalized_transcript:
        return True, transcript_lower
    
    # Strategy 3: Check for variations (e.g., "ola" for "hola")
    variations = get_wake_word_variations(language)
    for variation in variations:
        variation_normalized = normalize_text_for_matching(variation)
        if variation_normalized in normalized_transcript:
            logger.debug(f"✅ Found wake word variation '{variation}' in transcript")
            return True, transcript_lower
    
    # Strategy 4: Word boundary matching (check if wake word is a complete word)
    # This handles cases where "hola" might be part of "hola mundo" but not "holamundo"
    words = normalized_transcript.split()
    normalized_wake_word = normalize_text_for_matching(wake_word)
    if normalized_wake_word in words:
        return True, transcript_lower
    
    # Strategy 5: Fuzzy matching - check if any word starts with the wake word
    # This handles cases where Whisper might add extra characters
    for word in words:
        if word.startswith(normalized_wake_word) or normalized_wake_word.startswith(word):
            # Only match if the difference is small (1-2 characters)
            if abs(len(word) - len(normalized_wake_word)) <= 2:
                logger.debug(f"✅ Found fuzzy wake word match: '{word}' matches '{wake_word}'")
                return True, transcript_lower
    
    return False, transcript_lower

# Word limit for LLM responses (enforced post-processing)
MAX_RESPONSE_WORDS = 30  # Maximum number of words in responses (for word-based languages)

# RMS energy threshold for filtering weak audio (background noise)
# Audio with RMS below this threshold will be ignored
# Typical values: 0.01-0.05 for normalized audio (range -1 to 1)
# Lower values = more sensitive (allows quieter audio)
# Higher values = less sensitive (filters out more background noise)
RMS_ENERGY_THRESHOLD = 0.02  # Default: filter out audio with RMS < 0.02

# System prompt for LLM - language-aware
def get_system_prompt(language="English"):
    """Get system prompt in the specified language."""
    config = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["English"])
    template = config["system_prompt_template"]
    
    if config["uses_word_limit"]:
        return template.format(limit=MAX_RESPONSE_WORDS)
    else:
        # For sentence-based languages, the template already includes the instruction
        return template

# Default system prompt (for backward compatibility)
SYSTEM_PROMPT = get_system_prompt()

# TTS options function - language-aware
def get_tts_options(language="English"):
    """Get TTS options based on language preference."""
    config = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["English"])
    return KokoroTTSOptions(
        voice=config["tts_voice"],
        lang=config["tts_lang"],
        speed=1.0
    )

# Whisper-based STT function
def transcribe_with_whisper(audio, language="English"):
    """
    Transcribe audio using Whisper model.
    
    Args:
        audio: Audio tuple (sample_rate, audio_array) or audio array
        language: Language preference ("English" or "Spanish") for better accuracy
        
    Returns:
        Transcribed text string
    """
    try:
        # Handle tuple format (sample_rate, audio_array)
        if isinstance(audio, tuple):
            sr, audio_np = audio
        else:
            # Assume default sample rate if not provided
            audio_np = audio
            sr = 16000  # Default sample rate
        
        # Ensure audio is a numpy array
        if not isinstance(audio_np, np.ndarray):
            audio_np = np.array(audio_np)
        
        # Ensure audio is 1D (mono) - Whisper expects 1D array
        if audio_np.ndim > 1:
            # If it's 2D (channels, samples), take the first channel or average
            if audio_np.shape[0] == 1:
                audio_np = audio_np[0]
            elif audio_np.shape[1] == 1:
                audio_np = audio_np[:, 0]
            else:
                # Average across channels to create mono
                audio_np = np.mean(audio_np, axis=0)
        
        # Convert to float32 if needed
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)
        
        # Normalize to [-1, 1] range if needed
        if audio_np.dtype == np.int16:
            audio_np = audio_np.astype(np.float32) / 32768.0
        elif audio_np.dtype == np.int32:
            audio_np = audio_np.astype(np.float32) / 2147483648.0
        elif audio_np.max() > 1.0 or audio_np.min() < -1.0:
            # Normalize if values are outside [-1, 1] range
            max_val = np.abs(audio_np).max()
            if max_val > 0:
                audio_np = audio_np / max_val
        
        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sr != 16000:
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
        
        # Determine language code for Whisper
        config = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["English"])
        language_code = config["whisper_lang"]
        
        # Transcribe with Whisper
        logger.debug(f"🔊 Transcribing audio: shape={audio_np.shape}, dtype={audio_np.dtype}, language={language_code}")
        result = whisper_model.transcribe(
            audio_np,
            language=language_code,
            task="transcribe",
            fp16=False  # Use FP32 for compatibility
        )
        
        logger.debug(f"🔊 Whisper result type: {type(result)}, keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
        
        # Handle different return types from Whisper
        # Whisper typically returns a dict with "text" key, but handle edge cases
        if isinstance(result, dict):
            transcript = result.get("text", "").strip()
        elif isinstance(result, list):
            # If result is a list, try to get text from first item
            if len(result) > 0 and isinstance(result[0], dict):
                transcript = result[0].get("text", "").strip()
            else:
                logger.warning(f"⚠️ Unexpected Whisper result format (list): {result}")
                transcript = ""
        else:
            logger.warning(f"⚠️ Unexpected Whisper result type: {type(result)}, value: {result}")
            transcript = ""
        
        if not transcript:
            logger.warning("⚠️ Whisper returned empty transcript")
        
        logger.debug(f"🎤 Whisper transcript ({language}): {transcript!r}")
        return transcript
        
    except Exception as e:
        logger.error(f"❌ Error in Whisper transcription: {e}")
        logger.exception(e)
        return ""

# LLM Model configuration
LLM_MODEL = "llama3.2:3b"
#LLM_MODEL = "gemma3:1b"
# Speaking state - tracks whether the system is currently generating/speaking TTS
# Use a lock to ensure thread-safe access
speaking_lock = threading.Lock()
is_speaking = False

# Thread pool executor for parallel LLM processing
llm_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm_worker")

# RAG Database - cached for performance
_rag_db = None
CHROMA_PATH = "chroma"
RAG_K = 1  # Number of documents to retrieve
MAX_CONTEXT_LENGTH = 2000  # Maximum context length in characters

# HTTP server for serving donation page
_donation_server = None
_donation_server_port = 8001
_donation_server_thread = None

# HTML file to open with the new button (change this to your desired HTML file)
HTML_FILE_TO_OPEN = "donation.html"  # Change this to your HTML filename


def get_local_ip_address():
    """
    Get the local IP address of this computer on the network.
    
    Returns:
        str: Local IP address (e.g., "192.168.1.100") or "localhost" if unable to determine
    """
    try:
        # Connect to a remote address to determine the local IP
        # This doesn't actually send data, just determines which interface would be used
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Connect to a non-routable address (doesn't actually connect)
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
        s.close()
        return ip_address
    except Exception as e:
        logger.warning(f"⚠️ Could not determine local IP address: {e}. Using localhost.")
        return "localhost"


def get_rag_db():
    """Get cached RAG database instance to avoid reinitializing every time."""
    global _rag_db
    if _rag_db is None:
        try:
            embedding_function = get_embedding_function()
            _rag_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
            logger.info("✅ RAG database initialized")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize RAG database: {e}")
            _rag_db = None
    return _rag_db


def query_rag(query_text: str) -> str:
    """
    Query the RAG database for relevant context.
    
    Args:
        query_text: The user's query/question
        
    Returns:
        Context string with relevant information, or empty string if no context found
    """
    db = get_rag_db()
    if db is None:
        return ""
    
    try:
        # Search the database for relevant documents
        results = db.similarity_search_with_score(query_text, k=RAG_K)
        
        if not results:
            logger.debug("No relevant documents found in RAG database")
            return ""
        
        # Build context from retrieved documents
        context_parts = []
        total_length = 0
        
        for doc, score in results:
            content = doc.page_content
            if total_length + len(content) > MAX_CONTEXT_LENGTH:
                # Truncate the last document if needed
                remaining = MAX_CONTEXT_LENGTH - total_length
                if remaining > 100:  # Only add if there's meaningful content left
                    content = content[:remaining] + "..."
                    context_parts.append(content)
                break
            context_parts.append(content)
            total_length += len(content)
        
        if not context_parts:
            return ""
        
        context_text = "\n\n---\n\n".join(context_parts)
        logger.debug(f"Retrieved {len(context_parts)} relevant documents from RAG database")
        return context_text
        
    except Exception as e:
        logger.error(f"❌ Error querying RAG database: {e}")
        return ""


def truncate_to_sentence_limit(text, max_sentences=3):
    """
    Truncate text to a maximum number of sentences.
    
    Args:
        text: Text to truncate
        max_sentences: Maximum number of sentences (default: 3)
        
    Returns:
        Truncated text with ellipsis if truncated
    """
    if not text:
        return text
    
    # Split by sentence endings (., !, ?, 。, ！, ？)
    # Pattern matches: . ! ? 。 ！ ？ followed by space or end of string
    sentence_pattern = r'([.!?。！？]+(?:\s+|$))'
    sentences = re.split(sentence_pattern, text)
    
    # Recombine sentences with their punctuation
    sentence_parts = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence_parts.append(sentences[i] + sentences[i + 1])
        else:
            sentence_parts.append(sentences[i])
    if len(sentences) % 2 == 1:
        sentence_parts.append(sentences[-1])
    
    # Filter out empty sentences
    sentence_parts = [s.strip() for s in sentence_parts if s.strip()]
    
    if len(sentence_parts) <= max_sentences:
        return text
    
    # Truncate to max_sentences
    truncated = " ".join(sentence_parts[:max_sentences])
    if len(sentence_parts) > max_sentences:
        truncated += "..."
    
    logger.debug(f"✂️ Truncated response from {len(sentence_parts)} sentences to {max_sentences} sentences")
    return truncated


def truncate_to_word_limit(text, max_words=MAX_RESPONSE_WORDS):
    """
    Truncate text to a maximum number of words.
    
    Args:
        text: Text to truncate
        max_words: Maximum number of words (default: MAX_RESPONSE_WORDS)
        
    Returns:
        Truncated text with ellipsis if truncated
    """
    if not text:
        return text
    
    words = text.split()
    if len(words) <= max_words:
        return text
    
    # Truncate to max_words and add ellipsis
    truncated = " ".join(words[:max_words])
    # Add ellipsis only if we actually truncated
    if len(words) > max_words:
        truncated += "..."
    
    logger.debug(f"✂️ Truncated response from {len(words)} words to {max_words} words")
    return truncated


def truncate_response(text, language="English"):
    """
    Truncate response based on language-specific limits (words or sentences).
    
    Args:
        text: Text to truncate
        language: Language preference
        
    Returns:
        Truncated text
    """
    config = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["English"])
    
    if config["uses_word_limit"]:
        return truncate_to_word_limit(text, MAX_RESPONSE_WORDS)
    else:
        max_sentences = config.get("max_sentences", 3)
        return truncate_to_sentence_limit(text, max_sentences)


def clean_text_for_tts(text):
    return text
    """Remove emojis and special characters that TTS models can't handle well."""
    # Remove emojis (Unicode emoji ranges)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub("", text)
    
    # Remove markdown formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
    text = re.sub(r'`(.*?)`', r'\1', text)        # Code
    
    # Clean up extra whitespace
    text = re.sub(r'\n+', ' ', text)  # Replace newlines with space
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
    text = text.strip()
    
    return text


# Streaming audio generation strategies
# Options:
#   "full"     - Generate TTS for entire response at once (current default behavior)
#   "sentence" - Stream TTS sentence by sentence (faster perceived response, recommended)
#   "phrase"   - Stream TTS phrase by phrase (split by commas, semicolons, etc.)
#   "word"     - Stream TTS word by word or in small word groups (most responsive but may sound choppy)
STREAMING_MODE = "sentence"  # Change this to switch streaming strategies

def stream_tts_by_sentences(text, tts_options):
    """
    Stream TTS audio sentence by sentence for faster perceived response time.
    
    Args:
        text: Full text to convert to speech
        tts_options: TTS options
        
    Yields:
        Audio chunks as they're generated
    """
    # Split by sentence endings (., !, ?) but keep the punctuation
    sentences = re.split(r'([.!?]+)', text)
    # Recombine sentences with their punctuation
    sentence_parts = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence_parts.append(sentences[i] + sentences[i + 1])
        else:
            sentence_parts.append(sentences[i])
    if len(sentences) % 2 == 1:
        sentence_parts.append(sentences[-1])
    
    # Filter out empty sentences
    sentence_parts = [s.strip() for s in sentence_parts if s.strip()]
    
    for sentence in sentence_parts:
        cleaned_sentence = clean_text_for_tts(sentence)
        if cleaned_sentence:
            logger.debug(f"🔊 Streaming sentence: {cleaned_sentence[:50]}...")
            for audio_chunk in tts_model.stream_tts_sync(cleaned_sentence, options=tts_options):
                yield audio_chunk


def stream_tts_by_phrases(text, tts_options, min_chunk_length=20):
    """
    Stream TTS audio phrase by phrase (split by commas, semicolons, etc.).
    
    Args:
        text: Full text to convert to speech
        tts_options: TTS options
        min_chunk_length: Minimum characters per chunk (default: 20)
        
    Yields:
        Audio chunks as they're generated
    """
    # Split by phrase boundaries (commas, semicolons, colons, periods)
    phrases = re.split(r'([,;:\.!?]+)', text)
    # Recombine phrases with their punctuation
    phrase_parts = []
    for i in range(0, len(phrases) - 1, 2):
        if i + 1 < len(phrases):
            phrase_parts.append(phrases[i] + phrases[i + 1])
        else:
            phrase_parts.append(phrases[i])
    if len(phrases) % 2 == 1:
        phrase_parts.append(phrases[-1])
    
    # Filter and combine small phrases
    current_chunk = ""
    for phrase in phrase_parts:
        phrase = phrase.strip()
        if not phrase:
            continue
        current_chunk += phrase + " "
        if len(current_chunk) >= min_chunk_length:
            cleaned_chunk = clean_text_for_tts(current_chunk)
            if cleaned_chunk:
                logger.debug(f"🔊 Streaming phrase: {cleaned_chunk[:50]}...")
                for audio_chunk in tts_model.stream_tts_sync(cleaned_chunk, options=tts_options):
                    yield audio_chunk
            current_chunk = ""
    
    # Yield remaining chunk
    if current_chunk.strip():
        cleaned_chunk = clean_text_for_tts(current_chunk)
        if cleaned_chunk:
            for audio_chunk in tts_model.stream_tts_sync(cleaned_chunk, options=tts_options):
                yield audio_chunk


def stream_tts_by_words(text, tts_options, words_per_chunk=10):
    """
    Stream TTS audio word by word or in small word groups.
    
    Args:
        text: Full text to convert to speech
        tts_options: TTS options
        words_per_chunk: Number of words per chunk (default: 10)
        
    Yields:
        Audio chunks as they're generated
    """
    words = text.split()
    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i:i + words_per_chunk])
        cleaned_chunk = clean_text_for_tts(chunk)
        if cleaned_chunk:
            logger.debug(f"🔊 Streaming word chunk: {cleaned_chunk[:50]}...")
            for audio_chunk in tts_model.stream_tts_sync(cleaned_chunk, options=tts_options):
                yield audio_chunk


def stream_tts_full(text, tts_options):
    """
    Stream TTS audio for the full text (current behavior).
    
    Args:
        text: Full text to convert to speech
        tts_options: TTS options
        
    Yields:
        Audio chunks as they're generated
    """
    cleaned_text = clean_text_for_tts(text)
    if cleaned_text:
        logger.debug(f"🔊 Streaming full text: {cleaned_text[:50]}...")
        for audio_chunk in tts_model.stream_tts_sync(cleaned_text, options=tts_options):
            yield audio_chunk


def stream_tts_with_strategy(text, tts_options, strategy=None):
    """
    Stream TTS audio using the specified strategy.
    
    Args:
        text: Full text to convert to speech
        tts_options: TTS options
        strategy: Streaming strategy ("full", "sentence", "phrase", "word")
                  If None, uses global STREAMING_MODE
        
    Yields:
        Audio chunks as they're generated
    """
    strategy = strategy or STREAMING_MODE
    
    if strategy == "sentence":
        yield from stream_tts_by_sentences(text, tts_options)
    elif strategy == "phrase":
        yield from stream_tts_by_phrases(text, tts_options)
    elif strategy == "word":
        yield from stream_tts_by_words(text, tts_options)
    else:  # "full" or default
        yield from stream_tts_full(text, tts_options)


def process_text_input(text, language="English"):
    """Process text input from the user (bypasses wake word and STT).
    Returns response text only (no audio for text input).
    
    Args:
        text: User's text input
        language: Language preference
    """
    if not text or not text.strip():
        logger.debug("⏸️ Empty text input, ignoring...")
        return ""
    
    logger.info(f"📝 Processing text input: {text} (Language: {language})")
    
    try:
        # Get language configuration
        config = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["English"])
        
        # Get language-specific system prompt
        system_prompt = get_system_prompt(language)
        
        # Query RAG database for relevant context
        user_query = text.strip()
        context = query_rag(user_query)
        
        # Build user message with context if available
        if context:
            context_instruction = config["context_instruction"]
            user_query_with_context = f"{user_query}\n\n{context_instruction}{context}"
        else:
            user_query_with_context = user_query
        
        # Add word/sentence limit instruction to the end of user prompt
        if config["uses_word_limit"]:
            word_limit_reminder = config["word_limit_reminder_template"].format(limit=MAX_RESPONSE_WORDS)
        else:
            max_sentences = config.get("max_sentences", 3)
            word_limit_reminder = config["word_limit_reminder_template"].format(limit=max_sentences)
        user_query_with_context = user_query_with_context + word_limit_reminder
        
        # Send to LLM with system prompt and context in user message
        response = chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {"role": "user", "content": user_query_with_context}
            ]
        )
        response_text = response["message"]["content"]
        logger.debug(f"🤖 LLM Response (raw): {response_text}")
        
        # Truncate response based on language-specific limits
        response_text = truncate_response(response_text, language)
        
        return response_text
        
    except Exception as e:
        logger.error(f"❌ Error in process_text_input function: {type(e).__name__}: {e}")
        logger.exception(e)
        config = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["English"])
        error_msg = "Error: " + str(e)
        return error_msg


def echo(audio):
    """Wrapper function that checks for wake word before processing audio."""
    global wake_word_detected, is_speaking, current_language
    
    try:
        logger.debug(f"🎤 Received audio input, type: {type(audio)}, length: {len(audio) if hasattr(audio, '__len__') else 'N/A'}")
        
        # Check if system is currently speaking - if so, ignore this input
        with speaking_lock:
            if is_speaking:
                logger.debug("🔇 System is currently speaking, ignoring new audio input")
                return
        
        # RMS energy filtering: filter out weak audio (likely background noise)
        try:
            # Extract audio array from tuple format (sample_rate, audio_array) or use directly
            if isinstance(audio, tuple):
                _, audio_np = audio
            else:
                audio_np = audio
            
            # Ensure audio is a numpy array
            if not isinstance(audio_np, np.ndarray):
                audio_np = np.array(audio_np)
            
            # Ensure audio is 1D (mono) for RMS calculation
            if audio_np.ndim > 1:
                # If it's 2D (channels, samples), take the first channel or average
                if audio_np.shape[0] == 1:
                    audio_np = audio_np[0]
                elif audio_np.shape[1] == 1:
                    audio_np = audio_np[:, 0]
                else:
                    # Average across channels to create mono
                    audio_np = np.mean(audio_np, axis=0)
            
            # Convert to float32 if needed for RMS calculation
            if audio_np.dtype != np.float32:
                if audio_np.dtype == np.int16:
                    audio_np = audio_np.astype(np.float32) / 32768.0
                elif audio_np.dtype == np.int32:
                    audio_np = audio_np.astype(np.float32) / 2147483648.0
                else:
                    audio_np = audio_np.astype(np.float32)
            
            # Normalize to [-1, 1] range if needed (for consistent RMS calculation)
            if audio_np.max() > 1.0 or audio_np.min() < -1.0:
                max_val = np.abs(audio_np).max()
                if max_val > 0:
                    audio_np = audio_np / max_val
            
            # Calculate RMS energy: sqrt(mean(square(audio)))
            if len(audio_np) > 0:
                rms_energy = np.sqrt(np.mean(audio_np ** 2))
                logger.debug(f"🔊 RMS energy: {rms_energy:.6f} (threshold: {RMS_ENERGY_THRESHOLD})")
                
                # Filter out weak audio (likely background noise)
                if rms_energy < RMS_ENERGY_THRESHOLD:
                    logger.debug(f"⏸️ Audio filtered out: RMS energy ({rms_energy:.6f}) below threshold ({RMS_ENERGY_THRESHOLD})")
                    return
            else:
                # Empty audio array, ignore it
                logger.debug("⏸️ Empty audio array, ignoring...")
                return
                
        except Exception as e:
            # If RMS calculation fails, log warning but continue processing
            # (don't break the entire flow if there's an issue with RMS calculation)
            logger.warning(f"⚠️ Error calculating RMS energy: {e}. Continuing with audio processing...")
        
        # Get current wake word based on language
        wake_word = get_wake_word(current_language)
        
        # Wake word not detected yet - check for it
        # Note: We need to transcribe to check for the wake word, but we won't process
        # the audio or send it to the LLM unless the wake word is detected
        logger.debug(f"🔍 Checking for wake word '{wake_word}' (audio will be ignored if not found)...")
        transcript = transcribe_with_whisper(audio, current_language)
        logger.info(f"🎤 Transcript result (Language: {current_language}): '{transcript}'")
        logger.debug(f"🎤 Transcript (raw): {transcript!r}")
        
        if not transcript or not transcript.strip():
            # Empty transcript, ignore it completely
            logger.debug("⏸️ Empty transcript, ignoring audio (no wake word check needed).")
            return
        
        # Use improved wake word detection with multiple strategies
        found, transcript_lower = check_wake_word_in_transcript(transcript, wake_word, current_language)
        
        if found:
            wake_word_detected = True
            logger.info(f"👋 Wake word '{wake_word}' detected in transcript: '{transcript}'")
            
            # Extract the message after the wake word if there is one
            # Try multiple ways to split in case the wake word was transcribed differently
            user_message = None
            normalized_transcript = normalize_text_for_matching(transcript)
            normalized_wake_word = normalize_text_for_matching(wake_word)
            
            # Try to find where the wake word appears and extract text after it
            wake_word_variations = get_wake_word_variations(current_language)
            for variation in wake_word_variations:
                variation_normalized = normalize_text_for_matching(variation)
                if variation_normalized in normalized_transcript:
                    # Find the position in the original transcript (case-insensitive)
                    variation_lower = variation.lower()
                    if variation_lower in transcript_lower:
                        parts = transcript_lower.split(variation_lower, 1)
                        if len(parts) > 1:
                            user_message = parts[1].strip()
                            break
                    # Also try in normalized version
                    parts = normalized_transcript.split(variation_normalized, 1)
                    if len(parts) > 1:
                        user_message = parts[1].strip()
                        break
            
            # If we still don't have a message, try word-based extraction
            if user_message is None:
                words = normalized_transcript.split()
                try:
                    wake_word_index = words.index(normalized_wake_word)
                    if wake_word_index < len(words) - 1:
                        user_message = " ".join(words[wake_word_index + 1:])
                except ValueError:
                    # Wake word not found as a separate word, might be part of another word
                    pass
            
            # Clean up the user message
            if user_message:
                user_message = user_message.strip()
                if not user_message:
                    user_message = None
            
            if user_message:
                # User said "hello" followed by a message, process the message
                logger.info(f"📝 Processing message after wake word: {user_message}")
                
                # Reset wake word after processing
                wake_word_detected = False
                logger.debug("🔄 Wake word reset. Waiting for 'hello' for next interaction.")
                
                # Mark as speaking before TTS generation
                with speaking_lock:
                    is_speaking = True
                logger.debug("🔊 Marked as speaking - new inputs will be ignored until finished")
                
                try:
                    # Get language configuration
                    config = LANGUAGE_CONFIG.get(current_language, LANGUAGE_CONFIG["English"])
                    
                    # Get language-specific system prompt
                    system_prompt = get_system_prompt(current_language)
                    
                    # Query RAG database for relevant context
                    context = query_rag(user_message)
                    
                    # Build user message with context if available
                    if context:
                        context_instruction = config["context_instruction"]
                        user_message_with_context = f"{user_message}\n\n{context_instruction}{context}"
                    else:
                        user_message_with_context = user_message
                    
                    # Add word/sentence limit instruction to the end of user prompt
                    if config["uses_word_limit"]:
                        word_limit_reminder = config["word_limit_reminder_template"].format(limit=MAX_RESPONSE_WORDS)
                    else:
                        max_sentences = config.get("max_sentences", 3)
                        word_limit_reminder = config["word_limit_reminder_template"].format(limit=max_sentences)
                    user_message_with_context = user_message_with_context + word_limit_reminder
                    
                    # Submit LLM chat call to thread pool for parallel processing
                    logger.info(f"🚀 Submitting LLM request to thread pool... (Language: {current_language})")
                    llm_future = llm_executor.submit(
                        chat,
                        model=LLM_MODEL,
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {"role": "user", "content": user_message_with_context}
                        ]
                    )
                    
                    # Immediately echo back the user message as audio (confirmation)
                    logger.info(f"🔊 Echoing user message: {user_message}")
                    echo_text = config["echo_message_template"].format(message=user_message)
                    cleaned_echo = clean_text_for_tts(echo_text)
                    
                    # Get language-aware TTS options
                    tts_options = get_tts_options(current_language)
                    for audio_chunk in tts_model.stream_tts_sync(cleaned_echo, options=tts_options):
                        yield audio_chunk
                    
                    # Wait for LLM response (this happens in parallel, so delay is hidden)
                    logger.info("⏳ Waiting for LLM response...")
                    response = llm_future.result()  # This will block until chat() completes
                    response_text = response["message"]["content"]
                    logger.debug(f"🤖 LLM Response (raw): {response_text}")
                    
                    # Truncate response based on language-specific limits
                    response_text = truncate_response(response_text, current_language)
                    
                    # Clean and generate TTS for LLM response
                    cleaned_text = clean_text_for_tts(response_text)
                    logger.info(f"🔊 TTS Text (cleaned): {cleaned_text}")
                    
                    if not cleaned_text or not cleaned_text.strip():
                        config = LANGUAGE_CONFIG.get(current_language, LANGUAGE_CONFIG["English"])
                        cleaned_text = config["empty_response_message"]
                    
                    # Generate and yield TTS audio for LLM response using streaming strategy
                    logger.info(f"🔊 Generating TTS for LLM response (mode: {STREAMING_MODE})...")
                    # Get language-aware TTS options
                    tts_options = get_tts_options(current_language)
                    # Use streaming strategy for faster perceived response time
                    for audio_chunk in stream_tts_with_strategy(response_text, tts_options):
                        yield audio_chunk
                    
                    # Also yield the response text to update the text output box
                    yield AdditionalOutputs(response_text)
                        
                except Exception as e:
                    logger.error(f"❌ Error processing user message: {type(e).__name__}: {e}")
                    logger.exception(e)
                    # Try to yield an error message as audio
                    try:
                        config = LANGUAGE_CONFIG.get(current_language, LANGUAGE_CONFIG["English"])
                        error_msg = config["error_message"]
                        # Get language-aware TTS options
                        tts_options = get_tts_options(current_language)
                        # Use streaming strategy for error messages too
                        for audio_chunk in stream_tts_with_strategy(error_msg, tts_options):
                            yield audio_chunk
                        # Also yield the error message to update the text output box
                        yield AdditionalOutputs(error_msg)
                    except Exception as tts_error:
                        logger.error(f"❌ Failed to generate error TTS: {tts_error}")
                finally:
                    # Mark as finished speaking
                    with speaking_lock:
                        is_speaking = False
                    logger.debug("🔇 Finished speaking2 - ready to accept new inputs")
            else:
                # User just said wake word without a message, greet them
                config = LANGUAGE_CONFIG.get(current_language, LANGUAGE_CONFIG["English"])
                response_text = config["greeting"]
                cleaned_text = clean_text_for_tts(response_text)
                logger.info(f"🔊 TTS Text (cleaned): {cleaned_text}")
                
                # Mark as speaking before TTS generation
                with speaking_lock:
                    is_speaking = True
                logger.debug("🔊 Marked as speaking - new inputs will be ignored until finished")
                
                # Keep wake_word_detected = True so next audio will be processed
                try:
                    # Get language-aware TTS options
                    tts_options = get_tts_options(current_language)
                    # Use streaming strategy for faster perceived response time
                    for audio_chunk in stream_tts_with_strategy(response_text, tts_options, strategy="sentence"):
                        yield audio_chunk
                    
                    # Also yield the response text to update the text output box
                    yield AdditionalOutputs(response_text)
                finally:
                    # Mark as finished speaking
                    with speaking_lock:
                        is_speaking = False
                    logger.debug("🔇 Finished speaking - ready to accept new inputs")
        else:
            # Wake word not detected, ignore this audio completely
            # No processing, no LLM call, no response - audio is discarded
            # Get current wake word for logging
            wake_word = get_wake_word(current_language)
            logger.info(f"⏸️ Wake word '{wake_word}' not found in '{transcript}'. Audio ignored (not recording).")
            return
            
    except Exception as e:
        logger.error(f"❌ Error in echo function: {type(e).__name__}: {e}")
        logger.exception(e)  # Full traceback
        # Reset wake word on error
        wake_word_detected = False


# Configure ReplyOnPause to stop recording when a pause is detected
# AlgoOptions controls pause detection:
# - speech_threshold: if a chunk has less than this many seconds of speech (after user started talking),
#   the user is considered to have stopped speaking, triggering pause detection and stopping recording
# - Lower values = more sensitive (stops recording sooner)
# - Higher values = less sensitive (requires longer pause before stopping)
# - audio_chunk_duration: Duration of audio chunks analyzed by VAD model
#   before considering it a pause and processing the audio
algo_options = AlgoOptions(
    speech_threshold=0.1,  # Stop recording when less than 0.1s of speech detected in a chunk
    started_talking_threshold=0.2,  # User must have at least 0.2s of speech to be considered "started talking"
    audio_chunk_duration=1.2,  # Duration of audio chunks analyzed by VAD model
    max_continuous_speech_s=10.0  # Maximum duration of continuous speech in seconds (stops recording after 10 seconds)
)

def generate_qr_code(url, filename="qr_code.png"):
    """Generate a QR code image from a URL and save it to the current directory."""
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        filepath = os.path.join(os.getcwd(), filename)
        img.save(filepath)
        logger.info(f"✅ QR code saved to: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"❌ Failed to generate QR code: {e}")
        return None


def wait_for_url_and_generate_qr(stream_obj, max_attempts=10, delay=1.0):
    """Wait for the share URL to become available and generate QR code."""
    for _ in range(max_attempts):
        time.sleep(delay)
        # Try multiple ways to get the URL
        share_url = None
        
        # Method 1: Check if launch() returned a result (if called with blocking=False)
        if hasattr(stream_obj.ui, '_launch_server') and stream_obj.ui._launch_server:
            if hasattr(stream_obj.ui._launch_server, 'share_url'):
                share_url = stream_obj.ui._launch_server.share_url
        
        # Method 2: Check stream.ui attributes
        if not share_url:
            for attr in ['share_url', 'public_url', 'url']:
                if hasattr(stream_obj.ui, attr):
                    url_value = getattr(stream_obj.ui, attr)
                    if url_value and isinstance(url_value, str) and url_value.startswith('http'):
                        share_url = url_value
                        break
        
        # Method 3: Check if there's a gradio interface with share_url
        if not share_url and hasattr(stream_obj.ui, 'interface'):
            interface = stream_obj.ui.interface
            if hasattr(interface, 'share_url'):
                share_url = interface.share_url
            elif hasattr(interface, 'share'):
                share_url = interface.share
        
        if share_url:
            logger.info(f"🔗 Share URL found: {share_url}")
            generate_qr_code(share_url)
            return
        
    logger.warning("⚠️ Could not determine share URL after multiple attempts. QR code not generated.")


# Set can_interrupt=False to prevent new audio from interrupting while speaking
stream = Stream(
    ReplyOnPause(echo, algo_options=algo_options, can_interrupt=False), 
    modality="audio", 
    mode="send-receive",
    ui_args={"button_labels": {"start": "AI-Based Documentation"}}
)

# Add text input functionality to the UI
# (handle_text_submit function removed - no longer needed)

# Customize the UI to add text input
with stream.ui:
    gr.Markdown("## Voice Chat Assistant")
    gr.Markdown("You can interact via voice (say 'hello' / 'hola' to start) or type your question below.")
    
    # Language dropdown menu
    with gr.Row():
        language_toggle = gr.Dropdown(
            choices=["English", "Spanish", "Mandarin", "Japanese"],
            value="English",
            label="Language / Idioma / 语言 / 言語",
            info="Select the language for the AI assistant"
        )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Text Input")
            text_input = gr.Textbox(
                label="Type your question here",
                placeholder="Enter your question and press Submit...",
                lines=3
            )
            submit_btn = gr.Button("Submit", variant="primary")
            text_output = gr.Textbox(
                label="Response",
                interactive=False,
                lines=5
            )
    
    # Helper function to get URL for a specific HTML file
    def get_html_file_url_for_file(filename):
        """Get URL for a specific HTML file (ensures server is running)."""
        try:
            # Ensure server is running by calling ensure_server_and_get_url
            # This will start the server if needed (it checks HTML_FILE_TO_OPEN, but server serves all files)
            ensure_server_and_get_url()
            
            # Check if HTML file exists
            html_file_path = os.path.join(os.getcwd(), filename)
            if not os.path.exists(html_file_path):
                logger.warning(f"⚠️ HTML file not found: {html_file_path}")
                return None
            
            # Get local IP address for network access
            local_ip = get_local_ip_address()
            if not local_ip or local_ip == "localhost":
                local_ip = "localhost"
            
            # Create URL for the HTML file
            html_url = f"http://{local_ip}:{_donation_server_port}/{filename}"
            logger.debug(f"📖 Generated URL for {filename}: {html_url}")
            return html_url
        except Exception as e:
            logger.error(f"❌ Error getting URL for {filename}: {e}")
            return None
    
    # Donate button row - placed under audio input area
    with gr.Row():
        # Create a function that returns HTML with a button-styled link
        def get_donate_button_link():
            """Get HTML with a button-styled link that directly opens the URL."""
            html_url = ensure_server_and_get_url()
            
            if html_url is None:
                error_html = f'<div style="color: red; padding: 10px;">Error: HTML file \'{HTML_FILE_TO_OPEN}\' not found in current directory.</div>'
                return error_html
            
            # Return HTML with a link styled to look like a button, showing the URL
            button_link_html = f'''
            <div style="padding: 10px; width: 100%;">
                <a href="{html_url}" target="_blank" 
                   style="display: block; padding: 15px 25px; background-color: #4CAF50; color: white; 
                          text-decoration: none; border-radius: 5px; font-weight: bold; cursor: pointer;
                          border: none; transition: background-color 0.3s; text-align: center; width: 100%;
                          box-sizing: border-box;"
                   onmouseover="this.style.backgroundColor='#45a049'"
                   onmouseout="this.style.backgroundColor='#4CAF50'">
                    <div style="font-size: 18px; margin-bottom: 5px;">Donate</div>
                    <div style="font-size: 12px; opacity: 0.9; word-break: break-all;">{html_url}</div>
                </a>
            </div>
            '''
            return button_link_html
        
        # Create a function for the Animated Artifact button
        def get_animated_artifact_button_link():
            """Get HTML with a button-styled link that opens sora.html."""
            html_url = get_html_file_url_for_file("sora.html")
            
            if html_url is None:
                error_html = f'<div style="color: red; padding: 10px;">Error: HTML file \'sora.html\' not found in current directory.</div>'
                return error_html
            
            # Return HTML with a link styled to look like a button, showing the URL
            button_link_html = f'''
            <div style="padding: 10px; width: 100%;">
                <a href="{html_url}" target="_blank" 
                   style="display: block; padding: 15px 25px; background-color: #2196F3; color: white; 
                          text-decoration: none; border-radius: 5px; font-weight: bold; cursor: pointer;
                          border: none; transition: background-color 0.3s; text-align: center; width: 100%;
                          box-sizing: border-box;"
                   onmouseover="this.style.backgroundColor='#0b7dda'"
                   onmouseout="this.style.backgroundColor='#2196F3'">
                    <div style="font-size: 18px; margin-bottom: 5px;">Animated Artifact</div>
                    <div style="font-size: 12px; opacity: 0.9; word-break: break-all;">{html_url}</div>
                </a>
            </div>
            '''
            return button_link_html
        
        # Initialize the button links with placeholders (will be updated by load event)
        donate_btn_link = gr.HTML(value='<div style="padding: 10px; color: #666;">Loading...</div>')  # HTML component with button-styled link
        animated_artifact_btn_link = gr.HTML(value='<div style="padding: 10px; color: #666;">Loading...</div>')  # HTML component for animated artifact button
    
    # Function to update language preference
    def update_language(language):
        """Update the global language preference."""
        global current_language
        current_language = language
        logger.info(f"🌐 Language changed to: {language}")
    
    # Connect language toggle to update function
    language_toggle.change(
        fn=update_language,
        inputs=language_toggle,
        outputs=None
    )
    
    # Connect the submit button to process text
    def process_text_and_show(text, language):
        """Process text and show response (text only, no audio)."""
        if not text or not text.strip():
            return "", ""
        
        logger.info(f"📝 Processing text: {text} (Language: {language})")
        response_text = process_text_input(text, language)
        
        # Clear the input after processing
        return "", response_text
    
    submit_btn.click(
        fn=process_text_and_show,
        inputs=[text_input, language_toggle],
        outputs=[text_input, text_output]
    )
    
    # Also allow Enter key to submit
    text_input.submit(
        fn=process_text_and_show,
        inputs=[text_input, language_toggle],
        outputs=[text_input, text_output]
    )
    
    # Donate button - starts HTTP server and opens donation.html
    def open_donation():
        """Start HTTP server and open donation.html in a new browser window."""
        global _donation_server, _donation_server_thread
        
        def start_server():
            """Start a simple HTTP server to serve donation files."""
            try:
                # Get the directory where donation.html is located
                serve_directory = os.getcwd()
                original_dir = os.getcwd()
                
                # Change to the serve directory (SimpleHTTPRequestHandler serves from current directory)
                os.chdir(serve_directory)
                
                try:
                    # Create a simple HTTP server
                    handler = http.server.SimpleHTTPRequestHandler
                    
                    # Try to start server on the port
                    with socketserver.TCPServer(("", _donation_server_port), handler) as httpd:
                        _donation_server = httpd
                        logger.info(f"🌐 Started HTTP server on port {_donation_server_port} serving from {serve_directory}")
                        httpd.serve_forever()
                finally:
                    # Restore original directory
                    os.chdir(original_dir)
            except OSError as e:
                # Restore directory on error
                try:
                    os.chdir(original_dir)
                except:
                    pass
                if "Address already in use" in str(e):
                    logger.info(f"ℹ️ Server already running on port {_donation_server_port}")
                else:
                    logger.error(f"❌ Error starting HTTP server: {e}")
            except Exception as e:
                # Restore directory on error
                try:
                    os.chdir(original_dir)
                except:
                    pass
                logger.error(f"❌ Error in HTTP server: {e}")
        
        # Start server in background thread if not already running
        if _donation_server_thread is None or not _donation_server_thread.is_alive():
            _donation_server_thread = threading.Thread(target=start_server, daemon=True)
            _donation_server_thread.start()
            # Give server a moment to start
            time.sleep(0.5)
        
        # Open the donation page in browser
        donation_url = f"http://localhost:{_donation_server_port}/donation.html"
        webbrowser.open(donation_url)
        logger.info(f"📖 Opened donation page: {donation_url}")
        return ""  # Return empty string since button doesn't need to update any outputs
    
    # Helper function to ensure server is running and get the URL
    def ensure_server_and_get_url():
        """Start HTTP server (if not running) and return the URL to the HTML file."""
        global _donation_server, _donation_server_thread
        
        def start_server():
            """Start a simple HTTP server to serve HTML files."""
            try:
                # Get the directory where HTML files are located
                serve_directory = os.getcwd()
                original_dir = os.getcwd()
                
                # Change to the serve directory (SimpleHTTPRequestHandler serves from current directory)
                os.chdir(serve_directory)
                
                try:
                    # Create a simple HTTP server
                    handler = http.server.SimpleHTTPRequestHandler
                    
                    # Try to start server on the port
                    with socketserver.TCPServer(("", _donation_server_port), handler) as httpd:
                        _donation_server = httpd
                        logger.info(f"🌐 Started HTTP server on port {_donation_server_port} serving from {serve_directory}")
                        httpd.serve_forever()
                finally:
                    # Restore original directory
                    os.chdir(original_dir)
            except OSError as e:
                # Restore directory on error
                try:
                    os.chdir(original_dir)
                except:
                    pass
                if "Address already in use" in str(e):
                    logger.info(f"ℹ️ Server already running on port {_donation_server_port}")
                else:
                    logger.error(f"❌ Error starting HTTP server: {e}")
            except Exception as e:
                # Restore directory on error
                try:
                    os.chdir(original_dir)
                except:
                    pass
                logger.error(f"❌ Error in HTTP server: {e}")
        
        try:
            logger.debug("🔍 ensure_server_and_get_url() called")
            # Start server in background thread if not already running
            if _donation_server_thread is None or not _donation_server_thread.is_alive():
                _donation_server_thread = threading.Thread(target=start_server, daemon=True)
                _donation_server_thread.start()
                # Give server a moment to start
                time.sleep(0.5)
            
            # Check if HTML file exists
            html_file_path = os.path.join(os.getcwd(), HTML_FILE_TO_OPEN)
            logger.debug(f"🔍 Checking for HTML file at: {html_file_path}")
            logger.debug(f"🔍 Current working directory: {os.getcwd()}")
            
            if not os.path.exists(html_file_path):
                logger.warning(f"⚠️ HTML file not found: {html_file_path}")
                try:
                    files = os.listdir(os.getcwd())
                    logger.warning(f"⚠️ Current directory contents: {files}")
                except:
                    pass
                return None
            
            # Get local IP address for network access
            try:
                local_ip = get_local_ip_address()
                logger.debug(f"🔍 Local IP address: {local_ip}")
                if not local_ip or local_ip == "localhost":
                    logger.warning(f"⚠️ Could not determine local IP address, using localhost")
                    local_ip = "localhost"
            except Exception as e:
                logger.error(f"❌ Error getting local IP address: {e}")
                local_ip = "localhost"
            
            # Create URL for the HTML file (accessible from other devices on network)
            html_url = f"http://{local_ip}:{_donation_server_port}/{HTML_FILE_TO_OPEN}"
            logger.info(f"📖 Generated URL for HTML file: {html_url}")
            
            return html_url
        except Exception as e:
            logger.error(f"❌ Unexpected error in ensure_server_and_get_url: {e}")
            logger.exception(e)
            return None
    
    # Function to generate HTML link to the HTML file
    def get_html_file_link():
        """Start HTTP server (if not running) and return HTML with clickable link."""
        html_url = ensure_server_and_get_url()
        
        if html_url is None:
            error_html = f'<div style="color: red; padding: 10px;">Error: HTML file \'{HTML_FILE_TO_OPEN}\' not found in current directory.</div>'
            return error_html
        
        # Return HTML with clickable link
        link_html = f'''
        <div style="padding: 10px;">
            <a href="{html_url}" target="_blank" style="color: #0066cc; text-decoration: underline; font-size: 16px; font-weight: bold;">
                Open URL
            </a>
            <span style="color: #666; font-size: 12px; margin-left: 10px;">({html_url})</span>
        </div>
        '''
        return link_html
    
    # No button click handler needed - the donate_btn_link is a direct HTML link
    
    # Set up additional_outputs connection inside UI context
    # Access the WebRTC component from the stream and set up the handler
    webrtc_component = stream.webrtc_component
    if webrtc_component:
        def update_text_output(old_value, new_value):
            """Update text output box with new value from AdditionalOutputs."""
            return new_value
        
        webrtc_component.on_additional_outputs(
            update_text_output,
            inputs=[text_output],
            outputs=[text_output],
            queue=False  # Don't queue this, it's just a UI update
        )
    
    # Load the button links when the page loads (using Blocks-level load event)
    stream.ui.load(
        fn=lambda: (get_donate_button_link(), get_animated_artifact_button_link()),
        inputs=None,
        outputs=[donate_btn_link, animated_artifact_btn_link]
    )

# Initialize RAG database at startup
logger.info("📚 Initializing RAG database...")
get_rag_db()

# Preload Ollama by sending a "hello" message
def preload_ollama():
    """Preload Ollama model by sending a simple 'hello' message."""
    global current_language
    try:
        logger.info("🗣️🔥 Preloading Ollama model...")
        system_prompt = get_system_prompt(current_language)
        response = chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {"role": "user", "content": "hello"}
            ]
        )
        logger.info("✅ Ollama model preloaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ Failed to preload Ollama: {e}")

# Preload Ollama in a background thread so it doesn't block startup
preload_thread = threading.Thread(target=preload_ollama, daemon=True)
preload_thread.start()

# Start a background thread to wait for the URL and generate QR code
qr_thread = threading.Thread(
    target=wait_for_url_and_generate_qr,
    args=(stream,),
    daemon=True
)
qr_thread.start()

# Launch the stream (this blocks)
# The QR code will be generated in the background thread once the URL is available
logger.info("🚀 Starting voice chat application...")
stream.ui.launch(share=True)
