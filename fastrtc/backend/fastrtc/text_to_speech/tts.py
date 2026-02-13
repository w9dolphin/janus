import asyncio
import importlib.util
import re
import threading
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar

if TYPE_CHECKING:
    from kokoro import KPipeline

import numpy as np
import torch
from numpy.typing import NDArray

from fastrtc.utils import async_aggregate_bytes_to_16bit


class TTSOptions:
    pass


T = TypeVar("T", bound=TTSOptions, contravariant=True)


class TTSModel(Protocol[T]):
    def tts(
        self, text: str, options: T | None = None
    ) -> tuple[int, NDArray[np.float32] | NDArray[np.int16]]: ...

    def stream_tts(
        self, text: str, options: T | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.float32] | NDArray[np.int16]], None]: ...

    def stream_tts_sync(
        self, text: str, options: T | None = None
    ) -> Generator[tuple[int, NDArray[np.float32] | NDArray[np.int16]], None, None]: ...


@dataclass
class KokoroTTSOptions(TTSOptions):
    voice: str = "af_heart"
    speed: float = 1.0
    lang: str = "en-us"  # Will be mapped to KPipeline lang_code (a, e, z, j, etc.)


@lru_cache
def get_tts_model(
    model: Literal["kokoro", "cartesia"] = "kokoro", **kwargs
) -> TTSModel:
    if model == "kokoro":
        m = KokoroTTSModel()
        m.tts("Hello, world!")
        return m
    elif model == "cartesia":
        m = CartesiaTTSModel(api_key=kwargs.get("cartesia_api_key", ""))
        return m
    else:
        raise ValueError(f"Invalid model: {model}")


class KokoroTTSModel(TTSModel):
    """TTS Model using kokoro KPipeline (same as test_chinese.py)."""
    
    # Language code mapping: lang parameter -> KPipeline lang_code
    # KPipeline uses single-character codes: a=English, e=Spanish, z=Mandarin, j=Japanese
    LANG_CODE_MAP = {
        "en-us": "a",  # American English
        "en": "a",     # English
        "es": "e",     # Spanish
        "cmn": "z",    # Mandarin Chinese (ISO 639-3)
        "z": "z",      # Mandarin Chinese (KPipeline code)
        "ja": "j",     # Japanese (ISO 639-1)
        "j": "j",      # Japanese (KPipeline code)
    }
    
    # KPipeline sample rate is fixed at 24000 Hz
    SAMPLE_RATE = 24000
    
    def __init__(self):
        # Cache pipelines per language code
        self._pipelines: dict[str, KPipeline] = {}
        self._lock = threading.Lock()
    
    def _get_pipeline(self, lang_code: str):
        """Get or create a KPipeline instance for the given language code."""
        from kokoro import KPipeline
        
        with self._lock:
            if lang_code not in self._pipelines:
                pipeline = KPipeline(lang_code=lang_code)
                self._pipelines[lang_code] = pipeline
            return self._pipelines[lang_code]
    
    def _map_lang_to_lang_code(self, lang: str) -> str:
        """Map language parameter to KPipeline lang_code."""
        return self.LANG_CODE_MAP.get(lang, "a")  # Default to English

    def tts(
        self, text: str, options: KokoroTTSOptions | None = None
    ) -> tuple[int, NDArray[np.float32]]:
        """Generate TTS audio for the full text."""
        options = options or KokoroTTSOptions()
        
        # Map language to KPipeline lang_code
        lang_code = self._map_lang_to_lang_code(options.lang)
        pipeline = self._get_pipeline(lang_code)
        
        # Generate audio using KPipeline
        generator = pipeline(
            text,
            voice=options.voice,
            speed=options.speed,
            split_pattern=r'\n+'
        )
        
        # Collect all audio chunks
        audio_chunks = []
        for _graphemes, _phonemes, audio_tensor in generator:
            # Convert torch.Tensor to numpy array
            if isinstance(audio_tensor, torch.Tensor):
                audio_np = audio_tensor.detach().cpu().numpy().astype(np.float32)
            else:
                audio_np = np.array(audio_tensor, dtype=np.float32)
            audio_chunks.append(audio_np)
        
        # Concatenate all chunks
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
        else:
            full_audio = np.array([], dtype=np.float32)
        
        return self.SAMPLE_RATE, full_audio

    async def stream_tts(
        self, text: str, options: KokoroTTSOptions | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.float32]], None]:
        """Stream TTS audio asynchronously."""
        options = options or KokoroTTSOptions()
        
        # Map language to KPipeline lang_code
        lang_code = self._map_lang_to_lang_code(options.lang)
        pipeline = self._get_pipeline(lang_code)
        
        # Split text into sentences for streaming
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        
        for s_idx, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            # Generate audio for this sentence using KPipeline
            generator = pipeline(
                sentence,
                voice=options.voice,
                speed=options.speed,
                split_pattern=r'\n+'
            )
            
            chunk_idx = 0
            for _graphemes, _phonemes, audio_tensor in generator:
                # Convert torch.Tensor to numpy array
                if isinstance(audio_tensor, torch.Tensor):
                    audio_np = audio_tensor.detach().cpu().numpy().astype(np.float32)
                else:
                    audio_np = np.array(audio_tensor, dtype=np.float32)
                
                # Add small pause between sentences (except first chunk of first sentence)
                if s_idx != 0 and chunk_idx == 0:
                    # Add a small silence gap between sentences
                    silence_samples = self.SAMPLE_RATE // 7  # ~143ms of silence
                    yield self.SAMPLE_RATE, np.zeros(silence_samples, dtype=np.float32)
                
                chunk_idx += 1
                yield self.SAMPLE_RATE, audio_np

    def stream_tts_sync(
        self, text: str, options: KokoroTTSOptions | None = None
    ) -> Generator[tuple[int, NDArray[np.float32]], None, None]:
        loop = asyncio.new_event_loop()

        # Use the new loop to run the async generator
        iterator = self.stream_tts(text, options).__aiter__()
        while True:
            try:
                yield loop.run_until_complete(iterator.__anext__())
            except StopAsyncIteration:
                break


@dataclass
class CartesiaTTSOptions(TTSOptions):
    voice: str = "71a7ad14-091c-4e8e-a314-022ece01c121"
    language: str = "en"
    emotion: list[str] = field(default_factory=list)
    cartesia_version: str = "2024-06-10"
    model: str = "sonic-2"
    sample_rate: int = 22_050


class CartesiaTTSModel(TTSModel):
    def __init__(self, api_key: str):
        if importlib.util.find_spec("cartesia") is None:
            raise RuntimeError(
                "cartesia is not installed. Please install it using 'pip install cartesia'."
            )
        from cartesia import AsyncCartesia

        self.client = AsyncCartesia(api_key=api_key)

    async def stream_tts(
        self, text: str, options: CartesiaTTSOptions | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.int16]], None]:
        options = options or CartesiaTTSOptions()

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        for sentence in sentences:
            if not sentence.strip():
                continue
            async for output in async_aggregate_bytes_to_16bit(
                self.client.tts.bytes(
                    model_id="sonic-2",
                    transcript=sentence,
                    voice={"id": options.voice},  # type: ignore
                    language="en",
                    output_format={
                        "container": "raw",
                        "sample_rate": options.sample_rate,
                        "encoding": "pcm_s16le",
                    },
                )
            ):
                yield options.sample_rate, np.frombuffer(output, dtype=np.int16)

    def stream_tts_sync(
        self, text: str, options: CartesiaTTSOptions | None = None
    ) -> Generator[tuple[int, NDArray[np.int16]], None, None]:
        loop = asyncio.new_event_loop()

        iterator = self.stream_tts(text, options).__aiter__()
        while True:
            try:
                yield loop.run_until_complete(iterator.__anext__())
            except StopAsyncIteration:
                break

    def tts(
        self, text: str, options: CartesiaTTSOptions | None = None
    ) -> tuple[int, NDArray[np.int16]]:
        loop = asyncio.new_event_loop()
        buffer = np.array([], dtype=np.int16)

        options = options or CartesiaTTSOptions()

        iterator = self.stream_tts(text, options).__aiter__()
        while True:
            try:
                _, chunk = loop.run_until_complete(iterator.__anext__())
                buffer = np.concatenate([buffer, chunk])
            except StopAsyncIteration:
                break
        return options.sample_rate, buffer
