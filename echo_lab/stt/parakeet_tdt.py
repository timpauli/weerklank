"""NVIDIA Parakeet TDT speech-to-text, run locally with nano-parakeet (no NeMo needed)."""

import asyncio
import time
from collections.abc import AsyncGenerator

import numpy as np
import torch
from loguru import logger
from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601

from echo_lab.registry import StageContext

PARAKEET_SAMPLE_RATE = 16000


class ParakeetTDTSTTService(SegmentedSTTService):
    """Transcribes VAD segments with Parakeet TDT.

    Parakeet TDT v3 detects the spoken language by itself, but doesn't report
    which one it heard: transcripts carry no language, and the translation
    stage translates from ``llm.input_language``.
    """

    def __init__(
        self,
        *,
        model: str = "nvidia/parakeet-tdt-0.6b-v3",
        device: str = "cuda:0",
        dtype: str | None = None,
        **kwargs,
    ):
        """Load the model.

        Args:
            model: Hugging Face repo of a Parakeet TDT ``.nemo`` checkpoint.
            device: Torch device, e.g. ``"cuda:0"`` or ``"cpu"``.
            dtype: ``"bfloat16"``, ``"float16"`` or ``"float32"``. Default: bfloat16
                on GPUs that support it, otherwise float16 autocast.
            **kwargs: Additional arguments passed to SegmentedSTTService.
        """
        super().__init__(
            sample_rate=PARAKEET_SAMPLE_RATE,
            settings=STTSettings(model=model, language=None),  # auto-detected
            ttfs_p99_latency=0.5,
            **kwargs,
        )

        from nano_parakeet import from_pretrained

        logger.info(f"Loading Parakeet TDT '{model}' on {device}")
        # from_pretrained also warms the model up (and captures its CUDA graph).
        self._model = from_pretrained(
            model_name=model, device=device, dtype=getattr(torch, dtype) if dtype else None
        )

    @property
    def wants_wav_segments(self) -> bool:
        """Receive raw 16-bit PCM segments, which go straight into the model."""
        return False

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe one segment of 16 kHz mono 16-bit PCM."""
        started = time.perf_counter()
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        text = (await asyncio.to_thread(self._model.transcribe, samples)).strip()

        duration = len(samples) / PARAKEET_SAMPLE_RATE
        elapsed = time.perf_counter() - started
        if not text:
            logger.debug(f"Parakeet: no speech in {duration:.1f}s segment ({elapsed:.2f}s)")
            return
        logger.info(f"Heard [{duration:.1f}s audio, STT {elapsed:.2f}s]: {text}")
        yield TranscriptionFrame(text, self._user_id, time_now_iso8601())


def create(ctx: StageContext):
    """Build the Parakeet TDT service.

    Options (``stt.parakeet_tdt``): model, device, dtype (see ParakeetTDTSTTService).
    """
    return ParakeetTDTSTTService(**ctx.options)
