"""Base class for the translation services in the LLM slot of the pipeline."""

import asyncio
import time
from abc import abstractmethod

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService
from pipecat.services.settings import ServiceSettings


class TranslationService(AIService):
    """Translates each final transcription and passes the translation to the TTS.

    The translation goes out the way an LLM response does
    (``LLMFullResponseStartFrame``, ``LLMTextFrame``, ``LLMFullResponseEndFrame``),
    so the TTS service splits it into sentences and starts speaking the first
    one while the rest is synthesized. The transcription itself is passed on too.

    Subclasses load their model in ``__init__`` and implement :meth:`_translate`.
    """

    def __init__(self, *, model: str, source_language: str, target_language: str, **kwargs):
        """Initialize the service.

        Args:
            model: The translation model.
            source_language: Language translated from, e.g. ``"de"``.
            target_language: Language translated into, e.g. ``"nl"``.
            **kwargs: Additional arguments passed to AIService.
        """
        super().__init__(settings=ServiceSettings(model=model), **kwargs)
        self.source_language = source_language
        self.target_language = target_language

    @abstractmethod
    def _translate(self, text: str) -> str:
        """Translate ``text`` from ``source_language`` into ``target_language``.

        Runs in a worker thread, so audio keeps flowing meanwhile.
        """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            await self._translate_transcription(frame)

    async def _translate_transcription(self, frame: TranscriptionFrame):
        started = time.perf_counter()
        try:
            translation = (await asyncio.to_thread(self._translate, frame.text)).strip()
        except Exception as e:
            await self.push_error(f"{self}: translation failed: {e}", exception=e)
            return

        if not translation:
            logger.warning(f"{self}: empty translation for [{frame.text}]")
            return
        logger.info(
            f"Translated [{self.source_language}->{self.target_language}, "
            f"{time.perf_counter() - started:.2f}s]: {translation}"
        )

        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame(translation))
        await self.push_frame(LLMFullResponseEndFrame())
