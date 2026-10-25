"""Base class for local TTS models that synthesize a sentence in one call."""

import asyncio
import time
from abc import abstractmethod
from collections.abc import AsyncGenerator

import numpy as np
from loguru import logger
from pipecat.audio.utils import create_file_resampler
from pipecat.frames.frames import Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService


class LocalTTSService(TTSService):
    """A TTS service around a local model with a blocking :meth:`_synthesize` call.

    The base ``TTSService`` splits incoming text into sentences; each sentence
    is synthesized in a worker thread (so audio keeps flowing meanwhile) and
    pushed as one audio frame, resampled to the pipeline's output rate. While
    one sentence plays, the next one is being synthesized.

    Subclasses load their model in ``__init__`` and implement :meth:`_synthesize`.
    """

    def __init__(
        self,
        *,
        language: str,
        model: str,
        device: str,
        stop_frame_timeout_s: float = 60.0,
        **kwargs,
    ):
        """Initialize the service.

        Args:
            language: Language to speak, e.g. ``"pt"``.
            model: Model name, reported in logs.
            device: Torch device the model runs on, e.g. ``"cuda:0"``.
            stop_frame_timeout_s: How long the base class waits for a sentence's
                audio before it considers the utterance over. Pipecat's default
                (3 s) suits streaming cloud TTS; a local model can take longer
                than that to synthesize one sentence, so this is the upper bound
                on synthesis time.
            **kwargs: Additional arguments passed to TTSService.
        """
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            stop_frame_timeout_s=stop_frame_timeout_s,
            settings=TTSSettings(model=model, voice=None, language=language),
            **kwargs,
        )
        self._device = device
        self._resampler = create_file_resampler()

    @property
    def language(self) -> str:
        """The language being spoken, as configured, e.g. ``"nl"``."""
        return str(self._settings.language)

    @abstractmethod
    def _synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize ``text`` (runs in a worker thread).

        Returns:
            Mono float samples in [-1, 1] and their sample rate.
        """

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        started = time.perf_counter()
        try:
            samples, sample_rate = await asyncio.to_thread(self._synthesize, text)
        except Exception as e:
            await self.push_error(f"{self}: synthesis failed: {e}", exception=e)
            return

        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            logger.warning(f"{self}: no audio for [{text}]")
            return
        logger.info(
            f"Speaking [{samples.size / sample_rate:.1f}s audio, "
            f"TTS {time.perf_counter() - started:.2f}s]: {text}"
        )

        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        audio = await self._resampler.resample(pcm, sample_rate, self.sample_rate)
        yield TTSAudioRawFrame(
            audio=audio,
            sample_rate=self.sample_rate,
            num_channels=1,
            context_id=context_id,
        )
