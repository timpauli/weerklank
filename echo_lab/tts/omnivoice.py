"""k2-fsa OmniVoice TTS: zero-shot voice cloning in 600+ languages."""

from typing import Any

import numpy as np
import torch
from loguru import logger

from echo_lab.registry import StageContext
from echo_lab.tts.base import LocalTTSService

# A short sentence in the output language for the warm-up; other languages get "Hello."
WARMUP_TEXT = {"nl": "Hallo.", "pt": "Olá.", "de": "Hallo."}


class OmniVoiceTTSService(LocalTTSService):
    """Speaks with OmniVoice, cloning the voice of a reference recording."""

    def __init__(
        self,
        *,
        language: str,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        model: str = "k2-fsa/OmniVoice",
        device: str = "cuda:0",
        dtype: str = "float16",
        num_step: int = 32,
        speed: float | None = None,
        instruct: str | None = None,
        generate_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Load the model and prepare the cloned voice.

        Args:
            language: Language to speak, e.g. ``"pt"``.
            ref_audio: Recording of the voice to clone. Without it OmniVoice picks
                a voice itself (or designs one from ``instruct``).
            ref_text: Transcript of ``ref_audio``. If missing, OmniVoice
                transcribes the recording with its own ASR model (a large extra
                download, and slower to start).
            model: Hugging Face repo or local path of the OmniVoice checkpoint.
            device: Torch device, e.g. ``"cuda:0"``.
            dtype: ``"float16"``, ``"bfloat16"`` or ``"float32"``.
            num_step: Diffusion decoding steps; fewer is faster, more is cleaner.
            speed: Speaking rate factor (>1 faster); None lets the model decide.
            instruct: Voice description used when there is no ``ref_audio``,
                e.g. ``"female, low pitch"``.
            generate_kwargs: Extra arguments for ``OmniVoice.generate`` (e.g.
                ``guidance_scale``, ``t_shift``).
            **kwargs: Additional arguments passed to TTSService.
        """
        super().__init__(language=language, model=model, device=device, **kwargs)
        self._num_step = num_step
        self._speed = speed
        self._instruct = instruct
        self._generate_kwargs = dict(generate_kwargs or {})
        self._load(model, getattr(torch, dtype), ref_audio, ref_text)

    def _load(self, model: str, dtype: torch.dtype, ref_audio: str | None, ref_text: str | None):
        from omnivoice import OmniVoice

        logger.info(f"Loading OmniVoice '{model}' on {self._device}")
        self._model = OmniVoice.from_pretrained(model, device_map=self._device, dtype=dtype)

        self._voice_clone_prompt = None
        if ref_audio:
            if not ref_text:
                logger.warning(
                    "No tts.ref_text given: OmniVoice will transcribe the reference audio itself"
                )
            self._voice_clone_prompt = self._model.create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=ref_text or None
            )
            logger.info(f"OmniVoice: cloning the voice from {ref_audio}")
        elif self._instruct:
            logger.info(f"OmniVoice: designing a voice from '{self._instruct}'")
        else:
            logger.info("OmniVoice: no reference audio, the model picks a voice")

        self._synthesize(WARMUP_TEXT.get(self.language, "Hello."))
        # Loading peaks at about 6 GB, speaking at about 2.5 GB. Give the rest back,
        # so other bots on the same GPU can use it.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _synthesize(self, text: str) -> tuple[np.ndarray, int]:
        kwargs: dict[str, Any] = {
            "text": text,
            "language": self.language,
            "num_step": self._num_step,
            **self._generate_kwargs,
        }
        if self._speed is not None:
            kwargs["speed"] = self._speed
        if self._voice_clone_prompt is not None:
            kwargs["voice_clone_prompt"] = self._voice_clone_prompt
        elif self._instruct:
            kwargs["instruct"] = self._instruct
        audios = self._model.generate(**kwargs)
        return audios[0], int(self._model.sampling_rate)


def create(ctx: StageContext):
    """Build the OmniVoice service.

    Shared options (``tts``): ref_audio, ref_text.
    Options (``tts.omnivoice``): model, device, dtype, num_step, speed, instruct, generate_kwargs.
    """
    return OmniVoiceTTSService(
        language=ctx.output_language,
        ref_audio=ctx.stage.get("ref_audio"),
        ref_text=ctx.stage.get("ref_text"),
        **ctx.options,
    )
