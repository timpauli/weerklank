"""k2-fsa OmniVoice TTS: zero-shot voice cloning in 600+ languages."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from loguru import logger
from pipecat.frames.frames import TTSUpdateSettingsFrame
from pipecat.services.settings import TTSSettings
from pipecat.utils.types import NOT_GIVEN, NotGiven

from echo_lab.control import Control, flag, number, optional, settings_controls
from echo_lab.registry import StageContext
from echo_lab.tts.base import LocalTTSService

# A short sentence in the output language for the warm-up; other languages get "Hello."
WARMUP_TEXT = {"nl": "Hallo.", "pt": "Olá.", "de": "Hallo."}


@dataclass
class OmniVoiceSettings(TTSSettings):
    """OmniVoice's settings. The generation parameters can change while running.

    Parameters:
        speed: Speaking rate factor (>1 faster); None lets the model decide.
        guidance_scale: How strongly the speech follows the text and the cloned
            voice (classifier-free guidance); 0 turns guidance off.
        t_shift: Shifts the diffusion time steps; smaller values spend more of
            the steps on the noisy start.
        denoise: Asks for clean speech even if the reference recording is noisy
            (only used when cloning a voice).
    """

    speed: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    guidance_scale: float | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    t_shift: float | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    denoise: bool | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class OmniVoiceTTSService(LocalTTSService):
    """Speaks with OmniVoice, cloning the voice of a reference recording."""

    _settings: OmniVoiceSettings

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
        guidance_scale: float = 2.0,
        t_shift: float = 0.1,
        denoise: bool = True,
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
            speed, guidance_scale, t_shift, denoise: Generation parameters, see
                :class:`OmniVoiceSettings`; changeable while running.
            instruct: Voice description used when there is no ``ref_audio``,
                e.g. ``"female, low pitch"``.
            generate_kwargs: Further arguments for ``OmniVoice.generate``.
            **kwargs: Additional arguments passed to TTSService.
        """
        super().__init__(
            settings=OmniVoiceSettings(
                model=model,
                voice=None,
                language=language,
                speed=speed,
                guidance_scale=guidance_scale,
                t_shift=t_shift,
                denoise=denoise,
            ),
            device=device,
            **kwargs,
        )
        self._num_step = num_step
        self._instruct = instruct
        self._generate_kwargs = dict(generate_kwargs or {})
        self._load(model, getattr(torch, dtype), ref_audio, ref_text)

    def controls(self) -> list[Control]:
        """The parameters that OSC can change while running (``/tts/<name>``)."""
        return settings_controls(
            self,
            TTSUpdateSettingsFrame,
            {
                "speed": optional(number(above=0)),
                "guidance_scale": number(at_least=0),
                "t_shift": number(above=0),
                "denoise": flag,
            },
        )

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
        settings = self._settings
        kwargs: dict[str, Any] = {
            "text": text,
            "language": self.language,
            "num_step": self._num_step,
            **self._generate_kwargs,
            "guidance_scale": settings.guidance_scale,
            "t_shift": settings.t_shift,
            "denoise": settings.denoise,
        }
        if settings.speed is not None:
            kwargs["speed"] = settings.speed
        if self._voice_clone_prompt is not None:
            kwargs["voice_clone_prompt"] = self._voice_clone_prompt
        elif self._instruct:
            kwargs["instruct"] = self._instruct
        audios = self._model.generate(**kwargs)
        return audios[0], int(self._model.sampling_rate)


def create(ctx: StageContext):
    """Build the OmniVoice service.

    Shared options (``tts``): ref_audio, ref_text.
    Options (``tts.omnivoice``): model, device, dtype, num_step, speed, guidance_scale,
    t_shift, denoise, instruct, generate_kwargs.
    """
    return OmniVoiceTTSService(
        language=ctx.output_language,
        ref_audio=ctx.stage.get("ref_audio"),
        ref_text=ctx.stage.get("ref_text"),
        **ctx.options,
    )
