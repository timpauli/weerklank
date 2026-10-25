"""The exchangeable backends of each pipeline stage.

Each stage section in the config names its backend::

    tts:
      backend: omnivoice
      omnivoice: {...}       # options for that backend

A backend is a factory function ``create(ctx: StageContext)`` that returns the
stage's frame processor (or a list of processors). Backends are imported only
when selected, so an unused backend's heavy dependencies are never loaded.
To add a backend, write such a factory and list it in ``BACKENDS``.
"""

import importlib
from dataclasses import dataclass
from typing import Any

from pipecat.processors.frame_processor import FrameProcessor

from echo_lab.config import ConfigError, get_value

STAGES = ("vad", "stt", "llm", "tts")

BACKENDS: dict[str, dict[str, str]] = {
    "vad": {
        "silero_smart_turn": "echo_lab.vad.smart_turn",
        "fixed_interval": "echo_lab.vad.fixed_interval",
    },
    "stt": {
        "parakeet_tdt": "echo_lab.stt.parakeet_tdt",
    },
    "llm": {
        "m2m100_ct2": "echo_lab.llm.m2m100_ct2",
        "translategemma_ct2": "echo_lab.llm.translategemma_ct2",
    },
    "tts": {
        "omnivoice": "echo_lab.tts.omnivoice",
    },
}


@dataclass
class StageContext:
    """What a backend factory gets to build its processor(s).

    Parameters:
        input_language: Language the speaker uses (``llm.input_language``).
        output_language: Language to translate into (``llm.output_language``).
        stage: The stage's config section, e.g. ``config["tts"]``. Holds the
            parameters every backend of the stage shares (``ref_audio``, ...).
        options: The selected backend's own section, e.g. ``config["tts"]["omnivoice"]``.
    """

    input_language: str
    output_language: str
    stage: dict[str, Any]
    options: dict[str, Any]


def create_stage(config: dict[str, Any], stage: str) -> list[FrameProcessor]:
    """Build the processor(s) for one stage from the config."""
    section = config.get(stage) or {}
    backend = section["backend"]
    module_name = BACKENDS[stage].get(backend)
    if module_name is None:
        raise ConfigError(
            f"Unknown {stage} backend '{backend}'. Available: {', '.join(BACKENDS[stage])}"
        )

    ctx = StageContext(
        input_language=get_value(config, "llm.input_language"),
        output_language=get_value(config, "llm.output_language"),
        stage=section,
        options=dict(section.get(backend) or {}),
    )
    processors = importlib.import_module(module_name).create(ctx)
    return list(processors) if isinstance(processors, (list, tuple)) else [processors]
