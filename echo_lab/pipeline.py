"""Assembling and running the translation pipeline."""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from loguru import logger
from pipecat.frames.frames import Frame, InputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_mute.always_user_mute_strategy import AlwaysUserMuteStrategy
from pipecat.workers.runner import WorkerRunner

from echo_lab.audio_devices import describe_device
from echo_lab.config import get_value
from echo_lab.registry import STAGES, create_stage

# Silero VAD and Parakeet both work on 16 kHz audio; OmniVoice speaks at 24 kHz.
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000


class InputMuteWhileSpeaking(FrameProcessor):
    """Replaces input audio with silence while the translation is playing.

    Stops the microphone from picking up the speakers and translating the
    translation again. Goes right after the input transport, so the echo never
    reaches the VAD; the output transport's ``BotStartedSpeakingFrame`` /
    ``BotStoppedSpeakingFrame`` travel upstream to it.

    When to mute is decided by Pipecat's ``AlwaysUserMuteStrategy`` (muted
    while the bot speaks). Pipecat normally runs it inside its conversation
    aggregator, which this pipeline doesn't have.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._strategy = AlwaysUserMuteStrategy()

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        await self._strategy.setup(setup)

    async def cleanup(self):
        await super().cleanup()
        await self._strategy.cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        muted = await self._strategy.process_frame(frame)
        if muted and isinstance(frame, InputAudioRawFrame):
            frame = InputAudioRawFrame(
                audio=bytes(len(frame.audio)),
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
            )

        await self.push_frame(frame, direction)


def create_output_recorder(path: str | Path) -> AudioBufferProcessor:
    """Record the translated speech to a WAV file, on the session's real timeline.

    Pipecat's recorder lines the TTS audio up with the microphone audio, so each
    translation sits in the file at the time it was spoken, with silence in
    between. It keeps the session in memory (about 350 MB per hour for both
    tracks) and the file is written once, when the pipeline ends or on Ctrl+C:
    flushing in chunks would shift utterances that run across a flush.

    Args:
        path: WAV file to write the translation track to.
    """
    path = Path(path)
    recorder = AudioBufferProcessor(auto_start_recording=True)

    @recorder.event_handler("on_track_audio_data")
    async def on_track_audio_data(
        recorder, user_audio: bytes, bot_audio: bytes, sample_rate: int, num_channels: int
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        samples = np.frombuffer(bot_audio, dtype=np.int16)
        sf.write(str(path), samples, sample_rate, subtype="PCM_16")
        logger.info(
            f"Wrote {len(samples) / sample_rate:.1f}s of the session's translation to {path}"
        )

    return recorder


def build_stages(config: dict[str, Any]) -> list[FrameProcessor]:
    """Build the VAD, STT, LLM (translation) and TTS processors, loading their models."""
    processors: list[FrameProcessor] = []
    for stage in STAGES:
        backend = get_value(config, f"{stage}.backend")
        logger.info(f"Setting up {stage}: {backend}")
        processors += create_stage(config, stage)
    return processors


async def run_pipeline(config: dict[str, Any], *, output_file: str | Path | None = None) -> None:
    """Run the pipeline until it is interrupted (Ctrl+C).

    Args:
        config: The validated configuration.
        output_file: Write the translated speech to this WAV file instead of
            the output device.
    """
    input_device = get_value(config, "audio.input_device")
    output_device = None if output_file else get_value(config, "audio.output_device")

    logger.info(f"Audio input: {describe_device(input_device, 'input')}")
    if not output_file:
        logger.info(f"Audio output: {describe_device(output_device, 'output')}")

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=not output_file,
            input_device_index=input_device,
            output_device_index=output_device,
        )
    )

    stages = build_stages(config)

    # Microphone -> (mute) -> VAD -> STT -> translation -> TTS -> speakers or WAV file.
    processors: list[FrameProcessor] = [transport.input()]
    if not output_file and get_value(config, "audio.mute_input_while_speaking", False):
        processors.append(InputMuteWhileSpeaking())
    processors += stages
    processors.append(create_output_recorder(output_file) if output_file else transport.output())

    worker = PipelineWorker(
        Pipeline(processors),
        params=PipelineParams(
            audio_in_sample_rate=INPUT_SAMPLE_RATE,
            audio_out_sample_rate=OUTPUT_SAMPLE_RATE,
        ),
        # An interpreter waits for speech as long as it takes.
        idle_timeout_secs=None,
        enable_rtvi=False,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    @worker.event_handler("on_pipeline_started")
    async def on_pipeline_started(worker, frame):
        logger.info(
            f"Translating {get_value(config, 'llm.input_language')} -> "
            f"{get_value(config, 'llm.output_language')}. Press Ctrl+C to stop."
        )

    # Pipecat's SIGINT handling needs loop.add_signal_handler, which Windows
    # lacks; there asyncio.run() turns Ctrl+C into a cancellation instead.
    runner = WorkerRunner(handle_sigint=sys.platform != "win32")
    await runner.add_workers(worker)
    await runner.run()
