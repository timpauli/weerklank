"""Silero + Smart Turn backend: a segment ends when the speaker has finished a sentence.

Silero reports short pauses; at each one, Pipecat's Smart Turn v3 model listens
to the speech so far and judges whether the sentence is complete. Complete
sentences end their segment at once, after a much shorter pause than plain
Silero needs; a pause in the middle of a sentence keeps the segment open.
"""

import time

from loguru import logger
from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup

from echo_lab.registry import StageContext
from echo_lab.vad.base import SegmentMarker


class SmartTurnSegmenter(SegmentMarker):
    """Turns the VAD's speech/pause reports into sentence-sized segments.

    Goes right after a ``VADProcessor``, whose VAD frames it replaces with its
    own: a segment starts when speech starts, and ends at a pause the turn
    analyzer judges to end a sentence, or once the pause outlasts the
    analyzer's own ``stop_secs`` (whatever it thinks of the sentence). Speech
    after a pause judged incomplete continues the same segment.
    """

    def __init__(self, *, turn_analyzer: BaseTurnAnalyzer, **kwargs):
        """Initialize the segmenter.

        Args:
            turn_analyzer: Decides whether a pause ends the sentence, e.g.
                ``LocalSmartTurnAnalyzerV3``.
            **kwargs: Additional arguments passed to FrameProcessor.
        """
        super().__init__(**kwargs)
        self._turn_analyzer = turn_analyzer
        self._vad_speaking = False

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        self._turn_analyzer.set_sample_rate(setup.audio_in_sample_rate)

    async def cleanup(self):
        await super().cleanup()
        await self._turn_analyzer.cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, VADUserStartedSpeakingFrame):
                await self._handle_speech_started(frame)
                return
            if isinstance(frame, VADUserStoppedSpeakingFrame):
                await self._handle_speech_stopped()
                return
            if isinstance(frame, EndFrame) and self._segment_open:
                await self._close_without_verdict()

        await self.push_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            state = self._turn_analyzer.append_audio(frame.audio, self._vad_speaking)
            # COMPLETE here means the pause outlasted the analyzer's stop_secs.
            if state == EndOfTurnState.COMPLETE and self._segment_open:
                logger.info("Smart Turn: pause reached max_pause_secs, ending the segment")
                await self._close_without_verdict()

    async def _handle_speech_started(self, frame: VADUserStartedSpeakingFrame):
        self._vad_speaking = True
        self._turn_analyzer.update_vad_start_secs(frame.start_secs)
        if not self._segment_open:
            await self._open_segment()

    async def _handle_speech_stopped(self):
        self._vad_speaking = False
        if not self._segment_open:
            return
        started = time.perf_counter()
        state, metrics = await self._turn_analyzer.analyze_end_of_turn()
        probability = getattr(metrics, "probability", None)
        verdict = "complete" if state == EndOfTurnState.COMPLETE else "incomplete, waiting"
        logger.info(
            f"Smart Turn: {verdict}"
            + (f" (p={probability:.2f})" if probability is not None else "")
            + f" in {(time.perf_counter() - started) * 1000:.0f} ms"
        )
        if state == EndOfTurnState.COMPLETE:
            await self._close_segment()

    async def _close_without_verdict(self):
        """Close the segment without asking the analyzer, and reset the analyzer."""
        self._turn_analyzer.clear()
        await self._close_segment()


def create(ctx: StageContext):
    """Build Silero and the Smart Turn segmenter.

    Options (``vad.silero_smart_turn``):
        stop_secs: Pause length in seconds after which Smart Turn is asked
            whether the sentence is complete.
        start_secs, confidence, min_volume: Silero's ``VADParams``.
        max_pause_secs: End the segment after this much silence even if Smart
            Turn judges the sentence incomplete.
        cpu_threads: CPU threads for the Smart Turn model.
    """
    options = dict(ctx.options)
    max_pause_secs = float(options.pop("max_pause_secs", 2.0))
    cpu_threads = int(options.pop("cpu_threads", 6))
    vad_params = VADParams(**options)

    turn_analyzer = LocalSmartTurnAnalyzerV3(
        cpu_count=cpu_threads,
        params=SmartTurnParams(stop_secs=max_pause_secs),
    )
    return [
        VADProcessor(vad_analyzer=SileroVADAnalyzer(params=vad_params)),
        SmartTurnSegmenter(turn_analyzer=turn_analyzer),
    ]
