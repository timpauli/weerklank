"""Fixed-interval backend: ignores speech pauses and cuts the audio about every N seconds."""

import numpy as np
from pipecat.frames.frames import EndFrame, Frame, InputAudioRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameDirection

from echo_lab.registry import StageContext
from echo_lab.vad.base import SegmentMarker


def _frame_rms(frame: InputAudioRawFrame) -> float:
    samples = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0


class FixedIntervalSegmenter(SegmentMarker):
    """Cuts the input into segments of about a fixed length, ignoring the VAD.

    A segment is always open: it starts with the pipeline and is closed and
    reopened about every ``interval_secs`` of audio. Rather than cut exactly
    there, possibly through a word, the cut goes at the quietest point of a
    ``search_secs`` window centred on that time. The audio of the window is held
    back until the window is complete, which delays each cut by up to
    ``search_secs / 2``. The last, shorter segment is closed when the pipeline
    ends gracefully.
    """

    # Loudness is averaged over this long, so a single quiet frame inside a
    # word doesn't count as a gap between words.
    SMOOTHING_SECS = 0.1

    def __init__(self, *, interval_secs: float, search_secs: float = 0.5, **kwargs):
        """Initialize the segmenter.

        Args:
            interval_secs: Target length of each segment in seconds.
            search_secs: Width of the window around the target in which the cut
                goes at the quietest point; 0 cuts exactly at ``interval_secs``.
            **kwargs: Additional arguments passed to FrameProcessor.
        """
        super().__init__(**kwargs)
        if interval_secs <= 0:
            raise ValueError(f"interval_secs must be positive, got {interval_secs}")
        if not 0 <= search_secs <= interval_secs / 2:
            raise ValueError(
                f"search_secs must be between 0 and interval_secs / 2, got {search_secs}"
            )
        self._interval_secs = interval_secs
        self._window_start = interval_secs - search_secs / 2
        self._window_end = interval_secs + search_secs / 2
        # Audio of the current search window, not pushed yet, and the segment
        # time where it starts.
        self._held: list[InputAudioRawFrame] = []
        self._held_start = 0.0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame) and self._segment_open:
            await self._handle_audio(frame)
            return

        if isinstance(frame, EndFrame) and self._segment_open:
            await self._push_held(self._held)
            self._held = []
            await self._close_segment()

        await self.push_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._open_segment()

    async def _handle_audio(self, frame: InputAudioRawFrame):
        start = self._segment_secs
        end = self._add_audio(frame)
        if not self._held:
            if end < self._window_start:
                await self.push_frame(frame)
                return
            self._held_start = start
        self._held.append(frame)
        if end >= self._window_end:
            await self._cut()

    async def _cut(self):
        """Cut at the quietest point of the held window and start the next segment."""
        held, self._held = self._held, []
        durations = np.array([f.num_frames / f.sample_rate for f in held])
        # Segment time at the end of each held frame, i.e. at each possible cut.
        ends = self._held_start + np.cumsum(durations)

        loudness = np.array([_frame_rms(f) for f in held])
        half = max(0, round(self.SMOOTHING_SECS / 2 / max(durations.mean(), 1e-3)))
        if half:
            kernel = np.ones(2 * half + 1) / (2 * half + 1)
            loudness = np.convolve(np.pad(loudness, half, mode="edge"), kernel, mode="valid")
        # Quietest frame; among equally quiet ones, the one closest to the target.
        i = int(np.lexsort((np.abs(ends - self._interval_secs), loudness))[0])

        await self._push_held(held[: i + 1])
        await self._close_segment()
        # The rest of the window starts the new segment (it's shorter than
        # search_secs, so it can't reach the next window).
        await self._open_segment(secs=float(durations[i + 1 :].sum()))
        await self._push_held(held[i + 1 :])

    async def _push_held(self, frames: list[InputAudioRawFrame]):
        for frame in frames:
            await self.push_frame(frame)


def create(ctx: StageContext):
    """Build the fixed-interval segmenter.

    Options (``vad.fixed_interval``):
        interval_secs: Target length of each segment in seconds.
        search_secs: Window around the target in which the cut goes at the
            quietest point (between words); 0 cuts exactly every interval_secs.
    """
    return FixedIntervalSegmenter(**ctx.options)
