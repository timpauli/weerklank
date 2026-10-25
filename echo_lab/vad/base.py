"""Base class for the VAD stage's segment markers.

Segments are measured in audio time (the samples that went by), not wall-clock
time.
"""

from pipecat.frames.frames import (
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessor


class SegmentMarker(FrameProcessor):
    """Opens and closes speech segments by pushing VAD frames downstream."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._segment_open = False
        self._segment_secs = 0.0

    async def _open_segment(self, secs: float = 0.0):
        """Open a segment that already holds ``secs`` of audio (pushed right after)."""
        self._segment_open = True
        self._segment_secs = secs
        await self.push_frame(VADUserStartedSpeakingFrame())

    async def _close_segment(self):
        self._segment_open = False
        # stop_secs=0: no silence preceded this stop, so the STT doesn't
        # subtract any from its latency measurement.
        await self.push_frame(VADUserStoppedSpeakingFrame(stop_secs=0.0))

    def _add_audio(self, frame: InputAudioRawFrame) -> float:
        """Count a frame's duration into the open segment and return the segment length."""
        self._segment_secs += frame.num_frames / frame.sample_rate
        return self._segment_secs
