"""Unit tests for the Smart Turn segmenter, with a scripted turn analyzer (no models needed)."""

import unittest

from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, BaseTurnParams, EndOfTurnState
from pipecat.frames.frames import (
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.tests.utils import run_test

from echo_lab.vad.smart_turn import SmartTurnSegmenter

Started = VADUserStartedSpeakingFrame
Stopped = VADUserStoppedSpeakingFrame
Audio = InputAudioRawFrame

COMPLETE = EndOfTurnState.COMPLETE
INCOMPLETE = EndOfTurnState.INCOMPLETE


def audio() -> InputAudioRawFrame:
    return InputAudioRawFrame(audio=bytes(640), sample_rate=16000, num_channels=1)


class ScriptedAnalyzer(BaseTurnAnalyzer):
    """Answers each pause with the next scripted verdict.

    Reports the pause as too long after ``timeout_frames`` frames of non-speech.
    """

    def __init__(self, verdicts, timeout_frames=None):
        super().__init__()
        self._verdicts = list(verdicts)
        self._timeout_frames = timeout_frames
        self._silent_frames = 0
        self.cleared = 0

    @property
    def speech_triggered(self) -> bool:
        return True

    @property
    def params(self) -> BaseTurnParams:
        return BaseTurnParams()

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        self._silent_frames = 0 if is_speech else self._silent_frames + 1
        if self._timeout_frames and self._silent_frames >= self._timeout_frames:
            return COMPLETE
        return INCOMPLETE

    async def analyze_end_of_turn(self):
        return self._verdicts.pop(0), None

    def clear(self):
        self.cleared += 1


class SmartTurnSegmenterTest(unittest.IsolatedAsyncioTestCase):
    async def test_complete_sentence_ends_the_segment_at_the_pause(self):
        await run_test(
            SmartTurnSegmenter(turn_analyzer=ScriptedAnalyzer([COMPLETE])),
            frames_to_send=[Started(), audio(), Stopped(), audio()],
            expected_down_frames=[Started, Audio, Stopped, Audio],
        )

    async def test_pause_mid_sentence_keeps_one_segment(self):
        await run_test(
            SmartTurnSegmenter(turn_analyzer=ScriptedAnalyzer([INCOMPLETE, COMPLETE])),
            frames_to_send=[Started(), audio(), Stopped(), audio(), Started(), audio(), Stopped()],
            expected_down_frames=[Started, Audio, Audio, Audio, Stopped],
        )

    async def test_long_pause_ends_the_segment_anyway(self):
        analyzer = ScriptedAnalyzer([INCOMPLETE], timeout_frames=3)
        await run_test(
            SmartTurnSegmenter(turn_analyzer=analyzer),
            frames_to_send=[Started(), audio(), Stopped(), audio(), audio(), audio(), audio()],
            expected_down_frames=[Started, Audio, Audio, Audio, Audio, Stopped, Audio],
        )
        self.assertEqual(analyzer.cleared, 1)

    async def test_open_segment_is_closed_on_end(self):
        await run_test(
            SmartTurnSegmenter(turn_analyzer=ScriptedAnalyzer([INCOMPLETE])),
            frames_to_send=[Started(), audio(), Stopped()],
            expected_down_frames=[Started, Audio, Stopped],
        )

    async def test_silence_before_speech_opens_nothing(self):
        await run_test(
            SmartTurnSegmenter(turn_analyzer=ScriptedAnalyzer([], timeout_frames=1)),
            frames_to_send=[audio(), audio()],
            expected_down_frames=[Audio, Audio],
        )


if __name__ == "__main__":
    unittest.main()
