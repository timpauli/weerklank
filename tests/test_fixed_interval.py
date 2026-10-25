"""Unit tests for the fixed-interval segmenter, including its quiet-point cuts."""

import unittest

import numpy as np
from pipecat.frames.frames import (
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.tests.utils import run_test

from echo_lab.vad.fixed_interval import FixedIntervalSegmenter

Started = VADUserStartedSpeakingFrame
Stopped = VADUserStoppedSpeakingFrame
Audio = InputAudioRawFrame


def audio(secs: float, sample_rate: int = 16000, amplitude: int = 0) -> InputAudioRawFrame:
    samples = np.full(int(sample_rate * secs), amplitude, dtype=np.int16)
    return InputAudioRawFrame(audio=samples.tobytes(), sample_rate=sample_rate, num_channels=1)


def audio_before_first_stop(frames) -> int:
    """How many audio frames went into the first segment."""
    first_stop = next(i for i, f in enumerate(frames) if isinstance(f, Stopped))
    return sum(isinstance(f, Audio) for f in frames[:first_stop])


class FixedIntervalSegmenterTest(unittest.IsolatedAsyncioTestCase):
    async def test_cuts_every_interval_and_closes_last_segment_on_end(self):
        await run_test(
            FixedIntervalSegmenter(interval_secs=1.0, search_secs=0),
            frames_to_send=[audio(0.5) for _ in range(5)],
            expected_down_frames=[
                Started,
                Audio, Audio, Stopped, Started,
                Audio, Audio, Stopped, Started,
                Audio,
                Stopped,
            ],
        )  # fmt: skip

    async def test_counts_audio_time_not_frames(self):
        # 8 kHz frames of 0.25 s: four of them make one second.
        await run_test(
            FixedIntervalSegmenter(interval_secs=1.0, search_secs=0),
            frames_to_send=[audio(0.25, sample_rate=8000) for _ in range(4)],
            expected_down_frames=[Started, Audio, Audio, Audio, Audio, Stopped, Started, Stopped],
        )

    def test_rejects_invalid_lengths(self):
        with self.assertRaises(ValueError):
            FixedIntervalSegmenter(interval_secs=0)
        with self.assertRaises(ValueError):
            FixedIntervalSegmenter(interval_secs=1.0, search_secs=0.6)


class QuietCutTest(unittest.IsolatedAsyncioTestCase):
    """Interval 1 s, search window 0.75-1.25 s, frames of 0.05 s."""

    async def run_segmenter(self, frames):
        down, _ = await run_test(
            FixedIntervalSegmenter(interval_secs=1.0, search_secs=0.5), frames_to_send=frames
        )
        # No audio is lost, duplicated or reordered by holding it back.
        self.assertEqual([f.id for f in down if isinstance(f, Audio)], [f.id for f in frames])
        return down

    async def test_cuts_in_the_quietest_part_of_the_window(self):
        # Speech everywhere except a pause from 0.80 to 0.95 s (frames 16-18).
        frames = [audio(0.05, amplitude=0 if i in (16, 17, 18) else 3000) for i in range(30)]
        down = await self.run_segmenter(frames)
        # The cut goes after frame 17, the middle of the pause.
        self.assertEqual(audio_before_first_stop(down), 18)
        self.assertEqual(sum(isinstance(f, Stopped) for f in down), 2)  # the cut and the end

    async def test_ignores_quiet_points_before_the_window(self):
        frames = [audio(0.05, amplitude=0 if i in (5, 6, 7) else 3000) for i in range(30)]
        down = await self.run_segmenter(frames)
        self.assertGreaterEqual(audio_before_first_stop(down), 15)

    async def test_without_a_pause_cuts_at_the_target(self):
        frames = [audio(0.05, amplitude=3000) for _ in range(30)]
        down = await self.run_segmenter(frames)
        self.assertEqual(audio_before_first_stop(down), 20)  # 20 x 0.05 s = 1.0 s

    async def test_end_flushes_held_audio_before_closing(self):
        frames = [audio(0.05, amplitude=3000) for _ in range(20)]  # ends inside the window
        down = await self.run_segmenter(frames)
        self.assertIsInstance(down[-1], Stopped)
        self.assertEqual(audio_before_first_stop(down), 20)


if __name__ == "__main__":
    unittest.main()
