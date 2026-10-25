"""Unit tests for the processors in pipeline.py (no models or audio devices needed)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InputAudioRawFrame,
    TTSAudioRawFrame,
)
from pipecat.tests.utils import SleepFrame, run_test

from echo_lab.pipeline import InputMuteWhileSpeaking, create_output_recorder


def audio() -> InputAudioRawFrame:
    return InputAudioRawFrame(audio=b"\x01\x02" * 160, sample_rate=16000, num_channels=1)


class InputMuteWhileSpeakingTest(unittest.IsolatedAsyncioTestCase):
    async def test_silences_input_only_while_bot_speaks(self):
        down, _ = await run_test(
            InputMuteWhileSpeaking(),
            frames_to_send=[
                audio(),
                BotStartedSpeakingFrame(),
                audio(),
                BotStoppedSpeakingFrame(),
                audio(),
            ],
        )
        chunks = [f.audio for f in down if isinstance(f, InputAudioRawFrame)]
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], audio().audio)
        self.assertEqual(chunks[1], bytes(len(audio().audio)))
        self.assertEqual(chunks[2], audio().audio)


def tone_onsets(samples: np.ndarray, rate: int) -> list[float]:
    """Start times in seconds of the loud stretches in ``samples``."""
    loud = np.flatnonzero(np.abs(samples) > 1000)
    starts = [loud[0]] + [b for a, b in zip(loud, loud[1:]) if b - a > rate // 10]
    return [s / rate for s in starts]


class OutputRecorderTest(unittest.IsolatedAsyncioTestCase):
    async def test_translations_keep_their_real_spacing(self):
        # 20 ms of microphone silence, and a 0.5 s tone as the translation.
        mic_frame = InputAudioRawFrame(audio=bytes(640), sample_rate=16000, num_channels=1)
        tone = (np.sin(np.arange(12000) / 24000 * 2 * np.pi * 440) * 8000).astype(np.int16)

        def translation():
            return TTSAudioRawFrame(audio=tone.tobytes(), sample_rate=24000, num_channels=1)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.wav"
            await run_test(
                create_output_recorder(path),
                frames_to_send=[
                    *[mic_frame] * 50,  # 1 s of microphone audio
                    SleepFrame(sleep=0.05),
                    translation(),
                    SleepFrame(sleep=0.05),
                    *[mic_frame] * 100,  # 2 more seconds
                    SleepFrame(sleep=0.05),
                    translation(),
                    SleepFrame(sleep=0.05),
                    *[mic_frame] * 50,
                    SleepFrame(sleep=0.05),
                ],
            )
            samples, rate = sf.read(str(path), dtype="int16")

        first, second = tone_onsets(samples, rate)
        # The recorder's resampler for the microphone track holds back ~0.1 s,
        # so everything sits that much early; the offset is constant. (Pauses in
        # the frames stay under 0.2 s: after that, the resampler drops what it
        # holds, which a live microphone never gives it the chance to do.)
        self.assertAlmostEqual(first, 1.0, delta=0.15)
        # What matters: the translations are as far apart as they were spoken.
        self.assertAlmostEqual(second - first, 2.0, delta=0.03)


if __name__ == "__main__":
    unittest.main()
