"""Unit tests for runtime control: value parsers, the OSC server and the TTS applying changes."""

import asyncio
import unittest

import numpy as np
from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.tests.utils import run_test
from pythonosc.udp_client import SimpleUDPClient

from echo_lab.control import Control, OSCServer, flag, number, optional
from echo_lab.tts.omnivoice import OmniVoiceSettings, OmniVoiceTTSService


class FakeModel:
    """Stands in for OmniVoice: remembers the arguments of each generate() call."""

    sampling_rate = 24000

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [np.zeros(2400, dtype=np.float32)]


class FakeOmniVoice(OmniVoiceTTSService):
    """OmniVoiceTTSService with the fake model instead of loading the real one."""

    def _load(self, model, dtype, ref_audio, ref_text):
        self._model = FakeModel()
        self._voice_clone_prompt = None


def sentence(text: str) -> list:
    return [LLMFullResponseStartFrame(), LLMTextFrame(text), LLMFullResponseEndFrame()]


class ParserTest(unittest.TestCase):
    def test_number(self):
        self.assertEqual(number()(3), 3.0)
        self.assertEqual(number(above=0)(0.5), 0.5)
        self.assertEqual(number(at_least=0)(0), 0.0)
        for parse, value in [(number(), "fast"), (number(), True), (number(above=0), 0)]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse(value)

    def test_flag_takes_bools_and_numbers(self):
        self.assertEqual(
            [flag(True), flag(False), flag(1), flag(0), flag(0.0)],
            [True, False, True, False, False],
        )
        with self.assertRaises(ValueError):
            flag("yes")

    def test_optional_takes_nil(self):
        self.assertIsNone(optional(number())(None))
        self.assertEqual(optional(number())(2), 2.0)
        with self.assertRaises(ValueError):
            number()(None)

    def test_osc_float_precision_is_kept_not_invented(self):
        self.assertEqual(number()(1.2000000476837158), 1.2)


class FakeWorker:
    def __init__(self):
        self.frames = []

    async def queue_frame(self, frame):
        self.frames.append(frame)


class Settable:
    """A processor with two controls, ``/tts/level`` and the optional ``/tts/default``."""

    def controls(self):
        return [
            Control("level", number(at_least=0), lambda value: ("level", value)),
            Control("default", optional(number()), lambda value: ("default", value)),
        ]


class OSCServerTest(unittest.IsolatedAsyncioTestCase):
    async def send(self, *messages):
        async with OSCServer("127.0.0.1", 0) as osc:
            osc.add("tts", [Settable(), object()])  # processors without controls are skipped
            worker = FakeWorker()
            osc.attach(worker)
            client = SimpleUDPClient("127.0.0.1", osc.port)
            for address, value in messages:
                client.send_message(address, value)
            await asyncio.sleep(0.2)
        return worker.frames

    async def test_message_becomes_a_frame(self):
        self.assertEqual(await self.send(("/tts/level", 0.5)), [("level", 0.5)])

    async def test_message_without_value_reaches_an_optional_control_as_none(self):
        self.assertEqual(await self.send(("/tts/default", None)), [("default", None)])
        self.assertEqual(await self.send(("/tts/level", None)), [])  # required: rejected

    async def test_bad_values_and_unknown_addresses_are_ignored(self):
        frames = await self.send(("/tts/level", -1), ("/tts/level", "high"), ("/tts/nope", 1))
        self.assertEqual(frames, [])

    async def test_port_none_is_off(self):
        async with OSCServer("127.0.0.1", None) as osc:
            osc.add("tts", [Settable()])
            osc.attach(FakeWorker())
            self.assertIsNone(osc._transport)


class OmniVoiceControlTest(unittest.IsolatedAsyncioTestCase):
    async def test_change_applies_from_the_next_sentence_on(self):
        tts = FakeOmniVoice(language="nl", speed=None, guidance_scale=2.0)
        controls = {c.name: c for c in tts.controls()}
        self.assertEqual(set(controls), {"speed", "guidance_scale", "t_shift", "denoise"})

        update = controls["speed"].frame(controls["speed"].parse(1.3))
        self.assertIsInstance(update, TTSUpdateSettingsFrame)
        self.assertIs(update.service, tts)

        await run_test(
            tts,
            frames_to_send=[
                *sentence("Eerste zin."),
                update,
                controls["guidance_scale"].frame(3),
                controls["denoise"].frame(False),
                *sentence("Tweede zin."),
            ],
        )
        first, second = tts._model.calls
        self.assertNotIn("speed", first)
        self.assertEqual((first["guidance_scale"], first["denoise"]), (2.0, True))
        self.assertEqual(second["speed"], 1.3)
        self.assertEqual((second["guidance_scale"], second["denoise"]), (3, False))

    def test_settings_are_complete(self):
        tts = FakeOmniVoice(language="pt")
        self.assertIsInstance(tts.settings, OmniVoiceSettings)
        tts.settings.validate_complete()  # every field set


if __name__ == "__main__":
    unittest.main()
