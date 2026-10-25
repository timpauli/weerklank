"""Unit tests for the translation base class, with a stand-in translator (no models needed)."""

import unittest

from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.tests.utils import run_test

from echo_lab.llm.base import TranslationService


class Upper(TranslationService):
    """Translates by upper-casing, and remembers what it was asked to translate."""

    def __init__(self, **kwargs):
        super().__init__(model="upper", **kwargs)
        self.calls = []

    def _translate(self, text):
        self.calls.append(text)
        return text.upper()


def transcript(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text, "", "2026-10-25T00:00:00Z")


class TranslationServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_sends_the_translation_like_an_llm_reply(self):
        down, _ = await run_test(
            Upper(source_language="de", target_language="nl"),
            frames_to_send=[transcript("guten morgen")],
            expected_down_frames=[
                TranscriptionFrame,
                LLMFullResponseStartFrame,
                LLMTextFrame,
                LLMFullResponseEndFrame,
            ],
        )
        self.assertEqual(down[2].text, "GUTEN MORGEN")

    async def test_each_transcript_is_translated_in_order(self):
        service = Upper(source_language="pt", target_language="nl")
        await run_test(service, frames_to_send=[transcript("bom dia"), transcript("obrigado")])
        self.assertEqual(service.calls, ["bom dia", "obrigado"])

    async def test_empty_transcripts_are_not_translated(self):
        service = Upper(source_language="de", target_language="nl")
        await run_test(
            service, frames_to_send=[transcript("  ")], expected_down_frames=[TranscriptionFrame]
        )
        self.assertEqual(service.calls, [])

    def test_settings_hold_the_model(self):
        service = Upper(source_language="pt", target_language="nl")
        self.assertEqual(service.settings.model, "upper")
        service.settings.validate_complete()  # every field set


if __name__ == "__main__":
    unittest.main()
