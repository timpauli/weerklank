"""Unit tests for the command line: overrides, GPU selection and the log format."""

import io
import os
import unittest
from unittest import mock

from loguru import logger

from echo_lab.cli import build_config, log_format, parse_args, use_only_gpu
from echo_lab.config import get_value


class SetOptionTest(unittest.TestCase):
    def test_set_overrides_config(self):
        args = parse_args(["--set", "llm.input_language=pt", "--set", "audio.input_device=4"])
        config = build_config(args)
        self.assertEqual(get_value(config, "llm.input_language"), "pt")
        self.assertEqual(get_value(config, "audio.input_device"), 4)


@mock.patch.dict(os.environ, {}, clear=False)
class UseOnlyGpuTest(unittest.TestCase):
    def test_hides_the_other_gpus(self):
        use_only_gpu(0)
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(os.environ["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")

    def test_none_leaves_gpus_alone(self):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        use_only_gpu(None)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)


class LogFormatTest(unittest.TestCase):
    def log(self, level: str, message: str, colorize: bool) -> str:
        out = io.StringIO()
        sink = logger.add(out, format=log_format, colorize=colorize)
        try:
            logger.log(level, message)
        finally:
            logger.remove(sink)
        return out.getvalue()

    def test_info_lines_have_no_level_label(self):
        self.assertNotIn("INFO", self.log("INFO", "Loading model", colorize=False))
        self.assertIn("WARNING", self.log("WARNING", "Careful", colorize=False))

    def test_pipeline_lines_are_colored_by_their_start(self):
        for message, ansi in [
            ("Heard [1.0s audio, STT 0.1s]: hallo", "\x1b[31m"),
            ("Translated [de->nl, 0.1s]: hallo", "\x1b[32m"),
            ("Speaking [1.0s audio, TTS 0.5s]: hallo", "\x1b[33m"),
        ]:
            self.assertIn(ansi + message, self.log("INFO", message, colorize=True))

    def test_messages_may_hold_markup_like_text(self):
        line = self.log("INFO", "Heard: <red> {braces}", colorize=False)
        self.assertIn("Heard: <red> {braces}", line)


if __name__ == "__main__":
    unittest.main()
