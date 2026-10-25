"""Unit tests for config loading, presets, overrides, validation and the registry."""

import tempfile
import unittest
from pathlib import Path

from echo_lab.cli import build_config, parse_args
from echo_lab.config import (
    ConfigError,
    get_value,
    load_config,
    parse_override,
    set_value,
    validate_config,
)
from echo_lab.registry import BACKENDS, STAGES, create_stage
from echo_lab.vad.fixed_interval import FixedIntervalSegmenter


def minimal_config() -> dict:
    return {
        "vad": {"backend": "fixed_interval", "fixed_interval": {"interval_secs": 4.0}},
        "stt": {"backend": "parakeet_tdt"},
        "llm": {"backend": "m2m100_ct2", "input_language": "nl", "output_language": "pt"},
        "tts": {"backend": "omnivoice"},
    }


class OverrideTest(unittest.TestCase):
    def test_values_are_read_as_yaml(self):
        self.assertEqual(
            parse_override("vad.fixed_interval.interval_secs=3"),
            ("vad.fixed_interval.interval_secs", 3),
        )
        self.assertEqual(parse_override("tts.ref_audio=null"), ("tts.ref_audio", None))
        self.assertEqual(
            parse_override("audio.mute_input_while_speaking=true"),
            ("audio.mute_input_while_speaking", True),
        )
        self.assertEqual(parse_override("vad.backend=fixed_interval")[1], "fixed_interval")

    def test_rejects_missing_equals(self):
        with self.assertRaises(ConfigError):
            parse_override("vad.backend")

    def test_set_value_creates_sections(self):
        config = {}
        set_value(config, "tts.omnivoice.speed", 1.1)
        self.assertEqual(get_value(config, "tts.omnivoice.speed"), 1.1)
        self.assertIsNone(get_value(config, "tts.omnivoice.num_step"))


class ValidateTest(unittest.TestCase):
    def test_minimal_config_is_valid(self):
        validate_config(minimal_config())

    def test_missing_language(self):
        config = minimal_config()
        del config["llm"]["input_language"]
        with self.assertRaisesRegex(ConfigError, "llm.input_language"):
            validate_config(config)

    def test_language_read_as_boolean(self):
        config = minimal_config()
        config["llm"]["output_language"] = False  # what YAML makes of a bare `no`
        with self.assertRaisesRegex(ConfigError, "quotes"):
            validate_config(config)

    def test_missing_reference_audio(self):
        config = minimal_config()
        config["tts"]["ref_audio"] = "does-not-exist.wav"
        with self.assertRaisesRegex(ConfigError, "Reference audio"):
            validate_config(config)


class PresetTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.base = self.dir / "config.yaml"
        self.base.write_text(
            "vad:\n  backend: fixed_interval\n"
            "stt:\n  backend: parakeet_tdt\n"
            "llm:\n  backend: m2m100_ct2\n  input_language: nl\n  output_language: pt\n"
            "tts:\n  backend: omnivoice\n  omnivoice:\n    num_step: 32\n    speed: null\n",
            encoding="utf-8",
        )
        (self.dir / "presets").mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def preset(self, name: str, text: str) -> Path:
        path = self.dir / "presets" / f"{name}.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_preset_overrides_only_what_it_sets(self):
        self.preset("fast", "tts:\n  omnivoice:\n    num_step: 16\n")
        config = load_config(self.base, ["fast"])
        self.assertEqual(get_value(config, "tts.omnivoice.num_step"), 16)
        self.assertIsNone(get_value(config, "tts.omnivoice.speed"))  # sibling kept
        self.assertEqual(get_value(config, "tts.backend"), "omnivoice")
        self.assertEqual(get_value(config, "llm.input_language"), "nl")

    def test_presets_apply_in_order_and_by_path(self):
        self.preset("a", "llm:\n  output_language: nl\n")
        other = self.dir / "elsewhere.yaml"
        other.write_text("llm:\n  output_language: en\n", encoding="utf-8")
        self.assertEqual(
            get_value(load_config(self.base, ["a", str(other)]), "llm.output_language"), "en"
        )
        self.assertEqual(
            get_value(load_config(self.base, [str(other), "a"]), "llm.output_language"), "nl"
        )

    def test_relative_paths_resolve_against_the_preset(self):
        self.preset("voice", "tts:\n  ref_audio: !path me.wav\n")
        config = load_config(self.base, ["voice"])
        self.assertEqual(
            Path(get_value(config, "tts.ref_audio")), (self.dir / "presets" / "me.wav").resolve()
        )

    def test_unknown_preset_lists_the_available_ones(self):
        self.preset("fast", "{}\n")
        with self.assertRaisesRegex(ConfigError, r"'slow' not found .*: fast"):
            load_config(self.base, ["slow"])

    def test_set_beats_presets(self):
        self.preset("pt", "llm:\n  input_language: pt\ntts:\n  omnivoice:\n    num_step: 16\n")
        args = parse_args(
            [
                "-c",
                str(self.base),
                "-p",
                "pt",
                "--set",
                "llm.input_language=nl",
                "--set",
                "tts.omnivoice.num_step=8",
            ]
        )
        config = build_config(args)
        self.assertEqual(get_value(config, "llm.input_language"), "nl")
        self.assertEqual(get_value(config, "tts.omnivoice.num_step"), 8)

    def test_project_presets_load(self):
        for path in sorted((Path(__file__).resolve().parent.parent / "presets").glob("*.yaml")):
            with self.subTest(preset=path.stem):
                validate_config(load_config(presets=[path.stem]))


class ProjectConfigTest(unittest.TestCase):
    def test_default_config_lists_known_backends(self):
        config = load_config()
        for stage in STAGES:
            self.assertIn(get_value(config, f"{stage}.backend"), BACKENDS[stage])
        self.assertTrue(Path(get_value(config, "tts.ref_audio")).is_absolute())

    def test_tagged_paths_resolve_against_config_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "tts:\n  ref_audio: !path voices/me.wav\n  ref_text: voices/me.wav\n",
                encoding="utf-8",
            )
            config = load_config(path)
        self.assertEqual(
            Path(config["tts"]["ref_audio"]), (Path(tmp) / "voices" / "me.wav").resolve()
        )
        self.assertEqual(config["tts"]["ref_text"], "voices/me.wav")  # untagged: left alone


class RegistryTest(unittest.TestCase):
    def test_builds_a_stage(self):
        [segmenter] = create_stage(minimal_config(), "vad")
        self.assertIsInstance(segmenter, FixedIntervalSegmenter)

    def test_unknown_backend(self):
        config = minimal_config()
        config["vad"]["backend"] = "webrtc"
        with self.assertRaisesRegex(ConfigError, "Available: silero_smart_turn, fixed_interval"):
            create_stage(config, "vad")


if __name__ == "__main__":
    unittest.main()
