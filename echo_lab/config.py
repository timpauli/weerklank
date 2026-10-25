"""Loading, overriding and validating the YAML configuration."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class ConfigError(ValueError):
    """Raised for missing or invalid configuration values."""


class _PathLoader(yaml.SafeLoader):
    """YAML loader that turns ``!path some/file`` into an absolute path.

    Relative paths are resolved against ``folder``, the folder of the file being
    read, so a config works from any working directory.
    """

    folder: Path


def _construct_path(loader: _PathLoader, node: yaml.ScalarNode) -> str:
    return str((loader.folder / Path(loader.construct_scalar(node)).expanduser()).resolve())


_PathLoader.add_constructor("!path", _construct_path)


def load_config(path: str | Path | None = None, presets: Sequence[str] = ()) -> dict[str, Any]:
    """Load the base config file, with presets merged on top in order.

    Values tagged ``!path`` are resolved against the directory of the file
    they're in, so the config works from any working directory.

    Args:
        path: Base config file. Defaults to ``config.yaml`` in the project root.
        presets: Partial configs, each a file path or the name of a file in the
            ``presets`` folder next to the base config (``nl1`` for
            ``presets/nl1.yaml``).

    Returns:
        The configuration as nested dicts.
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    config = _read_yaml(path)
    for preset in presets:
        merge(config, _read_yaml(find_preset(preset, path.parent / "presets")))
    return config


def find_preset(spec: str, presets_dir: Path) -> Path:
    """Turn a preset name or path into the preset's file."""
    for candidate in (Path(spec), presets_dir / f"{spec}.yaml"):
        if candidate.is_file():
            return candidate
    available = ", ".join(sorted(p.stem for p in presets_dir.glob("*.yaml"))) or "none"
    raise ConfigError(f"Preset '{spec}' not found (presets in {presets_dir}: {available})")


def merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Merge ``override`` into ``base`` section by section; other values replace."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        loader = _PathLoader(f)
        loader.folder = path.resolve().parent
        try:
            config = loader.get_single_data() or {}
        finally:
            loader.dispose()
    if not isinstance(config, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    return config


def get_value(config: dict[str, Any], key: str, default: Any = None) -> Any:
    """Read a dotted key such as ``"tts.omnivoice.num_step"``."""
    node: Any = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def set_value(config: dict[str, Any], key: str, value: Any) -> None:
    """Set a dotted key such as ``"tts.omnivoice.num_step"``, creating sections as needed."""
    *parents, leaf = key.split(".")
    node = config
    for part in parents:
        child = node.get(part)
        if child is None:
            child = node[part] = {}
        elif not isinstance(child, dict):
            raise ConfigError(f"Cannot set '{key}': '{part}' is not a section")
        node = child
    node[leaf] = value


def parse_override(text: str) -> tuple[str, Any]:
    """Parse a ``KEY=VALUE`` override; the value is read as YAML (numbers, booleans, null)."""
    key, sep, raw = text.partition("=")
    if not sep or not key.strip():
        raise ConfigError(
            f"Override '{text}' must look like KEY=VALUE, e.g. vad.fixed_interval.interval_secs=3"
        )
    raw = raw.strip()
    return key.strip(), yaml.safe_load(raw) if raw else None


def validate_config(config: dict[str, Any]) -> None:
    """Check the values every pipeline needs, failing early with a clear message."""
    for stage in ("vad", "stt", "llm", "tts"):
        if not get_value(config, f"{stage}.backend"):
            raise ConfigError(f"'{stage}.backend' is not set")

    for key, meaning in (
        ("llm.input_language", "the language the speaker uses, e.g. nl"),
        ("llm.output_language", "the language to translate into, e.g. pt"),
    ):
        language = get_value(config, key)
        if language is None or language == "":
            raise ConfigError(f"'{key}' is not set ({meaning})")
        if not isinstance(language, str):
            # YAML reads a bare `no` (Norwegian) as false.
            raise ConfigError(f"'{key}' must be a language code; put it in quotes")

    ref_audio = get_value(config, "tts.ref_audio")
    if ref_audio and not Path(ref_audio).expanduser().is_file():
        raise ConfigError(f"Reference audio not found: {ref_audio}")
