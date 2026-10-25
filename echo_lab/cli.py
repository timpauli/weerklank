"""Command line interface: ``uv run bot.py --help``."""

import argparse
import asyncio
import os
import sys

from loguru import logger

from echo_lab.config import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    load_config,
    parse_override,
    set_value,
    validate_config,
)

# Colors (Loguru markup) for the pipeline's own lines, chosen by how the message starts.
MESSAGE_COLORS = {
    "Heard": "red",
    "Translated": "yellow",
    "Speaking": "green",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line (``argv``, or ``sys.argv`` when None)."""
    parser = argparse.ArgumentParser(
        prog="bot.py",
        description="Realtime speech-to-speech translation: "
        "audio in -> VAD -> STT -> translation -> TTS -> audio out.",
        epilog="Applied in order: base config, presets, --set.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="base YAML config file (default: %(default)s)",
    )
    parser.add_argument(
        "-p",
        "--preset",
        action="append",
        default=[],
        metavar="NAME",
        help="partial config merged over the base config: a file, or a name from the presets "
        "folder (repeatable, applied in order)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any config value, e.g. --set llm.input_language=pt (repeatable)",
    )
    parser.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    parser.add_argument(
        "--output-file",
        metavar="WAV",
        help="write the translated speech to a WAV file instead of a device",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> dict:
    """Load the config file and apply the command line overrides to it."""
    config = load_config(args.config, args.preset)
    for override in args.set:
        set_value(config, *parse_override(override))
    validate_config(config)
    return config


def log_format(record: dict) -> str:
    """Loguru line format: no level label for INFO, and colored pipeline lines."""
    level = "" if record["level"].name == "INFO" else "<level>{level: <8}</level> | "
    color = next(
        (c for prefix, c in MESSAGE_COLORS.items() if record["message"].startswith(prefix)), None
    )
    message = f"<{color}>{{message}}</{color}>" if color else "<level>{message}</level>"
    return (
        "<light-black>{time:YYYY-MM-DD HH:mm:ss.SSS}</light-black> | "
        + level
        + "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        + message
        + "\n{exception}"
    )


def use_only_gpu(index: int | None) -> None:
    """Hide every GPU but one (nvidia-smi numbering); it becomes ``cuda:0``.

    Must run before any model touches CUDA.
    """
    if index is None:
        return
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    logger.info(f"Using only GPU {index} (nvidia-smi numbering), as cuda:0")


def main(argv: list[str] | None = None) -> int:
    """Run the translator; returns the process exit code."""
    args = parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO", format=log_format)
    # Device names and transcripts may hold characters the console code page lacks.
    sys.stdout.reconfigure(errors="replace")

    if args.list_devices:
        from echo_lab.audio_devices import print_devices

        print_devices()
        return 0

    try:
        config = build_config(args)
        use_only_gpu(config.get("gpu"))
        from echo_lab.pipeline import run_pipeline

        asyncio.run(run_pipeline(config, output_file=args.output_file))
    except ConfigError as e:
        logger.error(str(e))
        return 2
    except KeyboardInterrupt:
        logger.info("Stopped")
    return 0
