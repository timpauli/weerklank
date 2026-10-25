"""Realtime speech-to-speech translation with Pipecat.

    uv run bot.py --list-devices
    uv run bot.py -p nl1
    uv run bot.py --set audio.input_device=1 --set llm.input_language=pt

See README.md and config.yaml for all options.
"""

import sys

from echo_lab.cli import main

if __name__ == "__main__":
    sys.exit(main())
