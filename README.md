# echo-lab

Realtime speech-to-speech translation built on [Pipecat](https://docs.pipecat.ai) 1.14, with
local models only:

```
audio in ─▶ VAD ─▶ STT ─▶ LLM (translation) ─▶ TTS ─▶ audio out
            silero_smart_turn  parakeet_tdt  m2m100_ct2          omnivoice
            fixed_interval                   translategemma_ct2
```

Every stage is exchangeable: each one names its backend in `config.yaml`.

## Setup

```powershell
uv sync                       # Python 3.12, torch 2.14.1 + CUDA 12.6, all backends
uv run bot.py --list-devices  # find your input and output devices
```

Models download from Hugging Face on first use. M2M100 and TranslateGemma are converted to
CTranslate2 into `models/` on their first run. TranslateGemma is gated: accept its license at
<https://huggingface.co/google/translategemma-4b-it>, then run `uv run hf auth login`.

Everything runs on the GPU set by `gpu` in `config.yaml` (nvidia-smi index 0, the RTX 3090 Ti).
The other GPUs are hidden from the process, so that GPU is `cuda:0` in every `device` setting.

## Run

```powershell
uv run bot.py                                    # the settings from config.yaml
uv run bot.py -p nl1                             # with the preset presets/nl1.yaml
uv run bot.py -p nl1 --output-file out.wav       # record the translation instead of playing it

# Single values with --set, e.g. Portuguese -> Dutch in fixed 4 s chunks
uv run bot.py --set llm.input_language=pt --set llm.output_language=nl `
  --set vad.backend=fixed_interval --set vad.fixed_interval.interval_secs=4
```

Stop with Ctrl+C. `--output-file` records the translation on the session's real timeline
(silence where nothing was translated) and writes the file when you stop.

The terminal shows what happens to each segment: `Heard` lines (red) with the transcript,
`Translated` lines (green) and `Speaking` lines (yellow), each with its timing. `-v` adds debug
output.

## Configuration

`config.yaml` holds every parameter, with a comment on each. Two ways to change values for a
run without editing it, applied in this order (later ones win):

- **Presets** (`-p NAME`): a file in `presets/` holding only the values that differ.
  Sections merge, so a preset with just `tts: {omnivoice: {num_step: 16}}` keeps every other
  `tts` setting. Several presets apply left to right (`-p nl1 -p my_tweaks`), and `-p` also
  takes a path to a preset file anywhere. Included: `nl1`, `nl2`, `pt1`.
- **`--set KEY=VALUE`**: any single value, read as YAML, so `--set tts.ref_audio=null` turns
  voice cloning off.

Paths: a value tagged `!path` (`ref_audio: !path voices/me.wav`) is relative to the file it's in.
An untagged path, and a path given with `--set`, is relative to the folder you run from.

Audio devices are set by their index from `--list-devices`. Pick devices listed under MME or
DirectSound: they convert sample rates, while WASAPI devices only open at their native rate.
Windows can renumber devices after a reboot; the startup log shows which device was opened.

### What the parameters do

| Section | Parameter | Meaning |
|---|---|---|
| — | `gpu` | The only GPU used, by nvidia-smi index; `null` = all GPUs |
| audio | `input_device`, `output_device` | PyAudio device index; `null` = system default |
| audio | `mute_input_while_speaking` | Silence the microphone while the translation plays (speakers in the same room) |
| vad | `backend: silero_smart_turn` | At every pause of `stop_secs`, Smart Turn judges whether the sentence is finished: finished ends the segment, unfinished keeps it open, for at most `max_pause_secs` of silence |
| vad | `backend: fixed_interval` | Ignore pauses and cut about every `interval_secs`, at the quietest point of a `search_secs` wide window around it |
| llm | `input_language` | The speaker's language, translated from (`nl`, `pt`, `de`, ...) |
| llm | `output_language` | The language to translate into; the TTS speaks it |
| tts | `ref_audio`, `ref_text` | Voice to clone and its transcript; `ref_audio: null` = the model's own voice |
| tts | `omnivoice.num_step` | Diffusion steps: fewer is faster, more is cleaner |

Notes:
- **Parakeet TDT v3 detects the spoken language by itself**, but doesn't report which one it
  heard, so the stt section has no language; the translation starts from `llm.input_language`.
- **Language codes are used as written** (`nl`, `pt`, `de`). M2M100 rejects codes it doesn't
  know at startup, such as `pt-BR`.
- **Smart Turn has no length limit:** someone who talks without ever pausing for `stop_secs` is
  translated only once they pause.
- **Fixed-interval cuts** hold back the audio of the search window until it's complete, so each
  cut comes up to `search_secs / 2` later.

## Speed and memory (RTX 3090 Ti, generated test speech)

| Step | Time per sentence |
|---|---|
| Parakeet STT | ~0.1 s |
| M2M100 translation | ~0.1–0.2 s |
| TranslateGemma translation | ~0.26 s (with transformers instead of CTranslate2: ~1.25 s) |
| OmniVoice TTS, 32 steps | ~1.9 s for ~3 s of speech |
| OmniVoice TTS, 16 steps | ~0.9 s for ~3 s of speech |

Translation starts when the VAD stage ends a segment. With Smart Turn that happened ~0.5 s
sooner than with plain Silero and a 0.8 s pause (measured with a 0.2 s `stop_secs` plus ~90 ms
for the Smart Turn check). A hesitation mid-sentence doesn't split the sentence either: on a
Dutch sentence with a 0.7 s pause in the middle, Smart Turn kept it whole, while plain Silero
cut it in two and the first half was translated into nonsense.

GPU memory of one running bot: about 4.6 GB with M2M100 and 12.7 GB with TranslateGemma, about
1 GB more while speaking. Loading briefly needs 9.1 or 17.2 GB, so when several bots share the
GPU, start them one after another.

## Project layout

```
bot.py           entry point (calls echo_lab.cli.main)
config.yaml      the base config, every parameter
presets/         partial configs for -p
echo_lab/
  cli.py         command line, log format, GPU selection
  config.py      loading, presets, --set, !path, validation
  registry.py    the backends of each stage
  pipeline.py    microphone and speakers (or a WAV file) around the four stages
  audio_devices.py  --list-devices and device names for the log
  vad/           base.py, smart_turn.py, fixed_interval.py
  stt/           parakeet_tdt.py          (Pipecat SegmentedSTTService)
  llm/           base.py, m2m100_ct2.py, translategemma_ct2.py   (Pipecat AIService)
  tts/           base.py, omnivoice.py    (Pipecat TTSService)
tests/           unit tests
```

Smart Turn is Pipecat's end-of-turn model (`LocalSmartTurnAnalyzerV3`, bundled with Pipecat,
23 languages including Dutch and Portuguese). Pipecat normally runs it inside the context
aggregator of a conversational bot; this pipeline has no aggregator, so
`echo_lab/vad/smart_turn.py` drives it from the VAD stage instead.

### Adding a backend

A backend is a module with a factory `create(ctx: StageContext)` that returns a Pipecat
`FrameProcessor` (see `echo_lab/registry.py`). List the module in `BACKENDS` under a name, and
put its options in a config section of the same name. A new translator derives from
`TranslationService` and implements `_translate(text)`; a new TTS derives from
`LocalTTSService` and implements `_synthesize(text)`.

## Tests

```powershell
uv run python -m unittest discover -s tests   # no models or audio devices needed
uv run ruff format; uv run ruff check         # formatting and lint
```

The unit tests cover both segmenters, the translation stage (with a stand-in model), the
config handling, the command line and log format, the microphone mute and the output recorder.
Everything with models runs live: speak into the microphone and watch the log.
