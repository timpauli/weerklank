"""Listing the PyAudio devices for ``audio.input_device`` / ``audio.output_device``."""

import pyaudio


def describe_device(index: int | None, kind: str) -> str:
    """Name a device for the log; ``None`` is the system default ``kind`` ("input"/"output")."""
    pa = pyaudio.PyAudio()
    try:
        if index is None:
            default = (
                pa.get_default_input_device_info
                if kind == "input"
                else pa.get_default_output_device_info
            )
            info, label = default(), "system default: "
        else:
            info, label = pa.get_device_info_by_index(index), ""
        return f"[{info['index']}] {label}{info['name']}"
    except (OSError, ValueError) as e:
        return f"[{index}] (not available: {e})"
    finally:
        pa.terminate()


def print_devices() -> None:
    """Print every input and output device with the index to put in the config."""
    pa = pyaudio.PyAudio()
    try:
        host_apis = {
            i: pa.get_host_api_info_by_index(i)["name"] for i in range(pa.get_host_api_count())
        }
        devices = [pa.get_device_info_by_index(i) for i in range(pa.get_device_count())]
        for kind, channels in (("input", "maxInputChannels"), ("output", "maxOutputChannels")):
            print(f"\n{kind.upper()} DEVICES")
            print(f"  {'idx':>4}  {'host API':<22} {'rate':>6}  name")
            for d in devices:
                if d[channels] > 0:
                    host_api, rate = host_apis[d["hostApi"]], int(d["defaultSampleRate"])
                    print(f"  {d['index']:>4}  {host_api:<22} {rate:>6}  {d['name']}")
        print(
            "\nUse a device listed under MME or DirectSound: "
            "they convert the sample rate, WASAPI doesn't."
        )
    finally:
        pa.terminate()
