"""Changing parameters while the pipeline runs, over OSC.

A processor that can be changed while it runs lists its parameters in a
``controls()`` method (see :class:`Controllable`). Each :class:`Control` names a
parameter, checks an incoming value and builds the Pipecat frame that applies
it. :class:`OSCServer` receives UDP messages ``/<stage>/<name> <value>``, e.g.
``/tts/speed 1.2``, and queues the matching frame into the running pipeline. The
frame travels down the pipeline in order with everything already in flight, and
the processor applies it when it arrives; for the TTS, from the next sentence on.

To make another stage controllable, give its processor a ``controls()`` method.
For a Pipecat service, :func:`settings_controls` builds them from its settings.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from loguru import logger
from pipecat.frames.frames import Frame, ServiceUpdateSettingsFrame
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.ai_service import AIService
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer

from echo_lab.config import ConfigError

Parser = Callable[[Any], Any]


@dataclass(frozen=True)
class Control:
    """One parameter that can be changed while the pipeline runs.

    Parameters:
        name: The parameter's name, the last part of its OSC address.
        parse: Checks and converts the OSC argument; raises ValueError if it doesn't fit.
        frame: Builds the frame that applies the parsed value.
    """

    name: str
    parse: Parser
    frame: Callable[[Any], Frame]


@runtime_checkable
class Controllable(Protocol):
    """A processor with parameters that can be changed while it runs."""

    def controls(self) -> list[Control]: ...


def settings_controls(
    service: AIService,
    frame_type: type[ServiceUpdateSettingsFrame],
    parsers: dict[str, Parser],
) -> list[Control]:
    """Controls for fields of a Pipecat service's settings.

    Each value is applied with the service's settings-update frame, e.g.
    ``TTSUpdateSettingsFrame``, aimed at that service only.

    Args:
        service: The service; its settings class must have the fields.
        frame_type: The settings-update frame the service handles.
        parsers: Field name -> function that checks and converts the OSC argument.
    """
    settings_type = type(service.settings)

    def control(name: str, parse: Parser) -> Control:
        def frame(value: Any) -> Frame:
            return frame_type(delta=settings_type(**{name: value}), service=service)

        return Control(name, parse, frame)

    return [control(name, parse) for name, parse in parsers.items()]


# Parsers for OSC arguments. OSC apps differ in what they send (a fader may send
# an int or a float, a toggle a bool or 0/1), so these accept every reasonable form.


def number(*, above: float | None = None, at_least: float | None = None) -> Parser:
    """An int or float, optionally bounded."""

    def parse(value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"expected a number, got {value!r}")
        # OSC floats are 32-bit (1.2 arrives as 1.2000000476837158); keep what
        # that precision actually carries.
        value = float(f"{value:.6g}")
        if above is not None and not value > above:
            raise ValueError(f"must be greater than {above}, got {value}")
        if at_least is not None and not value >= at_least:
            raise ValueError(f"must be at least {at_least}, got {value}")
        return value

    return parse


def flag(value: Any) -> bool:
    """True/false, or a number (0 = false)."""
    if isinstance(value, bool):
        return value
    return number()(value) != 0


def optional(parse: Parser) -> Parser:
    """``parse``, or None for a message without a value (or with an OSC nil)."""
    return lambda value: None if value is None else parse(value)


class OSCServer:
    """Receives OSC messages and applies them to the running pipeline.

    Used as ``async with OSCServer(host, port) as osc:``, which binds the UDP
    port and closes it again. Inside, :meth:`add` registers a stage's
    controllable processors, and :meth:`attach` connects the pipeline once it
    has started; messages that arrive before that are ignored. With ``port``
    None the server is off and every method does nothing.
    """

    def __init__(self, host: str, port: int | None):
        self.host = host
        self.port = port  # with port 0 the system picks a free one; this is updated to it
        self._dispatcher = Dispatcher()
        self._dispatcher.set_default_handler(self._handle_unknown)
        self._controls: dict[str, Control] = {}
        self._worker: PipelineWorker | None = None
        self._transport: asyncio.BaseTransport | None = None
        self._tasks: set[asyncio.Task] = set()

    async def __aenter__(self) -> "OSCServer":
        if self.port is None:
            return self
        server = AsyncIOOSCUDPServer(
            (self.host, self.port), self._dispatcher, asyncio.get_running_loop()
        )
        try:
            self._transport, _ = await server.create_serve_endpoint()
        except OSError as e:
            raise ConfigError(
                f"OSC can't listen on {self.host}:{self.port} ({e}); set osc.port to a free port"
            ) from e
        self.port = self._transport.get_extra_info("sockname")[1]
        return self

    async def __aexit__(self, *exc_info: Any):
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    def add(self, stage: str, processors: list[FrameProcessor]):
        """Register the controls of a stage's processors, as ``/<stage>/<name>``."""
        for processor in processors:
            if not isinstance(processor, Controllable):
                continue
            for control in processor.controls():
                address = f"/{stage}/{control.name}"
                self._controls[address] = control
                self._dispatcher.map(address, self._handle)

    def attach(self, worker: PipelineWorker):
        """Start applying messages to ``worker``'s pipeline (call once it has started)."""
        if self.port is None:
            return
        self._worker = worker
        addresses = ", ".join(self._controls) or "none"
        logger.info(f"OSC: listening on {self.host}:{self.port} (UDP) for {addresses}")

    def _handle(self, address: str, *args: Any):
        if len(args) > 1:
            logger.warning(f"OSC {address}: expected one value, got {len(args)}")
            return
        control = self._controls[address]
        try:
            # No value is passed on as None: an optional parameter goes back to
            # its default, any other rejects it.
            value = control.parse(args[0] if args else None)
        except ValueError as e:
            logger.warning(f"OSC {address}: {e}")
            return
        if self._worker is None:
            logger.warning(f"OSC {address}: ignored, the pipeline hasn't started yet")
            return

        logger.info(f"OSC {address} = {value}")
        # Called by the UDP protocol inside the event loop; queueing is async.
        task = asyncio.get_running_loop().create_task(
            self._worker.queue_frame(control.frame(value))
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _handle_unknown(self, address: str, *args: Any):
        known = ", ".join(self._controls) or "none"
        logger.warning(f"OSC {address}: unknown address (known: {known})")
