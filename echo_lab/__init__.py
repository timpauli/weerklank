"""Realtime speech-to-speech translation pipeline built on Pipecat.

The pipeline has four exchangeable stages (see ``echo_lab.registry``):

    audio in -> vad -> stt -> llm (translation) -> tts -> audio out
"""
