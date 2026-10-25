"""LLM stage: translates each transcript into the output language.

The models here are seq2seq translators rather than chat LLMs, but the stage
sits where an LLM sits in a Pipecat voice pipeline and talks to the TTS the
same way an LLM does.
"""
