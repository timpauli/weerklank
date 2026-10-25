"""VAD stage: decides where speech segments start and end.

Every backend marks segments with ``VADUserStartedSpeakingFrame`` /
``VADUserStoppedSpeakingFrame``; the STT stage transcribes the audio between them.
"""
