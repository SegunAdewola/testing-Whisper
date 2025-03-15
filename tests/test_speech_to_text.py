"""
Test suite for the SpeechToText module.
"""

import os
import pytest
from source.speech_to_text import SpeechToText

@pytest.fixture
def stt():
    """Fixture to initialize SpeechToText before each test."""
    return SpeechToText(model_size="tiny")

def test_transcribe(stt):
    """Tests the transcription accuracy of the SpeechToText module."""
    audio_file = os.path.join("audio", "New Recording 5.m4a")  # Portable path handling
    text = stt.transcribe(audio_file)

    expected_text = "This is it for this recording. The audio should be perfect, done."

    # Assert that the transcription is reasonably accurate
    assert expected_text.lower() in text.lower(), f"Expected '{expected_text}', but got '{text}'"
