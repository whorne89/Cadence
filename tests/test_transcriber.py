import pytest
from unittest.mock import patch


def test_transcriber_init():
    """Transcriber initializes with correct defaults."""
    from src.core.transcriber import Transcriber
    t = Transcriber(model_size="base")
    assert t.model_size == "base"
    assert t.model is None  # lazy loaded


def test_transcriber_model_sizes():
    """Transcriber accepts valid model sizes."""
    from src.core.transcriber import Transcriber
    assert "tiny" in Transcriber.VALID_MODELS
    assert "base" in Transcriber.VALID_MODELS
    assert "small" in Transcriber.VALID_MODELS
    assert "medium" in Transcriber.VALID_MODELS


def test_transcriber_change_model():
    """change_model should update model_size and clear loaded model."""
    from src.core.transcriber import Transcriber
    t = Transcriber(model_size="base")
    t.change_model("small")
    assert t.model_size == "small"
    assert t.model is None
