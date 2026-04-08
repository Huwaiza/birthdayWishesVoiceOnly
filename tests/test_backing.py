import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from song.backing import load_and_loop


def test_returns_numpy_array():
    # Use a tiny synthetic audio array instead of a real file
    fake_audio = np.zeros(24000, dtype=np.float32)  # 1s at 24kHz
    with patch("song.backing.sf.read", return_value=(fake_audio, 24000)):
        with patch("song.backing.Path.exists", return_value=True):
            result = load_and_loop("fake.mp3", target_duration_s=3.0, sample_rate=24000)
    assert isinstance(result, np.ndarray)


def test_output_length_matches_target():
    fake_audio = np.zeros(24000, dtype=np.float32)  # 1s
    with patch("song.backing.sf.read", return_value=(fake_audio, 24000)):
        with patch("song.backing.Path.exists", return_value=True):
            result = load_and_loop("fake.mp3", target_duration_s=3.0, sample_rate=24000)
    expected_samples = int(3.0 * 24000)
    assert abs(len(result) - expected_samples) <= 100  # within ~4ms


def test_missing_file_raises():
    with patch("song.backing.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            load_and_loop("nonexistent.mp3", target_duration_s=10.0)
