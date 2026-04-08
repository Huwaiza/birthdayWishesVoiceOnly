import pytest
from song.lyrics import build_lines

def test_name_injected_in_lines():
    lines = build_lines("Huwaiza")
    full = " ".join(lines)
    assert "Huwaiza" in full

def test_returns_list_of_strings():
    lines = build_lines("Sara")
    assert isinstance(lines, list)
    assert all(isinstance(l, str) for l in lines)

def test_each_line_has_music_markers():
    lines = build_lines("Adam")
    for line in lines:
        assert line.startswith("♪"), f"Missing leading ♪ in: {line}"
        assert line.endswith("♪"), f"Missing trailing ♪ in: {line}"

def test_minimum_line_count():
    lines = build_lines("Tom")
    assert len(lines) >= 18  # full song structure

def test_section_separators_not_in_lines():
    lines = build_lines("Layla")
    for line in lines:
        assert not line.startswith("["), f"Section header leaked into lines: {line}"
