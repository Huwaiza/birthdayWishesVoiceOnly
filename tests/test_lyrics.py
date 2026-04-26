"""Tests for song.lyrics — v7 ACE-Step tagged-lyrics builder."""

import pytest
from song.lyrics import build_lyrics, build_lines


def test_build_lyrics_returns_string():
    out = build_lyrics("Huwaiza")
    assert isinstance(out, str)
    assert len(out) > 0


def test_build_lyrics_injects_name():
    out = build_lyrics("Huwaiza")
    assert "Huwaiza" in out
    # name appears in multiple sections
    assert out.count("Huwaiza") >= 4


def test_build_lyrics_has_section_tags():
    out = build_lyrics("Sara")
    for tag in ("[verse]", "[bridge]", "[chorus]", "[outro]"):
        assert tag in out, f"missing {tag} in:\n{out}"


def test_build_lyrics_no_bark_markers():
    """The Bark hacks ([singing] tag, ♪ markers) must not leak into ACE-Step lyrics."""
    out = build_lyrics("Adam")
    assert "[singing]" not in out
    assert "♪" not in out


def test_build_lyrics_strips_whitespace_in_name():
    out = build_lyrics("  Layla  ")
    assert "Layla" in out
    assert "  Layla  " not in out


def test_build_lyrics_rejects_empty_name():
    with pytest.raises(ValueError):
        build_lyrics("")
    with pytest.raises(ValueError):
        build_lyrics("   ")


def test_build_lines_back_compat_returns_list():
    """Older callers used build_lines(); shim still returns a list of section blocks."""
    blocks = build_lines("Tom")
    assert isinstance(blocks, list)
    assert len(blocks) >= 5  # verse, verse, bridge, verse, chorus, outro
    assert all(isinstance(b, str) for b in blocks)


def test_build_lines_blocks_contain_section_headers():
    blocks = build_lines("Tom")
    headers = [b.split("\n", 1)[0] for b in blocks]
    assert any("[verse]" in h for h in headers)
    assert any("[chorus]" in h for h in headers)
