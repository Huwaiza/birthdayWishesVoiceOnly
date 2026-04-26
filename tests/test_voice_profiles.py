"""Tests for song.voice_profiles — deterministic voice rotation."""

import pytest
from song.voice_profiles import VOICE_PROFILES, pick_voice, seed_for


def test_four_voice_profiles_defined():
    assert len(VOICE_PROFILES) == 4


def test_each_profile_has_required_fields():
    for v in VOICE_PROFILES:
        assert v.name and isinstance(v.name, str)
        assert v.prompt and "female" in v.prompt.lower()
        assert isinstance(v.base_seed, int) and v.base_seed > 0


def test_profile_names_are_unique():
    names = [v.name for v in VOICE_PROFILES]
    assert len(set(names)) == len(names)


def test_pick_voice_deterministic():
    assert pick_voice("Huwaiza").name == pick_voice("Huwaiza").name
    assert pick_voice("Sara").name == pick_voice("Sara").name


def test_pick_voice_case_and_whitespace_insensitive():
    assert pick_voice("Huwaiza").name == pick_voice("  HUWAIZA  ").name


def test_pick_voice_override_pins_index():
    for i in range(len(VOICE_PROFILES)):
        assert pick_voice("AnyName", override_index=i).name == VOICE_PROFILES[i].name


def test_pick_voice_override_out_of_range_raises():
    with pytest.raises(ValueError):
        pick_voice("Huwaiza", override_index=99)
    with pytest.raises(ValueError):
        pick_voice("Huwaiza", override_index=-1)


def test_pick_voice_distributes_across_names():
    """Across 50 sample names we should hit at least 3 of the 4 profiles."""
    sample = [
        "Alice", "Bob", "Charlie", "Diana", "Ethan", "Fatima", "Grace", "Hassan",
        "Ivy", "Jack", "Kira", "Liam", "Maya", "Noah", "Olivia", "Pierre",
        "Quinn", "Rosa", "Sam", "Tara", "Umar", "Vera", "Will", "Xiomara",
        "Yara", "Zoe", "Adam", "Bella", "Carlos", "Dora", "Eli", "Faye",
        "George", "Hana", "Isla", "Jose", "Kate", "Leo", "Mia", "Nia",
        "Omar", "Petra", "Riya", "Sven", "Tia", "Uri", "Vivi", "Wes",
        "Yuki", "Zane",
    ]
    hit = {pick_voice(n).name for n in sample}
    assert len(hit) >= 3, f"Voice rotation too clumpy: only hit {hit}"


def test_seed_deterministic_per_name():
    p = VOICE_PROFILES[0]
    assert seed_for(p, "Huwaiza") == seed_for(p, "Huwaiza")
    assert seed_for(p, "Huwaiza") == seed_for(p, "  huwaiza  ")


def test_seed_varies_by_name():
    p = VOICE_PROFILES[0]
    seeds = {seed_for(p, n) for n in ["Alice", "Bob", "Charlie", "Diana"]}
    # Vanishingly unlikely to collide on 4 names within a 100k window
    assert len(seeds) == 4
