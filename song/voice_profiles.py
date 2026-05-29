"""
Voice rotation profiles for ACE-Step.

ACE-Step doesn't expose a speaker ID — timbre/delivery is steered by the
natural-language prompt. These four profiles each bias the model toward
a distinct adult female English vocalist sound, and each ships with a
fixed base seed so the same (name, voice_idx) always produces the same
audio.

Rotation is deterministic: hash(name) % len(VOICE_PROFILES).
"""

from dataclasses import dataclass
from typing import List
import hashlib


@dataclass(frozen=True)
class VoiceProfile:
    name: str            # short human-readable ID, e.g. "warm-alto"
    prompt: str          # ACE-Step text prompt describing the song + vocalist
    base_seed: int       # pinned seed so renders are reproducible


# Keep the common tail of the prompt consistent (tempo/key/mood/instrumentation)
# — only the vocalist descriptor changes. This is what gives us distinguishable
# voices without drifting into different musical arrangements.
_COMMON = (
    "happy birthday song, warm piano ballad, light strings, "
    "studio recording, clean mix, 90 bpm, major key, joyful, "
    "professional production, loudness normalised"
)

VOICE_PROFILES: List[VoiceProfile] = [
    VoiceProfile(
        name="warm-alto",
        prompt=(
            "female vocalist, warm alto voice, soulful, expressive vibrato, "
            "mid-range, slightly breathy, acoustic pop delivery, " + _COMMON
        ),
        base_seed=20260101,
    ),
    VoiceProfile(
        name="bright-soprano",
        prompt=(
            "female vocalist, bright soprano voice, clear high notes, youthful, "
            "crisp articulation, cheerful pop delivery, " + _COMMON
        ),
        base_seed=20260202,
    ),
    VoiceProfile(
        name="intimate-mezzo",
        prompt=(
            "female vocalist, intimate mezzo-soprano voice, smooth and breathy, "
            "indie pop delivery, close-mic'd, gentle phrasing, " + _COMMON
        ),
        base_seed=20260303,
    ),
    VoiceProfile(
        name="folk-clear",
        prompt=(
            "female vocalist, clear folk-pop voice, natural timbre, "
            "storyteller delivery, no heavy processing, sincere phrasing, " + _COMMON
        ),
        base_seed=20260404,
    ),
]


def pick_voice(name: str, override_index: int = None) -> VoiceProfile:
    """
    Deterministically pick a voice profile for a given name.

    Parameters
    ----------
    name : str
        Recipient's name.
    override_index : int or None
        If given, pin to that index (0 .. len(VOICE_PROFILES)-1).
        Otherwise uses hash(name) % N for rotation.

    Returns
    -------
    VoiceProfile
    """
    if override_index is not None:
        if not 0 <= override_index < len(VOICE_PROFILES):
            raise ValueError(
                f"voice index {override_index} out of range 0..{len(VOICE_PROFILES)-1}"
            )
        return VOICE_PROFILES[override_index]

    # MD5 rather than Python hash() — Python hash is salted per process.
    h = hashlib.md5(name.strip().lower().encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(VOICE_PROFILES)
    return VOICE_PROFILES[idx]


def seed_for(profile: VoiceProfile, name: str) -> int:
    """
    Mix the profile's base seed with the name so each recipient gets a
    slightly different micro-variation within the same voice profile.
    Stays deterministic.
    """
    h = hashlib.md5(name.strip().lower().encode("utf-8")).hexdigest()
    name_component = int(h[:8], 16) % 100000
    return profile.base_seed + name_component
