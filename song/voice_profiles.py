"""
Voice rotation profiles for ACE-Step.

ACE-Step doesn't expose a speaker ID — timbre/delivery is steered by the
natural-language prompt. These four profiles each bias the model toward
a distinct adult female English vocalist sound, and each ships with a
fixed base seed so the same (name, voice_idx) always produces the same
audio.

Each profile carries TWO prompts:

  * prompt_acapella — the default. Asks ACE-Step for continuous solo
    singing with no instrumental intro, breaks, or outro. This is what
    the vocal-only pipeline renders, so that once the (still slightly
    present) backing is separated out there are no long silent gaps
    left where an instrumental section used to be.
  * prompt — the full-band arrangement (a warm piano ballad). Used only
    when the caller opts into --with-music.

Rotation is deterministic: hash(name) % len(VOICE_PROFILES).
"""

from dataclasses import dataclass
from typing import List
import hashlib


@dataclass(frozen=True)
class VoiceProfile:
    name: str             # short human-readable ID, e.g. "warm-alto"
    prompt: str           # full-band prompt — used by --with-music
    prompt_acapella: str  # vocal-forward prompt — the default
    base_seed: int        # pinned seed so renders are reproducible


# Shared prompt tails. Only the vocalist descriptor changes between
# profiles; the arrangement tail is held constant so the four voices
# stay comparable.
_COMMON_MUSIC = (
    "happy birthday song, warm piano ballad, light strings, "
    "studio recording, clean mix, 90 bpm, major key, joyful, "
    "professional production, loudness normalised"
)

# The a cappella tail explicitly suppresses the instrumental intro,
# fills, and outro a ballad arrangement would otherwise insert — those
# are exactly what become long silent gaps once the backing is
# separated out. It also nudges the model to sing the lyrics close
# together rather than spacing them for a band.
_COMMON_ACAPELLA = (
    "a cappella, solo female voice, no instruments, no backing track, "
    "no instrumental intro, no instrumental breaks, no instrumental outro, "
    "continuous singing, steady natural phrasing, lyrics sung close "
    "together, studio vocal recording, clean, warm, joyful, major key"
)

# (name, vocalist descriptor, base seed). The descriptor is shared by
# both prompts; only the arrangement tail differs between modes.
_VOICES = [
    (
        "warm-alto",
        "female vocalist, warm alto voice, soulful, expressive vibrato, "
        "mid-range, slightly breathy, acoustic pop delivery",
        20260101,
    ),
    (
        "bright-soprano",
        "female vocalist, bright soprano voice, clear high notes, youthful, "
        "crisp articulation, cheerful pop delivery",
        20260202,
    ),
    (
        "intimate-mezzo",
        "female vocalist, intimate mezzo-soprano voice, smooth and breathy, "
        "indie pop delivery, close-mic'd, gentle phrasing",
        20260303,
    ),
    (
        "folk-clear",
        "female vocalist, clear folk-pop voice, natural timbre, "
        "storyteller delivery, no heavy processing, sincere phrasing",
        20260404,
    ),
]

VOICE_PROFILES: List[VoiceProfile] = [
    VoiceProfile(
        name=_name,
        prompt=f"{_voice}, {_COMMON_MUSIC}",
        prompt_acapella=f"{_voice}, {_COMMON_ACAPELLA}",
        base_seed=_seed,
    )
    for _name, _voice, _seed in _VOICES
]


def prompt_for(profile: VoiceProfile, with_music: bool = False) -> str:
    """
    Return the prompt to render with for the requested mode.

    with_music=False (default) → the a cappella / continuous-singing
    prompt; with_music=True → the full piano-ballad arrangement.
    """
    return profile.prompt if with_music else profile.prompt_acapella


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
