"""
Lyrics builder for the birthday song generator (v7, ACE-Step).

Produces a single lyrics string with [verse]/[bridge]/[chorus]/[outro]
section tags that ACE-Step uses for structural/style conditioning.
"""

# Each line is short (4-8 words) — ACE-Step aligns short lines cleanly.
# {name} is replaced with the recipient's name at runtime.
# The order below is the final song order: verse, verse, bridge, verse, chorus, outro.
_SONG_TEMPLATE = """[verse]
Happy birthday to you
Happy birthday to you
Happy birthday dear {name}
Happy birthday to you

[verse]
Happy birthday to you
Happy birthday to you
Happy birthday dear {name}
May your dreams come true

[bridge]
May your day be filled with joy
Laughter, love, and happiness
We're so glad you're in our lives
Happy birthday {name}

[verse]
Happy birthday to you
Happy birthday to you
Happy birthday dear {name}
Happy birthday to you

[chorus]
Hip hip hooray it's your special day
We sing for you {name} in every way
May all your wishes come true today
Happy birthday, happy birthday to you

[outro]
Happy birthday dear {name}
Happy birthday dear {name}
"""


def build_lyrics(name: str) -> str:
    """
    Return the full ACE-Step lyrics string with {name} replaced.

    Parameters
    ----------
    name : str
        The recipient's name, e.g. "Huwaiza"

    Returns
    -------
    str
        Multi-section lyrics blob ready to pass to ACEStepPipeline.
    """
    if not name or not name.strip():
        raise ValueError("name must be a non-empty string")
    return _SONG_TEMPLATE.format(name=name.strip())


# Back-compat alias — older callers imported build_lines.
# Deprecated: use build_lyrics() instead.
def build_lines(name: str):
    """Deprecated. Returns lyrics as a list split by blank lines (section blocks)."""
    blob = build_lyrics(name)
    return [b.strip() for b in blob.split("\n\n") if b.strip()]
