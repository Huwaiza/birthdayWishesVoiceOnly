"""
Lyrics builder for the birthday song generator.

Produces a list of singable lines, each wrapped in ♪ markers,
with the recipient's name injected at the correct positions.
"""

from typing import List

# Full song structure — each string is one Bark inference call.
# [singing] tag is the key prompt that makes Bark actually sing vs. speak.
# ♪ markers reinforce melodic delivery.
# {name} is replaced with the actual name at runtime.
_SONG_TEMPLATE = [
    # --- verse 1 ---
    "[singing] ♪ Happy birthday to you ♪",
    "[singing] ♪ Happy birthday to you ♪",
    "[singing] ♪ Happy birthday dear {name} ♪",
    "[singing] ♪ Happy birthday to you ♪",
    # --- verse 2 ---
    "[singing] ♪ Happy birthday to you ♪",
    "[singing] ♪ Happy birthday to you ♪",
    "[singing] ♪ Happy birthday dear {name} ♪",
    "[singing] ♪ May your dreams come true ♪",
    # --- bridge ---
    "[singing] ♪ May your day be filled with joy ♪",
    "[singing] ♪ Laughter, love, and happiness ♪",
    "[singing] ♪ We're so glad you're in our lives ♪",
    "[singing] ♪ Happy birthday {name} ♪",
    # --- verse 3 ---
    "[singing] ♪ Happy birthday to you ♪",
    "[singing] ♪ Happy birthday to you ♪",
    "[singing] ♪ Happy birthday dear {name} ♪",
    "[singing] ♪ Happy birthday to you ♪",
    # --- chorus ---
    "[singing] ♪ Hip hip hooray, it's your special day ♪",
    "[singing] ♪ We sing for you {name}, in every way ♪",
    "[singing] ♪ May all your wishes all come true ♪",
    "[singing] ♪ Happy birthday, happy birthday to you ♪",
    # --- outro ---
    "[singing] ♪ Happy birthday dear {name} ♪",
    "[singing] ♪ Happy birthday dear {name} ♪",
]

# Which line indices start a new section (used for longer pauses during mixing)
SECTION_BREAKS = {0, 4, 8, 12, 16, 20}


def build_lines(name: str) -> List[str]:
    """
    Return the full list of singable lines with {name} replaced.

    Parameters
    ----------
    name : str
        The recipient's name, e.g. "Huwaiza"

    Returns
    -------
    List[str]
        One string per Bark inference call, each wrapped in ♪ markers.
    """
    return [line.format(name=name) for line in _SONG_TEMPLATE]
