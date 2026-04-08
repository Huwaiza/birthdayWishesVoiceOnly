"""
Lyrics builder for the birthday song generator.

Produces a list of singable lines, each wrapped in ♪ markers,
with the recipient's name injected at the correct positions.
"""

from typing import List

# Full song structure — each string is one Bark inference line.
# {name} is replaced with the actual name at runtime.
_SONG_TEMPLATE = [
    # --- verse 1 ---
    "♪ Happy birthday to you ♪",
    "♪ Happy birthday to you ♪",
    "♪ Happy birthday dear {name} ♪",
    "♪ Happy birthday to you ♪",
    # --- verse 2 ---
    "♪ Happy birthday to you ♪",
    "♪ Happy birthday to you ♪",
    "♪ Happy birthday dear {name} ♪",
    "♪ May your dreams come true ♪",
    # --- bridge ---
    "♪ May your day be filled with joy ♪",
    "♪ Laughter, love, and happiness ♪",
    "♪ We're so glad you're in our lives ♪",
    "♪ Happy birthday {name} ♪",
    # --- verse 3 ---
    "♪ Happy birthday to you ♪",
    "♪ Happy birthday to you ♪",
    "♪ Happy birthday dear {name} ♪",
    "♪ Happy birthday to you ♪",
    # --- chorus ---
    "♪ Hip hip hooray, it's your special day ♪",
    "♪ We sing for you {name}, in every way ♪",
    "♪ May all your wishes all come true ♪",
    "♪ Happy birthday, happy birthday to you ♪",
    # --- outro ---
    "♪ Happy birthday dear {name} ♪",
    "♪ Happy birthday dear {name} ♪",
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
