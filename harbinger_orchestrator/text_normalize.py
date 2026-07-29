"""
Harbinger E2E — Text Normalization
Ensures every sentence-ending '!' or '?' is followed by a '.' so that
downstream splitting (this orchestrator or the TTS service) can reliably
split on '.' alone, even if the LLM ever deviates from its own formatting rule.

Example:
    "You cannot resist!" -> "You cannot resist!."
    "Why do you struggle?" -> "Why do you struggle?."
    "Already correct!." -> "Already correct!." (untouched, no double dot)
"""

import re

# Match a '!' or '?' that is NOT already immediately followed by a '.'
_PUNCT_PATTERN = re.compile(r"([!?])(?!\.)")


def normalize_punctuation(text: str) -> str:
    return _PUNCT_PATTERN.sub(r"\1.", text)
