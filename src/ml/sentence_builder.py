"""
sentence_builder.py
--------------------
Accumulates consecutively confirmed sign words into coherent sentences.
Provides editing controls (backspace, clear, space) and sentence playback.
"""

from typing import List


class SentenceBuilder:
    """Manages sentence construction from individual sign words."""

    def __init__(self, max_words: int = 15):
        self.words: List[str] = []
        self.max_words = max_words

    def add_word(self, word: str) -> bool:
        """
        Adds a new word to the sentence if it differs from the last word
        or if the sentence is empty.
        Returns True if added.
        """
        word = word.strip()
        if not word or word == "...":
            return False

        if len(self.words) >= self.max_words:
            return False

        # Prevent adding the exact same word twice consecutively
        if self.words and self.words[-1].lower() == word.lower():
            return False

        self.words.append(word.upper())
        return True

    def remove_last(self) -> bool:
        """Removes the most recent word (Backspace)."""
        if self.words:
            self.words.pop()
            return True
        return False

    def clear(self):
        """Clears the entire sentence."""
        self.words.clear()

    def get_sentence(self) -> str:
        """Returns the full constructed sentence as a formatted string."""
        return " ".join(self.words) if self.words else "..."

    def is_empty(self) -> bool:
        """Returns True if no words have been added yet."""
        return len(self.words) == 0
