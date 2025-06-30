import re
import string
from typing import Optional

from pathvalidate import sanitize_filepath

from configs import YouTubeConfig


def sanitize_title(raw_title: str) -> str:
    """
    Common file cleaner, some models do not correctly handle filepaths that are technically fine (emojis etc)
    :param raw_title:
    :return:
    """
    sanitize = ''.join([i if ord(i) < 128 else '_' for i in raw_title])
    sanitize = sanitize.replace('/', '_').replace(r'\\', '_')
    sanitized_title = sanitize_filepath(sanitize, platform="auto")

    return sanitized_title


class EpisodeTitleParser:
    HONORIFICS = [
                     'Mr', 'Mrs', 'Miss', 'Ms', 'Dr', 'Admiral', 'Ambassador', 'Baron', 'Baroness', 'Brigadier',
                     'Brother',
                     'Capt', 'Captain', 'Chief', 'Col', 'Colonel', 'Commander', 'Cmdr', 'Countess', 'Cpl', 'Corporal',
                     'Dame',
                     'Duchess', 'Duke', 'Earl', 'Father', 'General', 'Judge', 'Justice', 'Lady', 'Lord', 'Lt',
                     'Lieutenant',
                     'Madam', 'Madame', 'Major', 'Major General', 'Minister', 'Professor', 'Prof', 'Rabbi', 'Rev',
                     'Reverend',
                     'Senator', 'Sgt', 'Sergeant', 'Sheriff', 'Sir', 'Sister', 'Ret', 'Retired'
                 ] + list(string.ascii_uppercase)

    '''
    try:
        # Try to load it directly
        spacy.load("en_core_web_sm")
    except OSError:
        # Model not found; try to download it
        import spacy.cli

        spacy.cli.download("en_core_web_sm")
    
    nlp = spacy.load("en_core_web_sm")
    '''

    def __init__(self, title: str, config: YouTubeConfig):
        """
        Initialize the parser with the episode title and YouTube configuration.

        :param title: The raw episode title string, e.g., "Joe Rogan Experience #2330 - Bono".
        :param config: The YouTubeConfig object containing parsing settings.
        """
        self.title = title.strip()
        self.config = config

    def extract_episode_number(self) -> int:
        """
        Extract the episode number from the title using the configured prefix or the first standalone number.

        :return: The episode number as an integer, or 0 if none found.
        """
        text = self.title
        if not re.search(r'\d', text):
            return 0

        prefix = self.config.episode_prefix
        if prefix:
            match = re.search(f"{re.escape(prefix)}(\\d+)", text)
            if match:
                return int(match.group(1))

        match = re.search(r'\b\d+\b', text)
        return int(match.group(0)) if match else 0

    def extract_guest(self) -> str:
        """
        Extract the guest name from the title based on guest_searches configuration, honorifics, and fallback logic.

        :return: The guest name as a string, or an empty string if none found.
        """
        title = self.title
        config = self.config

        # If title is purely numeric/spaces, return as-is
        if re.fullmatch(r"[0-9 ]+", title):
            return title

        # Apply any guest_replace patterns first
        guest_replace = config.guest_replace
        if guest_replace:
            pattern = re.compile('|'.join(map(re.escape, guest_replace)), re.IGNORECASE)
            title = pattern.sub('', title)

        guest = ""
        guest_searches = config.guest_searches

        # Attempt to parse based on configured delimiters
        for search in guest_searches:
            delimiter = search.separator
            # If title ends with the delimiter, strip it before matching
            if delimiter and title.endswith(delimiter):
                title = title[:-len(delimiter)].strip()
            # Search in a case-insensitive manner
            if delimiter and delimiter.lower() in title.lower():
                guest = self._extract_based_on_delimiter(title, delimiter, search.position)
                if guest:
                    break

        # Fallback: remove the episode number portion and pick first words
        if not guest:
            match = re.search(r'\d(\D)', title)
            if match:
                title = title[match.start() + 1:].strip()
            words = title.split()
            guest = self._select_guest_from_words(words)

        # Remove any bracketed content and trim punctuation
        guest = re.sub(r"[\(\[\{].*?[\)\]\}]", "", guest).strip()
        guest = re.sub(r"^[^\w]+|[^\w]+$", "", guest).strip()
        return guest

    @classmethod
    def _select_guest_from_words(cls, words: list[str]) -> str:
        """
        Choose how many words to include in the guest name, giving preference to honorifics.

        :param words: List of words from the title portion.
        :return: A string combining either the first two or three words.
        """
        # If any word matches an honorific, include up to three words
        if any(re.sub(r'[\W]+', '', w) in cls.HONORIFICS for w in words):
            return ' '.join(words[:3])
        # Otherwise, default to first two words
        return ' '.join(words[:2])

    @classmethod
    def _extract_based_on_delimiter(cls, title: str, delimiter: str, position: Optional[str]) -> Optional[str]:
        """
        Extract a name segment based on the delimiter and its position ("before" or "after").

        :param title: The current title string (possibly modified).
        :param delimiter: The separator string to split on, e.g., "-" or ":" or "with".
        :param position: Either "before" or "after" indicating which side of delimiter is guest.
        :return: The extracted name segment, or None if no match.
        """
        escaped_delim = re.escape(delimiter)
        title = title.strip()

        if position == "after":
            pattern = rf'{escaped_delim}\s*([\w.\s\-\'\"]+)'
        elif position == "before":
            pattern = rf'([\w.\s\-\'\"]+)\s*{escaped_delim}(?:\s|$)'
        else:
            raise ValueError("Position must be 'before' or 'after'")

        match = re.search(pattern, title, re.IGNORECASE)
        if not match:
            return None

        name = match.group(1)
        if not name:
            return None

        words = name.split()
        return cls._select_guest_from_words(words)
