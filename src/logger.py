import logging
from logging.handlers import RotatingFileHandler

import unicodedata

from configs import get_global_config


def sanitize_log_message(msg: str) -> str:
    """
    Remove or replace characters that are not safe for UTF-8 or are not printable.
    Keeps printable Unicode, but strips surrogates and control chars.

    Filters out emojis etc.
    """
    # Remove all surrogate characters and control codes except for newlines and tabs
    return ''.join(
        ch if (
                (unicodedata.category(ch)[0] != 'C' or ch in '\n\t')
                and (0 <= ord(ch) <= 0xD7FF or 0xE000 <= ord(ch) <= 0x10FFFF)
        )
        else '?'
        for ch in msg
    )


class SanitizingFormatter(logging.Formatter):
    """
    Formatter that sanitizes messages to strip or replace non-UTF-8 and problematic characters,
    including those found in exception info/tracebacks.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Format message first (handles substitutions)
        orig_msg = super().format(record)
        # Sanitize full output (including exception if present)
        return sanitize_log_message(orig_msg)

    def formatException(self, ei) -> str:
        exc = super().formatException(ei)
        return sanitize_log_message(exc)


def global_logger(log_name: str) -> logging.Logger:
    """
    Return a logger writing sanitized log messages and tracebacks.
    """
    logging_dir = get_global_config().log_dir
    logging_dir.mkdir(parents=True, exist_ok=True)
    log_path = logging_dir / f'{log_name}.log'

    log = logging.getLogger(log_name)
    log.setLevel(logging.INFO)

    if not log.hasHandlers():
        file_handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
        file_handler.setFormatter(
            SanitizingFormatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        log.addHandler(file_handler)

    return log
