"""
logger.py — Centralised logging setup for the entire AD pipeline.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = "ad_pipeline", log_level: str = "INFO") -> logging.Logger:
    """
    Create and configure the root pipeline logger.
    Writes to both console (coloured) and a rotating log file.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured — return existing instance
        return logger

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # ── Console handler ────────────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(_ColourFormatter())
    logger.addHandler(console)

    # ── File handler ───────────────────────────────────────────────────────
    log_dir = Path("output/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_dir / f"pipeline_{timestamp}.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(file_handler)

    return logger


class _ColourFormatter(logging.Formatter):
    """Add ANSI colour codes to console log levels."""
    GREY    = "\x1b[38;5;240m"
    CYAN    = "\x1b[36m"
    YELLOW  = "\x1b[33m"
    RED     = "\x1b[31m"
    BOLD_RED = "\x1b[1;31m"
    RESET   = "\x1b[0m"

    FORMATS = {
        logging.DEBUG:    GREY    + "%(asctime)s [DEBUG]   %(message)s" + RESET,
        logging.INFO:     CYAN    + "%(asctime)s [INFO]    %(message)s" + RESET,
        logging.WARNING:  YELLOW  + "%(asctime)s [WARNING] %(message)s" + RESET,
        logging.ERROR:    RED     + "%(asctime)s [ERROR]   %(message)s" + RESET,
        logging.CRITICAL: BOLD_RED + "%(asctime)s [CRITICAL] %(message)s" + RESET,
    }

    def format(self, record: logging.LogRecord) -> str:
        fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


# Module-level convenience logger
log = setup_logger()
