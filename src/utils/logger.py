"""
logger.py — Standard logging configuration for the project.
"""

import logging
import sys


def setup_logger(
    level: int = logging.INFO,
    log_format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
) -> None:
    """Configure the project root logger."""
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
