"""
logger.py — Configuración estándar de logging para el proyecto.
"""

import logging
import sys


def setup_logger(
    level: int = logging.INFO,
    log_format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
) -> None:
    """
    Configura el logger raíz del proyecto.

    Args:
        level: Nivel de logging (default: INFO).
        log_format: Formato del mensaje de log.
    """
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
