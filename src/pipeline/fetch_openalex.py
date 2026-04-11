"""
fetch_openalex.py — Consulta OpenAlex API para obtener metadata y abstracts
de papers de ingeniería eléctrica con acceso abierto.
"""

import os
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "")
GROUND_TRUTH_DIR = Path(os.getenv("GROUND_TRUTH_DIR", "data/ground_truth"))


def fetch_works(config: dict) -> list[dict]:
    """
    Busca trabajos en OpenAlex según los filtros del config.

    Args:
        config: Diccionario con filtros (concept, year_range, language, per_page, etc.)

    Returns:
        Lista de diccionarios con metadata de cada trabajo.
    """
    # TODO: Implementar llamada a OpenAlex API usando pyalex
    # Filtrar por: concept_id, is_oa=True, language, publication_year
    raise NotImplementedError("Implementar consulta a OpenAlex API")


def save_abstracts(works: list[dict], output_dir: Path | None = None) -> None:
    """
    Guarda los abstracts limpios como archivos de texto (ground truth).

    Args:
        works: Lista de trabajos obtenidos de OpenAlex.
        output_dir: Directorio donde guardar los archivos.
    """
    output_dir = output_dir or GROUND_TRUTH_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for work in works:
        work_id = work.get("id", "unknown").split("/")[-1]
        abstract = work.get("abstract", "")
        if abstract:
            filepath = output_dir / f"{work_id}.txt"
            filepath.write_text(abstract, encoding="utf-8")
            logger.info(f"Guardado abstract: {filepath}")


if __name__ == "__main__":
    import yaml

    with open("configs/openalex_query.yaml", "r") as f:
        config = yaml.safe_load(f)

    works = fetch_works(config)
    save_abstracts(works)
    print(f"Se obtuvieron {len(works)} trabajos.")
