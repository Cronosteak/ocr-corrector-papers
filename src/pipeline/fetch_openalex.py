"""
fetch_openalex.py — Consulta OpenAlex API para obtener metadata y abstracts
de papers de ingeniería eléctrica con acceso abierto.
"""

import os
import json
import logging
from pathlib import Path

import pyalex
from pyalex import Works
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "")
GROUND_TRUTH_DIR = Path(os.getenv("GROUND_TRUTH_DIR", "data/ground_truth"))

if OPENALEX_EMAIL:
    pyalex.config.email = OPENALEX_EMAIL


def reconstruct_abstract(inverted_index: dict) -> str:
    """
    Reconstruye el abstract desde el formato abstract_inverted_index de OpenAlex.

    El formato es {palabra: [posicion1, posicion2, ...]}.
    """
    if not inverted_index:
        return ""
    max_pos = max(pos for positions in inverted_index.values() for pos in positions)
    words = [""] * (max_pos + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words)


def _fetch_by_concept(concept_id: str, filters: dict, select_fields: list, per_page: int, max_results: int) -> list[dict]:
    """Descarga works para un concept_id específico."""
    query = Works().filter(
        concepts={"id": concept_id},
        is_oa=filters.get("is_oa", True),
        language=filters.get("language", "en"),
        from_publication_date=filters.get("from_publication_date"),
        to_publication_date=filters.get("to_publication_date"),
    )
    if select_fields:
        query = query.select(select_fields)

    works = []
    for page in query.paginate(per_page=per_page, n_max=max_results):
        for work in page:
            inverted_index = work.get("abstract_inverted_index") or {}
            abstract = reconstruct_abstract(inverted_index)
            oa_url = (work.get("open_access") or {}).get("oa_url", "")
            works.append({
                "id": work.get("id", ""),
                "doi": work.get("doi", ""),
                "title": work.get("title", ""),
                "abstract": abstract,
                "oa_url": oa_url,
                "publication_date": work.get("publication_date", ""),
                "language": work.get("language", ""),
            })
    return works


def fetch_works(config: dict) -> list[dict]:
    """
    Busca trabajos en OpenAlex según los filtros del config.
    Itera sobre concept_id principal y extra_concept_ids si están definidos.

    Args:
        config: Diccionario con filtros (concept, year_range, language, per_page, etc.)

    Returns:
        Lista de diccionarios con metadata de cada trabajo (sin duplicados).
    """
    filters = config.get("filters", {})
    max_results = config.get("max_results", 500)
    per_page = config.get("per_page", 50)
    select_fields = config.get("select_fields", [])

    concept_ids = [filters["concept_id"]] + filters.get("extra_concept_ids", [])
    # Repartir max_results entre conceptos
    per_concept = max_results // len(concept_ids)

    seen_ids = set()
    all_works = []

    for concept_id in concept_ids:
        print(f"Fetching concept {concept_id} (max {per_concept})...")
        batch = _fetch_by_concept(concept_id, filters, select_fields, per_page, per_concept)
        for work in batch:
            if work["id"] not in seen_ids:
                seen_ids.add(work["id"])
                all_works.append(work)
        print(f"  → {len(batch)} obtenidos, {len(all_works)} únicos acumulados")

    logger.info(f"Total obtenidos: {len(all_works)} trabajos únicos.")
    return all_works


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


def save_works_json(works: list[dict], output_path: Path = Path("data/works.json")) -> None:
    """Guarda la lista completa de works (con oa_url) en un JSON para uso del pipeline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(works, f, ensure_ascii=False, indent=2)
    logger.info(f"Works guardados en {output_path}")


if __name__ == "__main__":
    import yaml
    from src.utils.pipeline_stats import StepTimer, print_summary

    with open("configs/openalex_query.yaml", "r") as f:
        config = yaml.safe_load(f)

    with StepTimer("1_fetch_openalex") as t:
        works = fetch_works(config)
        save_abstracts(works)
        save_works_json(works)
        n_with_abstract = sum(1 for w in works if w.get("abstract"))
        t.record("n_works_total", len(works))
        t.record("n_with_abstract", n_with_abstract)
        t.record("n_with_oa_url", sum(1 for w in works if w.get("oa_url")))

    print(f"Se obtuvieron {len(works)} trabajos ({n_with_abstract} con abstract).")
    print_summary()
