"""
fetch_openalex.py — Queries the OpenAlex API for metadata and abstracts of
open-access electrical engineering papers.
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
    """Rebuild the abstract from OpenAlex's {word: [pos1, pos2, ...]} inverted index."""
    if not inverted_index:
        return ""
    max_pos = max(pos for positions in inverted_index.values() for pos in positions)
    words = [""] * (max_pos + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words)


def _fetch_by_concept(concept_id: str, filters: dict, select_fields: list, per_page: int, max_results: int) -> list[dict]:
    """Fetch works for a single concept_id."""
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
    Search OpenAlex using the config filters, iterating over the main
    concept_id plus any extra_concept_ids. Returns deduplicated works.
    """
    filters = config.get("filters", {})
    max_results = config.get("max_results", 500)
    per_page = config.get("per_page", 50)
    select_fields = config.get("select_fields", [])

    concept_ids = [filters["concept_id"]] + filters.get("extra_concept_ids", [])
    # Split max_results across concepts
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
        print(f"  → {len(batch)} fetched, {len(all_works)} unique so far")

    logger.info(f"Total fetched: {len(all_works)} unique works.")
    return all_works


def save_abstracts(works: list[dict], output_dir: Path | None = None) -> None:
    """Save the clean abstracts as text files (ground truth)."""
    output_dir = output_dir or GROUND_TRUTH_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for work in works:
        work_id = work.get("id", "unknown").split("/")[-1]
        abstract = work.get("abstract", "")
        if abstract:
            filepath = output_dir / f"{work_id}.txt"
            filepath.write_text(abstract, encoding="utf-8")
            logger.info(f"Saved abstract: {filepath}")


def save_works_json(works: list[dict], output_path: Path = Path("data/works.json")) -> None:
    """Save the full works list (including oa_url) as JSON for the pipeline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(works, f, ensure_ascii=False, indent=2)
    logger.info(f"Works saved to {output_path}")


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

    print(f"Fetched {len(works)} works ({n_with_abstract} with abstract).")
    print_summary()
