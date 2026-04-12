"""
download_pdfs.py — Descarga PDFs de acceso abierto desde las URLs
proporcionadas por OpenAlex.
"""

import os
import time
import random
import logging
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
logger = logging.getLogger(__name__)

RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))


def download_pdf(url: str, output_path: Path, timeout: int = 30) -> bool:
    """
    Descarga un PDF desde una URL.

    Args:
        url: URL del PDF.
        output_path: Ruta donde guardar el archivo.
        timeout: Timeout en segundos para la descarga.

    Returns:
        True si la descarga fue exitosa, False en caso contrario.
    """
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        content = response.content
        if not content.startswith(b"%PDF"):
            logger.warning(f"Respuesta no es un PDF válido (URL: {url})")
            return False
        output_path.write_bytes(content)
        logger.info(f"Descargado: {output_path}")
        return True
    except requests.RequestException as e:
        logger.warning(f"Error descargando {url}: {e}")
        return False


def download_all_pdfs(works: list[dict], output_dir: Path | None = None) -> dict:
    """
    Descarga todos los PDFs de una lista de trabajos.

    Args:
        works: Lista de trabajos con campo 'open_access.oa_url'.
        output_dir: Directorio de salida.

    Returns:
        Diccionario con estadísticas de descarga (exitosos, fallidos).
    """
    output_dir = output_dir or RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"success": 0, "failed": 0}

    for work in tqdm(works, desc="Descargando PDFs"):
        work_id = work.get("id", "unknown").split("/")[-1]
        pdf_url = work.get("oa_url") or work.get("open_access", {}).get("oa_url")

        if not pdf_url:
            stats["failed"] += 1
            continue

        output_path = output_dir / f"{work_id}.pdf"
        if output_path.exists():
            stats["success"] += 1
            continue

        if download_pdf(pdf_url, output_path):
            stats["success"] += 1
            # Delay aleatorio para evitar rate limiting
            time.sleep(random.uniform(1.0, 3.0))
        else:
            stats["failed"] += 1

    return stats


if __name__ == "__main__":
    import json
    import sys

    works_path = Path("data/works.json")
    if not works_path.exists():
        print("ERROR: data/works.json no encontrado. Ejecuta primero fetch_openalex.py")
        sys.exit(1)

    with open(works_path, encoding="utf-8") as f:
        works = json.load(f)

    # Testear con 10 papers primero
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    gt_dir = Path("data/ground_truth")
    works_to_download = [
        w for w in works
        if w.get("oa_url")
        and (gt_dir / f"{w['id'].split('/')[-1]}.txt").exists()
    ]
    if limit:
        works_to_download = works_to_download[:limit]
    print(f"Descargando {len(works_to_download)} PDFs con abstract (de {len(works)} totales)...")

    from src.utils.pipeline_stats import StepTimer, print_summary
    with StepTimer("2_download_pdfs") as t:
        stats = download_all_pdfs(works_to_download)
        t.record("n_attempted", len(works_to_download))
        t.record("n_success", stats["success"])
        t.record("n_failed", stats["failed"])
        t.record("success_rate_pct", round(stats["success"] / len(works_to_download) * 100, 1) if works_to_download else 0)

    print(f"Exitosos: {stats['success']} | Fallidos: {stats['failed']}")
    print_summary()
