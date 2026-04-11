"""
download_pdfs.py — Descarga PDFs de acceso abierto desde las URLs
proporcionadas por OpenAlex.
"""

import os
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
        output_path.write_bytes(response.content)
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
        pdf_url = work.get("open_access", {}).get("oa_url")

        if not pdf_url:
            stats["failed"] += 1
            continue

        output_path = output_dir / f"{work_id}.pdf"
        if output_path.exists():
            stats["success"] += 1
            continue

        if download_pdf(pdf_url, output_path):
            stats["success"] += 1
        else:
            stats["failed"] += 1

    return stats


if __name__ == "__main__":
    # TODO: Cargar works desde JSON o ejecutar fetch_openalex primero
    print("Usa build_dataset.py para orquestar el pipeline completo.")
