"""
train.py — Fine-tune de un modelo seq2seq (T5/BART) sobre pares OCR → texto limpio.
"""

import json
import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def load_dataset(pairs_path: str = "data/pairs/dataset.json") -> dict:
    """
    Carga el dataset de pares OCR ↔ ground truth.

    Returns:
        Diccionario con splits train/val/test.
    """
    with open(pairs_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    # TODO: Implementar split train/val/test
    # Usar sklearn.model_selection.train_test_split o datasets.Dataset.train_test_split
    raise NotImplementedError("Implementar carga y split del dataset")


def train(config_path: str = "configs/train_config.yaml") -> None:
    """
    Entrena el modelo de corrección OCR.

    Args:
        config_path: Ruta al archivo de configuración de entrenamiento.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = config.get("model_name", "google/flan-t5-base")
    batch_size = config.get("batch_size", 8)
    num_epochs = config.get("num_epochs", 3)
    learning_rate = config.get("learning_rate", 5e-5)

    logger.info(f"Modelo: {model_name}")
    logger.info(f"Batch size: {batch_size}, Epochs: {num_epochs}, LR: {learning_rate}")

    # TODO: Implementar fine-tuning con HuggingFace Transformers
    # 1. Cargar tokenizer y modelo (AutoTokenizer, AutoModelForSeq2SeqLM)
    # 2. Tokenizar el dataset (prefix: "correct OCR: ")
    # 3. Configurar TrainingArguments
    # 4. Crear Trainer y entrenar
    # 5. Guardar modelo en MODEL_PATH
    raise NotImplementedError("Implementar entrenamiento del modelo")


if __name__ == "__main__":
    train()
