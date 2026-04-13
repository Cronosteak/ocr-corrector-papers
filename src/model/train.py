"""
train.py — Fine-tune de un modelo seq2seq (T5/BART) sobre pares OCR → texto limpio.
"""

import json
import logging
import os
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_split(path: Path) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    return Dataset.from_list(pairs)


def _build_tokenize_fn(tokenizer, prefix: str, max_input: int, max_output: int):
    def tokenize(batch):
        inputs = [prefix + text for text in batch["ocr"]]
        model_inputs = tokenizer(
            inputs,
            max_length=max_input,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=batch["ground_truth"],
            max_length=max_output,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return tokenize


def train(config_path: str = "configs/train_config.yaml") -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model_name"]
    prefix = config.get("prefix", "correct OCR: ")
    max_input = config.get("max_input_length", 512)
    max_output = config.get("max_output_length", 512)
    batch_size = config.get("batch_size", 8)
    num_epochs = config.get("num_epochs", 3)
    learning_rate = config.get("learning_rate", 5e-5)
    warmup_steps = config.get("warmup_steps", 100)
    weight_decay = config.get("weight_decay", 0.01)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    output_dir = config.get("output_dir", "models/ocr-corrector")
    seed = config.get("seed", 42)
    save_strategy = config.get("save_strategy", "epoch")
    eval_strategy = config.get("evaluation_strategy", "epoch")
    load_best_model_at_end = config.get("load_best_model_at_end", True)
    metric_for_best_model = config.get("metric_for_best_model", "eval_loss")
    fp16 = config.get("fp16", False) and torch.cuda.is_available()
    dataloader_num_workers = config.get("dataloader_num_workers", 2)
    max_grad_norm = config.get("max_grad_norm", 1.0)
    optim = config.get("optim", "adafactor")

    logger.info(f"Modelo: {model_name} | GPU: {torch.cuda.is_available()} | fp16: {fp16} | optim: {optim}")

    # Cargar tokenizer y modelo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Cargar splits pre-generados
    pairs_dir = Path("data/pairs")
    train_ds = _load_split(pairs_dir / "train.json")
    val_ds = _load_split(pairs_dir / "val.json")
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Tokenizar
    tokenize_fn = _build_tokenize_fn(tokenizer, prefix, max_input, max_output)
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=val_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=fp16,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        predict_with_generate=False,
        logging_steps=50,
        seed=seed,
        dataloader_num_workers=dataloader_num_workers,
        max_grad_norm=max_grad_norm,
        optim=optim,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Iniciando entrenamiento...")
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Guardar historial de pérdidas para gráficos
    history_path = Path(output_dir) / "train_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    logger.info(f"Historial guardado en {history_path}")

    logger.info(f"Modelo guardado en {output_dir}")


if __name__ == "__main__":
    train()
