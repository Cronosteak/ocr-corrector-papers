# OCR Corrector Papers

Corrector automático de texto OCR en papers académicos de ingeniería eléctrica, usando modelos seq2seq (T5/BART) fine-tuneados con datos de OpenAlex.

## Descripción

Este proyecto construye un pipeline end-to-end para:

1. **Recolectar** papers de ingeniería eléctrica desde OpenAlex API
2. **Extraer** texto ruidoso con Tesseract OCR
3. **Alinear** texto OCR con el texto limpio (ground truth) de OpenAlex
4. **Entrenar** un modelo seq2seq para corregir errores OCR
5. **Evaluar** mejoras usando métricas CER/WER

## Estructura del Proyecto

```
ocr-corrector-papers/
├── README.md
├── data/
│   ├── raw/                    # PDFs descargados de OpenAlex
│   ├── ocr/                    # Texto extraído por Tesseract (ruidoso)
│   ├── ground_truth/           # Texto limpio obtenido de OpenAlex API
│   └── pairs/                  # Pares alineados (ruidoso ↔ limpio)
├── src/
│   ├── pipeline/               # Pipeline de datos
│   ├── model/                  # Entrenamiento y evaluación del modelo
│   ├── utils/                  # Utilidades compartidas
│   └── api/                    # (Opcional) Servicio REST
├── notebooks/                  # Jupyter notebooks de exploración
├── configs/                    # Configuración YAML
├── tests/                      # Tests unitarios
├── requirements.txt
└── .env.example
```

## Flujo de Datos

```
fetch_openalex.py → metadata + abstract URLs
      ↓
download_pdfs.py → archivos .pdf
      ↓
ocr_extract.py → texto OCR ruidoso
      ↓
align_text.py → pares alineados
      ↓
build_dataset.py → data/pairs/*.json
      ↓
train.py → modelo fine-tuned
      ↓
evaluate.py → métricas finales (CER/WER)
```

## Instalación

```bash
pip install -r requirements.txt
```

Copia `.env.example` a `.env` y configura las variables:

```bash
cp .env.example .env
```

## Uso Rápido

```bash
# 1. Construir el dataset
python -m src.pipeline.build_dataset

# 2. Entrenar el modelo
python -m src.model.train

# 3. Evaluar resultados
python -m src.model.evaluate

# 4. Corregir nuevo texto OCR
python -m src.model.predict --input "texto ruidoso aquí"
```

## Configuración

- `configs/openalex_query.yaml` — Filtros de búsqueda en OpenAlex
- `configs/train_config.yaml` — Hiperparámetros del modelo

## Tests

```bash
pytest tests/
```
