# Chinese Handwriting OCR API (Flask + PaddleOCR)

This API is tuned for Chinese handwriting recognition and returns ranked alternatives.

## Recommended stack

- Python 3.11
- paddleocr 3.x
- paddlepaddle 3.x

See `requirements.txt` for version ranges.

## Install

```bash
python -m venv venv311

# Windows
venv311\Scripts\activate

# Linux/Mac
source venv311/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python ocr_service.py
```

Server:

- `http://127.0.0.1:5006`

## OCR endpoint

- `POST /ocr`
- `Content-Type: multipart/form-data`

Fields:

- `image` (file, required)
- `lang` (text, optional, default `ch`)
- `mode` (text, optional, use `handwriting` for handwriting tuning)
- `single_char` (text, optional: `true`/`false`)
- `expected` (text, optional: expected answer like `它`)
- `allowed_chars` (text, optional: e.g. `它二一` or `它,二,一`)
- `top_k` (text/int, optional, default `5`)
- `debug` (text, optional: `true` to include engine/version metadata)

Notes:

- If `lang=ch` and `mode=handwriting`, `single_char` defaults to `true`.
- Response includes `best_text`, `best_score`, and `alternatives`.
- `expected` boosts final pick if present in candidates.
- `allowed_chars` filters candidates before final ranking.

## Example (single Chinese character)

```bash
curl -X POST http://127.0.0.1:5006/ocr \\
  -F "image=@image.png" \\
  -F "lang=ch" \\
  -F "mode=handwriting" \\
  -F "single_char=true"
```

## Optional model env vars

For custom handwriting models:

- `CH_HAND_DET_DIR`
- `CH_HAND_REC_DIR`
- `CH_HAND_CLS_DIR`

Model-name overrides (PaddleOCR 3.x):

- `CH_HAND_OCR_VERSION` (default `PP-OCRv5`)
- `CH_HAND_TEXT_DET_MODEL_NAME` (default `PP-OCRv5_server_det`)
- `CH_HAND_TEXT_REC_MODEL_NAME` (default `PP-OCRv5_server_rec`)

Detector tuning:

- `CH_HAND_DET_DB_THRESH` (default `0.2`)
- `CH_HAND_DET_DB_BOX_THRESH` (default `0.35`)
- `CH_HAND_DET_DB_UNCLIP_RATIO` (default `1.8`)

Runtime:

- `OCR_USE_GPU=true|false` (default `false`)

## Troubleshooting

- If you get old results after edits, kill old Python processes and restart.
- If recognition is still weak on your handwriting style, use a custom-trained Chinese handwriting recognition model and set `CH_HAND_REC_DIR`.
