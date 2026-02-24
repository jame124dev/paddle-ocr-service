# Chinese OCR Server (PaddleOCR)

A production-ready OCR API for Simplified Chinese text extraction.

## Requirements

- **Python 3.11** (Do NOT use 3.12+)
- 4GB+ RAM
- Internet (first run downloads models)

## Tech Stack

| Package      | Version  |
|--------------|----------|
| Python       | 3.11     |
| paddlepaddle | 2.6.2    |
| paddleocr    | 2.7.3    |
| numpy        | 1.24.4   |
| scipy        | 1.10.1   |
| opencv       | 4.6.0.66 |
| flask        | latest   |

## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv311

# Windows
venv311\Scripts\activate

# Linux/Mac
source venv311/bin/activate
```

### 2. Install Dependencies (Order Matters)

```bash
pip install numpy==1.24.4
pip install paddlepaddle==2.6.2
pip install paddleocr==2.7.3
pip install opencv-python==4.6.0.66
pip install scipy==1.10.1
pip install flask
```

## Project Structure

```
ocr/
├── ocr_service.py    # Flask API server
├── README.md
├── .gitignore
└── venv311/          # Virtual environment
```

## Usage

### Start Server

```bash
python ocr_service.py
```

Server runs at: `http://127.0.0.1:5004`

### API Endpoint

```
POST /ocr
Content-Type: multipart/form-data
```

| Key   | Type | Description       |
|-------|------|-------------------|
| image | File | Image file to OCR |

### Example Request

```bash
curl -X POST http://127.0.0.1:5004/ocr -F "image=@test.png"
```

### Response

```json
{
  "text": "Extracted text here..."
}
```

## First Run

Models download automatically on first run (~1-2 minutes).  
Stored in: `~/.paddleocr/`

## Production Deployment

```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:5004 ocr_service:app
```

## Troubleshooting

### NumPy ABI Error

```bash
pip uninstall numpy -y
pip install numpy==1.24.4
```

### SciPy Error

```bash
pip install scipy==1.10.1
```

### OCR Returns None

- Ensure image is clear
- Use dark text on light background
- Increase image resolution
