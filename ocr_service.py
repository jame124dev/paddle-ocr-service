from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
import os
import cv2

app = Flask(__name__)

# Cache OCR instances per language
ocr_instances = {}

def get_ocr(lang='ch'):
    if lang not in ocr_instances:
        ocr_instances[lang] = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=False
        )
    return ocr_instances[lang]

@app.route('/ocr', methods=['POST'])
def run_ocr():
    try:
        lang = request.form.get('lang', 'ch')
        ocr = get_ocr(lang)

        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        temp_path = "temp.png"
        file.save(temp_path)

        # Ensure image loads
        img = cv2.imread(temp_path)
        if img is None:
            return jsonify({"error": "Image could not be read"})

        # Resize slightly (helps detection)
        img = cv2.resize(img, None, fx=1.5, fy=1.5)
        cv2.imwrite(temp_path, img)

        result = ocr.ocr(temp_path, cls=True)

        os.remove(temp_path)

        if result is None:
            return jsonify({"error": "OCR returned None"})

        texts = []

        # SAFE PARSER
        for line in result:
            if not line:
                continue

            for item in line:
                if isinstance(item, list) and len(item) == 2:
                    text = item[1][0]
                    texts.append(text)

        if not texts:
            return jsonify({"error": "No text detected"})

        return jsonify({
            "text": "\n".join(texts)
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)