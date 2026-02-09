import os
import cv2
import pandas as pd
import easyocr
import difflib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

print("Initializing EasyOCR...")
reader = easyocr.Reader(["ar", "en"], gpu=False)

CSV_PATH = "./Rdata/labels.csv"


def cleanup_text(text):
    if not isinstance(text, str):
        return str(text).upper()
    return "".join(e for e in text if e.isalnum()).upper()


def load_database(csv_path):
    try:
        if not os.path.exists(csv_path):
            print(f"ERROR: CSV file not found at {os.path.abspath(csv_path)}")
            return {}

        df = pd.read_csv(csv_path)
        df["plate_number"] = df["plate_number"].astype(str)

        col_img = "file_name"
        col_plate = "plate_number"

        if col_img not in df.columns or col_plate not in df.columns:
            return {}

        db_map = {}
        for index, row in df.iterrows():
            img_filename = str(row[col_img]).strip()
            raw_plate = str(row[col_plate])
            clean_plate = cleanup_text(raw_plate)
            db_map[clean_plate] = img_filename

        print(f"DEBUG: Loaded {len(db_map)} plates from database.")
        return db_map

    except Exception as e:
        print(f"Error loading CSV: {e}")
        return {}


def find_best_match(detected_text, database_keys):
    detected_clean = cleanup_text(detected_text)
    detected_reversed = detected_clean[::-1]

    if detected_clean in database_keys:
        return detected_clean, 1.0

    if detected_reversed in database_keys:
        return detected_reversed, 0.95

    best_match = None
    best_score = 0.0

    for db_key in database_keys:
        if abs(len(detected_clean) - len(db_key)) > 1:
            continue

        similarity = difflib.SequenceMatcher(None, detected_clean, db_key).ratio()

        if similarity > 0.85 and similarity > best_score:
            best_score = similarity
            best_match = db_key

    return best_match, best_score


PLATE_DATABASE = load_database(CSV_PATH)
DB_KEYS = list(PLATE_DATABASE.keys())


def process_image_from_memory(img):
    result = {
        "success": False,
        "matched": False,
        "matched_plate": None,
        "detected_plate": None,
        "confidence": 0.0,
        "message": "",
    }

    if img is None:
        result["message"] = "Failed to load image."
        return result

    result["success"] = True

    try:
        ocr_results = reader.readtext(img)

        best_conf = 0.0
        best_candidate = None

        for bbox, text, prob in ocr_results:
            prob = float(prob)
            if len(text) < 3:
                continue

            matched_key, match_score = find_best_match(text, DB_KEYS)

            if matched_key:
                result["matched"] = True
                result["matched_plate"] = matched_key
                result["detected_plate"] = text
                result["confidence"] = prob
                result["message"] = f"ACCESS GRANTED (Match: {matched_key})"
                return result

            if prob > best_conf:
                best_conf = prob
                best_candidate = cleanup_text(text)

        if best_candidate:
            result["detected_plate"] = best_candidate
            result["message"] = "ACCESS DENIED"
        else:
            result["message"] = "No Text Detected"

    except Exception as e:
        print(f"Backend Error: {e}")
        result["message"] = f"OCR Error: {e}"

    return result


@app.route("/scan", methods=["POST"])
def scan_plate():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    data = process_image_from_memory(img)

    return jsonify(data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
