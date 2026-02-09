import os
import cv2
import pandas as pd
import easyocr
import difflib
import numpy as np
import torch
import pathlib
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- CONFIGURATION ---
# Path to your trained YOLO model (best.pt) or the pre-trained weights
MODEL_PATH = "./plate_model.pt"
CSV_PATH = "./Rdata/labels.csv"

app = Flask(__name__)
CORS(app)

# 1. LOAD OCR READER
print("Initializing EasyOCR...")
reader = easyocr.Reader(["ar", "en"], gpu=False)

# 2. LOAD YOLO MODEL (The "Working Code" Logic) [cite: 7, 8]
print(f"Loading YOLOv9 model from {MODEL_PATH}...")
try:
    # We use torch.hub to load the custom model exactly as your file did
    # Note: 'source="github"' requires internet first time to download repo structure
    model = torch.hub.load(
        "WongKinYiu/yolov9", "custom", path=MODEL_PATH, force_reload=False
    )
    # Set confidence threshold to reduce garbage detections
    model.conf = 0.5
except Exception as e:
    print(
        f"CRITICAL ERROR: Could not load YOLO model. Ensure '{MODEL_PATH}' exists. Error: {e}"
    )
    model = None


def cleanup_text(text):
    if not isinstance(text, str):
        return str(text).upper()
    return "".join(e for e in text if e.isalnum()).upper()


def load_database(csv_path):
    try:
        if not os.path.exists(csv_path):
            return {}
        df = pd.read_csv(csv_path)
        col_img = "file_name"
        col_plate = "plate_number"
        if col_img not in df.columns or col_plate not in df.columns:
            return {}
        db_map = {}
        for index, row in df.iterrows():
            img_filename = str(row[col_img]).strip()
            clean_plate = cleanup_text(str(row[col_plate]))
            db_map[clean_plate] = img_filename
        print(f"DEBUG: Loaded {len(db_map)} plates.")
        return db_map
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return {}


PLATE_DATABASE = load_database(CSV_PATH)
DB_KEYS = list(PLATE_DATABASE.keys())


def find_best_match(detected_text, database_keys):
    detected_clean = cleanup_text(detected_text)
    if detected_clean in database_keys:
        return detected_clean, 1.0

    # Reverse check
    if detected_clean[::-1] in database_keys:
        return detected_clean[::-1], 0.95

    best_match = None
    best_score = 0.0
    for db_key in database_keys:
        if abs(len(detected_clean) - len(db_key)) > 2:
            continue
        similarity = difflib.SequenceMatcher(None, detected_clean, db_key).ratio()
        if similarity > 0.85 and similarity > best_score:
            best_score = similarity
            best_match = db_key
    return best_match, best_score


# --- NEW LOGIC: YOLO DETECTION & CROP ---
def detect_and_crop(img):
    """
    Uses YOLO to find the plate and returns the cropped image.
    Based on working code[cite: 8, 9].
    """
    if model is None:
        return []

    # Inference
    # YOLOv9 expects BGR or RGB. cv2 is BGR.
    # We convert to RGB as seen in your working code [cite: 8]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model(img_rgb)

    # Parse results [cite: 8]
    # .xyxy[0] returns tensor of detections: [x1, y1, x2, y2, confidence, class]
    detections = results.xyxy[0].cpu().numpy()

    crops = []

    for box in detections:
        x1, y1, x2, y2, conf, cls = box

        # Convert to int for slicing
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Crop logic
        # Ensure coordinates are within image bounds
        h, w, _ = img.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Crop the BGR image (original)
        plate_crop = img[y1:y2, x1:x2]

        if plate_crop.size > 0:
            crops.append((plate_crop, conf))

    return crops


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
        # STEP 1: Detect Plate using YOLO
        crops = detect_and_crop(img)

        if not crops:
            result["message"] = "No License Plate Detected by YOLO."
            return result

        # Sort crops by confidence (highest first)
        crops.sort(key=lambda x: x[1], reverse=True)

        best_candidate_text = None
        best_candidate_conf = 0.0

        # STEP 2: OCR only the cropped areas
        for plate_img, detect_conf in crops:

            # Optional: Preprocess the crop slightly (grayscale) for EasyOCR
            gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

            # Read text
            ocr_results = reader.readtext(gray_plate)

            for bbox, text, ocr_prob in ocr_results:
                clean_txt = cleanup_text(text)

                if len(clean_txt) < 3:
                    continue

                # Match against DB
                matched_key, match_score = find_best_match(clean_txt, DB_KEYS)

                if matched_key:
                    result["matched"] = True
                    result["matched_plate"] = matched_key
                    result["detected_plate"] = clean_txt
                    result["confidence"] = float(ocr_prob)
                    result["message"] = f"ACCESS GRANTED (Match: {matched_key})"
                    return result

                # Track best non-match
                if ocr_prob > best_candidate_conf:
                    best_candidate_conf = float(ocr_prob)
                    best_candidate_text = clean_txt

        # If no strict match found
        if best_candidate_text:
            # Try fuzzy match one last time
            matched_key, match_score = find_best_match(best_candidate_text, DB_KEYS)
            if matched_key:
                result["matched"] = True
                result["matched_plate"] = matched_key
                result["detected_plate"] = best_candidate_text
                result["confidence"] = best_candidate_conf
                result["message"] = f"ACCESS GRANTED (Match: {matched_key})"
            else:
                result["detected_plate"] = best_candidate_text
                result["message"] = "ACCESS DENIED"
        else:
            result["message"] = "Plate Detected, but OCR failed to read text."

    except Exception as e:
        print(f"Backend Error: {e}")
        result["message"] = f"Error: {e}"

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
