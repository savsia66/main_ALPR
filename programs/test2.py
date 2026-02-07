import os
import cv2
import pandas as pd
import easyocr
import difflib

print("Initializing EasyOCR...")
reader = easyocr.Reader(["ar", "en"], gpu=False)


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
        if abs(len(detected_clean) - len(db_key)) > 2:
            continue

        if db_key in detected_clean or db_key in detected_reversed:
            score = 0.9
            if score > best_score:
                best_score = score
                best_match = db_key

        similarity = difflib.SequenceMatcher(None, detected_clean, db_key).ratio()

        if similarity > 0.85 and similarity > best_score:
            best_score = similarity
            best_match = db_key

    return best_match, best_score


def check_access(image_path, csv_path, db_image_dir=None):
    result = {
        "success": False,
        "matched": False,
        "matched_plate": None,
        "detected_plate": None,
        "confidence": 0.0,
        "annotated_image": None,
        "message": "",
    }

    if not os.path.exists(image_path):
        result["message"] = "Image not found."
        return result

    img = cv2.imread(image_path)
    if img is None:
        result["message"] = "Failed to load image."
        return result

    result["success"] = True
    result["annotated_image"] = img.copy()

    plate_database = load_database(csv_path)
    db_keys = list(plate_database.keys())

    try:
        ocr_results = reader.readtext(img)

        best_candidate = None
        best_bbox = None
        best_conf = 0.0

        for bbox, text, prob in ocr_results:
            prob = float(prob)

            if len(text) < 3:
                continue

            matched_key, match_score = find_best_match(text, db_keys)

            if matched_key:
                result["matched"] = True
                result["matched_plate"] = matched_key
                result["detected_plate"] = text
                result["confidence"] = prob

                (tl, tr, br, bl) = bbox
                cv2.rectangle(
                    result["annotated_image"],
                    (int(tl[0]), int(tl[1])),
                    (int(br[0]), int(br[1])),
                    (0, 255, 0),
                    5,
                )

                filename_from_csv = plate_database[matched_key]
                if db_image_dir and filename_from_csv:
                    full_db_path = os.path.join(db_image_dir, filename_from_csv)
                    result["db_image_path"] = (
                        full_db_path if os.path.exists(full_db_path) else None
                    )

                result["message"] = f"ACCESS GRANTED (Match: {matched_key})"
                return result

            if prob > best_conf:
                best_conf = prob
                best_candidate = cleanup_text(text)
                best_bbox = bbox

        if best_candidate:
            result["detected_plate"] = best_candidate
            result["message"] = "ACCESS DENIED"
            if best_bbox:
                (tl, tr, br, bl) = best_bbox
                cv2.rectangle(
                    result["annotated_image"],
                    (int(tl[0]), int(tl[1])),
                    (int(br[0]), int(br[1])),
                    (0, 0, 255),
                    5,
                )
        else:
            result["message"] = "No Plate Detected"

    except Exception as e:
        print(f"Backend Error: {e}")
        result["message"] = f"OCR Error: {e}"

    return result
