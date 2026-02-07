import os
import cv2
import pandas as pd
import easyocr

print("Initializing EasyOCR...")
reader = easyocr.Reader(["ar"], gpu=False)


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
        print(f"DEBUG: CSV Columns found: {df.columns.tolist()}")

        col_img = "file_name"
        col_plate = "plate_number"

        if col_img not in df.columns or col_plate not in df.columns:
            print(f"ERROR: CSV missing required columns. Found: {df.columns.tolist()}")
            print(f"       Expected: ['{col_img}', '{col_plate}']")
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


def check_access(image_path, csv_path, db_image_dir=None):
    result = {
        "success": False,
        "matched": False,
        "matched_plate": None,
        "detected_plate": None,
        "confidence": 0.0,
        "annotated_image": None,
        "db_image_path": None,
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

    try:
        ocr_results = reader.readtext(img)

        best_candidate = None
        best_bbox = None
        best_conf = 0.0

        for bbox, text, prob in ocr_results:
            clean_plate = cleanup_text(text)
            prob = float(prob)

            if len(clean_plate) > 3:
                if clean_plate in plate_database:
                    result["matched"] = True
                    result["matched_plate"] = clean_plate
                    result["detected_plate"] = clean_plate
                    result["confidence"] = prob

                    (tl, tr, br, bl) = bbox
                    cv2.rectangle(
                        result["annotated_image"],
                        (int(tl[0]), int(tl[1])),
                        (int(br[0]), int(br[1])),
                        (0, 255, 0),
                        5,
                    )

                    filename_from_csv = plate_database[clean_plate]

                    if db_image_dir is not None:
                        full_db_path = os.path.join(db_image_dir, filename_from_csv)

                        if os.path.exists(full_db_path):
                            result["db_image_path"] = full_db_path
                            print(f"DEBUG: Found DB Image at {full_db_path}")
                        else:
                            print(f"WARNING: DB Image missing at {full_db_path}")
                            result["db_image_path"] = None
                    else:
                        result["db_image_path"] = None

                    result["message"] = "ACCESS GRANTED"
                    return result

                if prob > best_conf:
                    best_conf = prob
                    best_candidate = clean_plate
                    best_bbox = bbox

        if best_candidate:
            result["detected_plate"] = best_candidate
            result["message"] = "ACCESS DENIED"

            if best_bbox is not None:
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
