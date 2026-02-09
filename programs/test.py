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
        if abs(len(detected_clean) - len(db_key)) > 1:
            continue

        similarity = difflib.SequenceMatcher(None, detected_clean, db_key).ratio()

        if similarity > 0.85 and similarity > best_score:
            best_score = similarity
            best_match = db_key

    return best_match, best_score


def get_yolo_crops(image, txt_path):
    crops = []
    if not os.path.exists(txt_path):
        return crops

    h_img, w_img = image.shape[:2]

    try:
        with open(txt_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            x_c_px = x_center * w_img
            y_c_px = y_center * h_img
            w_px = width * w_img
            h_px = height * h_img

            x_min = int(max(0, x_c_px - (w_px / 2)))
            y_min = int(max(0, y_c_px - (h_px / 2)))
            x_max = int(min(w_img, x_c_px + (w_px / 2)))
            y_max = int(min(h_img, y_c_px + (h_px / 2)))

            if x_max > x_min and y_max > y_min:
                crop = image[y_min:y_max, x_min:x_max]
                crops.append((crop, (x_min, y_min, x_max, y_max)))
    except Exception as e:
        print(f"Warning: Failed to parse YOLO file {txt_path}: {e}")

    return crops


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
    db_keys = list(plate_database.keys())

    txt_path = os.path.splitext(image_path)[0] + ".txt"
    crops_data = get_yolo_crops(img, txt_path)

    best_candidate = None
    best_conf = 0.0
    best_box = None

    try:
        if crops_data:
            print("DEBUG: YOLO coordinates found. Using crop-based detection.")
            for crop, (x1, y1, x2, y2) in crops_data:
                ocr_results = reader.readtext(crop)

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

                        cv2.rectangle(
                            result["annotated_image"],
                            (x1, y1),
                            (x2, y2),
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
                        best_box = (x1, y1, x2, y2)

        else:
            print("DEBUG: No YOLO coordinates found. Scanning full image with EasyOCR.")
            ocr_results = reader.readtext(img)

            for bbox, text, prob in ocr_results:
                prob = float(prob)
                if len(text) < 3:
                    continue

                (tl, tr, br, bl) = bbox
                x_min = int(min(tl[0], bl[0]))
                x_max = int(max(tr[0], br[0]))
                y_min = int(min(tl[1], tr[1]))
                y_max = int(max(bl[1], br[1]))

                curr_box = (x_min, y_min, x_max, y_max)

                matched_key, match_score = find_best_match(text, db_keys)
                if matched_key:
                    result["matched"] = True
                    result["matched_plate"] = matched_key
                    result["detected_plate"] = text
                    result["confidence"] = prob

                    cv2.rectangle(
                        result["annotated_image"],
                        (curr_box[0], curr_box[1]),
                        (curr_box[2], curr_box[3]),
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
                    best_box = curr_box

        if best_candidate:
            result["detected_plate"] = best_candidate
            result["message"] = "ACCESS DENIED"
            if best_box:
                cv2.rectangle(
                    result["annotated_image"],
                    (best_box[0], best_box[1]),
                    (best_box[2], best_box[3]),
                    (0, 0, 255),
                    5,
                )
        else:
            result["message"] = "No Text Detected"

    except Exception as e:
        print(f"Backend Error: {e}")
        result["message"] = f"OCR Error: {e}"

    return result
