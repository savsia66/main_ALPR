import os
import cv2
import pandas as pd
import test


IMAGE_FOLDER = "./Rdata/raw_data"
CSV_PATH = "./Rdata/labels.csv"


def run_test():

    df = pd.read_csv(CSV_PATH)
    ground_truth = {}

    for _, row in df.iterrows():
        fname = str(row["file_name"]).strip()
        plate = str(row["plate_number"])
        ground_truth[fname] = test.cleanup_text(plate)

    total = 0
    matches = 0

    print(f"{'IMAGE':<20} | {'REAL (CSV)':<15} | {'DETECTED':<15} | {'RESULT'}")
    print("-" * 70)

    for image_file in os.listdir(IMAGE_FOLDER):

        if image_file not in ground_truth:
            continue

        total += 1
        real_plate = ground_truth[image_file]

        img_path = os.path.join(IMAGE_FOLDER, image_file)
        img = cv2.imread(img_path)

        result = test.process_image_from_memory(img)

        if result["matched"]:
            detected_plate = result["matched_plate"]
        else:
            detected_plate = (
                str(result["detected_plate"]) if result["detected_plate"] else "NONE"
            )

        success = result["matched"] and detected_plate == real_plate

        if success:
            matches += 1
            status = "MATCH ✅"
        else:
            status = "FAIL ❌"

        print(f"{image_file:<20} | {real_plate:<15} | {detected_plate:<15} | {status}")

    print("-" * 70)
    if total > 0:
        accuracy = (matches / total) * 100
        print(f"Total: {total} | Success: {matches} | Fail: {total - matches}")
        print(f"Overall Success Rate: {accuracy:.2f}%")
    else:
        print("No images found in CSV/Folder to test.")


if __name__ == "__main__":
    run_test()
