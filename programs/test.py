import os
import pandas as pd
import test2

IMAGES_FOLDER = "./Rdata/raw_data"
DATABASE_CSV = "./Rdata/labels.csv"
FAILED_LOG_CSV = "./Rdata/failed_log.csv"


def load_failed_log():
    if os.path.exists(FAILED_LOG_CSV):
        return pd.read_csv(FAILED_LOG_CSV)
    else:
        return pd.DataFrame(columns=["filename", "status"])


def save_failed_log(df):
    df.to_csv(FAILED_LOG_CSV, index=False)
    print(f"Updated {FAILED_LOG_CSV}.")


def process_all_images():
    print("\n--- STARTING LOOP 1: Processing ALL images in folder ---")

    if not os.path.exists(IMAGES_FOLDER):
        print("Image folder not found.")
        return

    all_files = [
        f
        for f in os.listdir(IMAGES_FOLDER)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    failed_df = load_failed_log()

    failed_set = set(failed_df["filename"].tolist())

    for filename in all_files:
        image_path = os.path.join(IMAGES_FOLDER, filename)

        result = test2.check_access(image_path, DATABASE_CSV)

        is_success = result["matched"]

        if is_success:
            print(f"[SUCCESS] {filename} - {result['message']}")
            if filename in failed_set:
                print(f"   -> Removing {filename} from failed log (Issue resolved).")
                failed_df = failed_df[failed_df["filename"] != filename]
                failed_set.remove(filename)

        else:
            print(f"[FAILED]  {filename} - {result['message']}")
            if filename not in failed_set:
                print(f"   -> Adding {filename} to failed log.")
                new_row = {"filename": filename, "status": "Access Denied"}
                failed_df = pd.concat(
                    [failed_df, pd.DataFrame([new_row])], ignore_index=True
                )
                failed_set.add(filename)

    save_failed_log(failed_df)


def recheck_failed_images():
    print("\n--- STARTING LOOP 2: Re-checking ONLY failed images ---")

    if not os.path.exists(FAILED_LOG_CSV):
        print("No failed log file found. Run Loop 1 first.")
        return

    failed_df = load_failed_log()

    if failed_df.empty:
        print("Failed log is empty. Nothing to recheck.")
        return

    indexes_to_remove = []

    for index, row in failed_df.iterrows():
        filename = row["filename"]
        image_path = os.path.join(IMAGES_FOLDER, filename)

        if not os.path.exists(image_path):
            print(f"[WARNING] Image {filename} listed in CSV but not found on disk.")
            continue

        result = test2.check_access(image_path, DATABASE_CSV)

        if result["matched"]:
            print(f"[FIXED] {filename} is now ACCESS GRANTED. Removing from log.")
            indexes_to_remove.append(index)
        else:
            print(f"[STILL FAILED] {filename} - {result['message']}")

    if indexes_to_remove:
        failed_df.drop(indexes_to_remove, inplace=True)
        save_failed_log(failed_df)
    else:
        print("No changes. All failed images are still failing.")


if __name__ == "__main__":
    while True:
        print("\nSelect Mode:")
        print("1. Scan ALL images (Update Failed Log)")
        print("2. Re-check ONLY failed images (Fix Log)")
        print("q. Quit")

        choice = input("Enter choice: ").strip().lower()

        if choice == "1":
            process_all_images()
        elif choice == "2":
            recheck_failed_images()
        elif choice == "q":
            break
        else:
            print("Invalid choice.")
