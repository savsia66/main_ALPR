import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import alpr_server as core

BG_DARK = "#1E2838"
BG_MID = "#2C3E50"
ACCENT_PRIMARY = "#6C7BEC"
TEXT_LIGHT = "#F3F4F6"
STATUS_SUCCESS = "#34D399"
STATUS_DANGER = "#F87171"
STATUS_WARN = "#FBBF24"

FONT_HEADER = ("Segoe UI", 26, "bold")
FONT_BUTTON = ("Segoe UI", 12, "bold")
FONT_STATUS = ("Segoe UI", 16)
FONT_LABEL = ("Segoe UI", 11)


class ALPRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ALPR Access Control System (Desktop Mode)")
        self.root.configure(bg=BG_DARK)

        if not core.PLATE_DATABASE:
            messagebox.showwarning(
                "Database Empty", "Could not load database.csv from the server module."
            )

        window_width = 1100
        window_height = 700
        self.center_window_top(window_width, window_height)

        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_script_dir)
        self.db_dir = os.path.join(current_script_dir, "Rdata", "raw_data")
        if not os.path.exists(self.db_dir):
            self.db_dir = os.path.join(project_root, "Rdata", "raw_data")

        self.current_image_path = None

        header = tk.Label(
            root,
            text="🛡️ Secure Gate ALPR System",
            font=FONT_HEADER,
            bg=BG_DARK,
            fg=TEXT_LIGHT,
        )
        header.pack(pady=(30, 20))

        controls_frame = tk.Frame(root, bg=BG_DARK)
        controls_frame.pack(pady=10)

        self.btn_browse = tk.Button(
            controls_frame,
            text="📂 SELECT IMAGE",
            command=self.browse_image,
            font=FONT_BUTTON,
            bg=ACCENT_PRIMARY,
            fg=TEXT_LIGHT,
            activebackground=BG_MID,
            activeforeground=TEXT_LIGHT,
            relief=tk.FLAT,
            width=20,
            height=2,
            cursor="hand2",
        )
        self.btn_browse.pack(side=tk.LEFT, padx=15)

        self.btn_check = tk.Button(
            controls_frame,
            text="🔍 CHECK ACCESS",
            command=self.start_check_thread,
            font=FONT_BUTTON,
            bg=STATUS_SUCCESS,
            fg=BG_DARK,
            activebackground=BG_MID,
            activeforeground=TEXT_LIGHT,
            relief=tk.FLAT,
            width=20,
            height=2,
            state="disabled",
            cursor="hand2",
        )
        self.btn_check.pack(side=tk.LEFT, padx=15)

        self.lbl_status = tk.Label(
            root,
            text="Waiting for image selection...",
            font=FONT_STATUS,
            bg=BG_DARK,
            fg="#9CA3AF",
        )
        self.lbl_status.pack(pady=25)

        img_frame = tk.Frame(root, bg=BG_DARK)
        img_frame.pack(expand=True, fill="both", padx=50)
        img_frame.grid_columnconfigure(0, weight=1)
        img_frame.grid_columnconfigure(1, weight=1)

        self.box_size = (480, 360)

        self.frame_left = tk.Frame(
            img_frame, bg=BG_MID, width=self.box_size[0], height=self.box_size[1]
        )
        self.frame_left.grid(row=0, column=0, padx=20, pady=10)
        self.frame_left.pack_propagate(False)
        self.lbl_img_left = tk.Label(
            self.frame_left,
            text="Scanned Image",
            bg=BG_MID,
            fg=TEXT_LIGHT,
            font=FONT_LABEL,
        )
        self.lbl_img_left.pack(expand=True, fill="both")

        self.frame_right = tk.Frame(
            img_frame, bg=BG_MID, width=self.box_size[0], height=self.box_size[1]
        )
        self.frame_right.grid(row=0, column=1, padx=20, pady=10)
        self.frame_right.pack_propagate(False)
        self.lbl_img_right = tk.Label(
            self.frame_right,
            text="Database Match",
            bg=BG_MID,
            fg=TEXT_LIGHT,
            font=FONT_LABEL,
        )
        self.lbl_img_right.pack(expand=True, fill="both")

        footer = tk.Label(
            root,
            text="Powered by EasyOCR & OpenCV",
            bg=BG_DARK,
            fg="#6B7280",
            font=("Segoe UI", 9),
        )
        footer.pack(side="bottom", pady=15)

    def center_window_top(self, width, height):
        screen_width = self.root.winfo_screenwidth()
        x = (screen_width // 2) - (width // 2)
        y = 0
        self.root.geometry(f"{width}x{height}+{int(x)}+{int(y)}")

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path, self.lbl_img_left)
            self.lbl_status.config(
                text="Image Loaded. Ready to Check Access.", fg=TEXT_LIGHT
            )
            self.btn_check.config(state="normal", bg=ACCENT_PRIMARY)
            self.lbl_img_right.config(image="", text="Database Match")

    def display_image(self, img_source, label_widget):
        if isinstance(img_source, str):
            if not os.path.exists(img_source):
                return
            img = cv2.imread(img_source)
        else:
            img = img_source

        if img is None:
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_pil.thumbnail(self.box_size, Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)

        label_widget.config(image=img_tk, text="")
        label_widget.image = img_tk

    def start_check_thread(self):
        self.btn_check.config(state="disabled", text="PROCESSING...")
        self.lbl_status.config(text="Reading License Plate...", fg=STATUS_WARN)
        thread = threading.Thread(target=self.run_local_check)
        thread.start()

    def run_local_check(self):
        if self.current_image_path is None:
            return
        img = cv2.imread(self.current_image_path)
        if img is None:
            return

        ocr_results = core.reader.readtext(img)

        best_conf = 0.0
        result = {
            "matched": False,
            "plate": None,
            "annotated_image": img.copy(),
            "db_image_path": None,
            "msg": "No Text Detected",
        }

        found_match = False

        for bbox, text, prob in ocr_results:
            prob = float(prob)
            if len(text) < 3:
                continue

            (tl, tr, br, bl) = bbox
            x_min, x_max = int(min(tl[0], bl[0])), int(max(tr[0], br[0]))
            y_min, y_max = int(min(tl[1], tr[1])), int(max(bl[1], br[1]))

            matched_key, match_score = core.find_best_match(text, core.DB_KEYS)

            if matched_key:
                cv2.rectangle(
                    result["annotated_image"],
                    (x_min, y_min),
                    (x_max, y_max),
                    (0, 255, 0),
                    4,
                )

                result["matched"] = True
                result["plate"] = matched_key
                result["msg"] = f"ACCESS GRANTED ({matched_key})"

                filename = core.PLATE_DATABASE.get(matched_key)
                if filename:
                    full_path = os.path.join(self.db_dir, filename)
                    if os.path.exists(full_path):
                        result["db_image_path"] = full_path

                found_match = True
                break

            else:
                if prob > best_conf:
                    best_conf = prob
                    clean_txt = core.cleanup_text(text)
                    result["plate"] = clean_txt
                    result["msg"] = f"ACCESS DENIED | Detected: {clean_txt}"
                    cv2.rectangle(
                        result["annotated_image"],
                        (x_min, y_min),
                        (x_max, y_max),
                        (0, 0, 255),
                        4,
                    )

        self.root.after(0, self.update_ui_results, result)

    def update_ui_results(self, res):
        self.display_image(res["annotated_image"], self.lbl_img_left)

        if res["db_image_path"]:
            self.display_image(res["db_image_path"], self.lbl_img_right)
        else:
            self.lbl_img_right.config(image="", text="No Database Image Found")

        if res["matched"]:
            self.lbl_status.config(text=f"✅ {res['msg']}", fg=STATUS_SUCCESS)
            self.btn_check.config(bg=ACCENT_PRIMARY)
        else:
            self.lbl_status.config(text=f"⛔ {res['msg']}", fg=STATUS_DANGER)
            self.btn_check.config(bg=ACCENT_PRIMARY)

        self.btn_check.config(state="normal", text="🔍 CHECK ACCESS")


if __name__ == "__main__":
    root = tk.Tk()
    app = ALPRApp(root)
    root.mainloop()
