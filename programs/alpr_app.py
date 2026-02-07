import os
import threading
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import test3 as core

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


class ANPRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ANPR Access Control System")
        self.root.configure(bg=BG_DARK)

        window_width = 1000
        window_height = 625
        self.center_window(window_width, window_height)

        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_script_dir)

        self.csv_path = os.path.join(project_root, "Rdata", "labels.csv")
        self.db_dir = os.path.join(project_root, "Rdata", "raw_data")
        self.current_image_path = None

        header = tk.Label(
            root,
            text="🛡️ Secure Gate ANPR System",
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
            width=24,
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
            width=24,
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

    def center_window(self, width, height):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        y = y - 50

        if x < 0:
            x = 0
        if y < 0:
            y = 30

        self.root.geometry(f"{width}x{height}+{x}+{y}")

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
        thread = threading.Thread(target=self.run_check)
        thread.start()

    def run_check(self):
        res = core.check_access(self.current_image_path, self.csv_path, self.db_dir)
        self.root.after(0, self.update_ui_results, res)

    def update_ui_results(self, res):
        if res["annotated_image"] is not None:
            self.display_image(res["annotated_image"], self.lbl_img_left)

        if res["db_image_path"]:
            self.display_image(res["db_image_path"], self.lbl_img_right)
        else:
            self.lbl_img_right.config(image="", text="No Database Image Found")

        if res["matched"]:
            msg = f"✅ ACCESS GRANTED  |  Plate: {res['matched_plate']}"
            self.lbl_status.config(text=msg, fg=STATUS_SUCCESS)
            self.btn_check.config(bg=ACCENT_PRIMARY)
        else:
            if res.get("detected_plate"):
                msg = f"⛔ ACCESS DENIED  |  Plate: {res['detected_plate']}"
            else:
                msg = f"⛔ {res['message']}"
            self.lbl_status.config(text=msg, fg=STATUS_DANGER)
            self.btn_check.config(bg=ACCENT_PRIMARY)

        self.btn_check.config(state="normal", text="🔍 CHECK ACCESS")


if __name__ == "__main__":
    root = tk.Tk()
    app = ANPRApp(root)
    root.mainloop()
