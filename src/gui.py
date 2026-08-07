"""
Automatic License Plate Recognition & Fine Management System — GUI
=====================================================================
Python/Tkinter re-implementation of the MATLAB App Designer GUI shown in
Fig. 6 of the project report (Annexure: VehicleFineSystem_Fixed.m).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageDraw, ImageTk

from . import database as db
from . import ocr_engine
from .genetic_algorithm import GAConfig, GeneticPlateLocator
from .preprocessing import canny_edges, load_image, to_grayscale

VIOLATIONS = ["Signal Jumping", "Speeding", "No Helmet", "Triple Riding", "No Seatbelt"]


class ALPRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vehicle Fine Management with GA & NN")
        self.geometry("1050x640")
        self.configure(bg="#f0f0fa")

        db.init_db()

        self.cv_image = None          # original BGR image (numpy)
        self.detected_plate = None    # cleaned plate text string
        self.tk_photo = None          # keep a reference so Tk doesn't GC it

        self._build_layout()
        self._refresh_table()

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    def _build_layout(self):
        title = tk.Label(
            self,
            text="Automatic License Plate Recognition & Fine Management System",
            font=("Segoe UI", 16, "bold"),
            bg="#f0f0fa",
        )
        title.pack(pady=10)

        main = tk.Frame(self, bg="#f0f0fa")
        main.pack(fill="both", expand=True, padx=10)

        # --- Left: image + buttons ---
        left = tk.Frame(main, bg="#f0f0fa")
        left.pack(side="left", fill="y", padx=10)

        self.canvas = tk.Canvas(left, width=420, height=320, bg="white", relief="groove", bd=2)
        self.canvas.pack()

        tk.Button(left, text="Upload Vehicle Image", width=25, command=self.upload_image).pack(pady=(15, 5))
        self.detect_btn = tk.Button(
            left, text="Detect License Plate (GA + NN)", width=25,
            command=self.detect_plate, state="disabled",
        )
        self.detect_btn.pack(pady=5)
        self.status_label = tk.Label(left, text="", bg="#f0f0fa", fg="#555")
        self.status_label.pack(pady=5)

        # --- Right: vehicle detail form ---
        right = tk.Frame(main, bg="#f0f0fa")
        right.pack(side="left", fill="both", expand=True, padx=20)

        self.fields = {}
        field_defs = [
            ("License Plate", "plate"),
            ("Owner Name", "owner"),
            ("Phone Number", "phone"),
            ("Occupation", "occupation"),
            ("Area", "area"),
        ]
        for i, (label, key) in enumerate(field_defs):
            tk.Label(right, text=label + ":", bg="#f0f0fa").grid(row=i, column=0, sticky="w", pady=4)
            entry = tk.Entry(right, width=30)
            entry.grid(row=i, column=1, pady=4, sticky="w")
            self.fields[key] = entry

        row = len(field_defs)
        tk.Label(right, text="Violation:", bg="#f0f0fa").grid(row=row, column=0, sticky="w", pady=4)
        self.violation_var = tk.StringVar(value=VIOLATIONS[0])
        ttk.Combobox(
            right, textvariable=self.violation_var, values=VIOLATIONS, width=27, state="readonly"
        ).grid(row=row, column=1, pady=4, sticky="w")

        row += 1
        tk.Label(right, text="Fine Amount:", bg="#f0f0fa").grid(row=row, column=0, sticky="w", pady=4)
        self.fine_entry = tk.Entry(right, width=30)
        self.fine_entry.insert(0, "500")
        self.fine_entry.grid(row=row, column=1, pady=4, sticky="w")

        row += 1
        btn_frame = tk.Frame(right, bg="#f0f0fa")
        btn_frame.grid(row=row, column=0, columnspan=2, pady=15, sticky="w")
        tk.Button(btn_frame, text="Issue Fine", width=14, command=self.issue_fine).grid(row=0, column=0, padx=3)
        tk.Button(btn_frame, text="Pay Challan", width=14, command=self.pay_fine).grid(row=0, column=1, padx=3)
        tk.Button(btn_frame, text="Show All Records", width=16, command=self._refresh_table).grid(row=0, column=2, padx=3)

        # --- Bottom: records table ---
        table_frame = tk.Frame(self, bg="#f0f0fa")
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)

        columns = ("plate", "owner", "area", "violations", "fines", "status")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)
        headings = ["License Plate", "Owner", "Area", "Violations", "Total Fines", "Status"]
        for col, head in zip(columns, headings):
            self.tree.heading(col, text=head)
            self.tree.column(col, width=150 if col != "violations" else 220)
        self.tree.pack(fill="both", expand=True)

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Select Vehicle Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png")],
        )
        if not path:
            return
        self.cv_image = load_image(path)
        self._show_image(self.cv_image)
        self.detect_btn.config(state="normal")
        self.status_label.config(text="Image loaded. Click Detect.")

    def detect_plate(self):
        if self.cv_image is None:
            messagebox.showerror("Error", "Please upload an image first!")
            return

        self.status_label.config(text="Running GA localization ...")
        self.update_idletasks()

        gray = to_grayscale(self.cv_image)
        edges = canny_edges(gray)

        locator = GeneticPlateLocator(image_shape=gray.shape, config=GAConfig())
        result = locator.run(gray, edges)
        x, y, w, h = result.best_box

        plate_region = self.cv_image[y:y + h, x:x + w]

        self.status_label.config(text="Running NN OCR ...")
        self.update_idletasks()
        try:
            plate_text = ocr_engine.recognize(plate_region)
        except Exception as exc:  # pragma: no cover - depends on optional heavy dep
            messagebox.showwarning(
                "OCR unavailable",
                f"OCR engine failed to run ({exc}).\n"
                "You can still see the GA-detected bounding box.",
            )
            plate_text = None

        self.detected_plate = plate_text
        self.fields["plate"].delete(0, tk.END)
        self.fields["plate"].insert(0, plate_text or "")

        self._show_image(self.cv_image, box=(x, y, w, h))
        self.status_label.config(
            text=f"GA fitness: {result.best_fitness:.3f} | "
                 f"Generations: {result.generations_run} | "
                 f"Plate: {plate_text or '(not recognized)'}"
        )

        if plate_text:
            record = db.get_vehicle(plate_text)
            if record:
                self.fields["owner"].delete(0, tk.END)
                self.fields["owner"].insert(0, record.owner_name)
                self.fields["phone"].delete(0, tk.END)
                self.fields["phone"].insert(0, record.phone_number)
                self.fields["occupation"].delete(0, tk.END)
                self.fields["occupation"].insert(0, record.occupation)
                self.fields["area"].delete(0, tk.END)
                self.fields["area"].insert(0, record.area)
            else:
                messagebox.showinfo("Info", "Vehicle not found in database. Add details manually.")

    def issue_fine(self):
        plate = self.fields["plate"].get().strip().upper()
        if not plate:
            messagebox.showerror("Error", "No license plate detected / entered!")
            return
        try:
            amount = int(self.fine_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Fine amount must be a number.")
            return

        owner_details = {
            "owner_name": self.fields["owner"].get(),
            "phone_number": self.fields["phone"].get(),
            "occupation": self.fields["occupation"].get(),
            "area": self.fields["area"].get(),
        }
        db.issue_fine(plate, self.violation_var.get(), amount, owner_details)
        messagebox.showinfo("Success", "Fine issued successfully!")
        self._refresh_table()

    def pay_fine(self):
        plate = self.fields["plate"].get().strip().upper()
        if not plate:
            messagebox.showerror("Error", "No license plate detected / entered!")
            return
        record = db.pay_fine(plate)
        if record is None:
            messagebox.showerror("Error", "Vehicle not found!")
            return
        messagebox.showinfo("Success", "Challan paid successfully!")
        self._refresh_table()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _show_image(self, cv_image, box=None):
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((420, 320))

        if box is not None:
            scale_x = pil_img.width / cv_image.shape[1]
            scale_y = pil_img.height / cv_image.shape[0]
            x, y, w, h = box
            draw = ImageDraw.Draw(pil_img)
            draw.rectangle(
                [x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y],
                outline="yellow",
                width=3,
            )

        self.tk_photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(210, 160, image=self.tk_photo)

    def _refresh_table(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for record in db.get_all_vehicles():
            self.tree.insert(
                "", "end",
                values=(
                    record.license_plate,
                    record.owner_name,
                    record.area,
                    record.violations,
                    record.total_fines,
                    record.status,
                ),
            )


def run():
    app = ALPRApp()
    app.mainloop()
