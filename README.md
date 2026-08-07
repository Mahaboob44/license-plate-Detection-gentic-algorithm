# Detection of License Plates Through Genetic Algorithms and Neural Networks

A Python re-implementation of our B.Tech (ECE) final year project — an **Automatic
License Plate Recognition (ALPR)** system that combines a **Genetic Algorithm (GA)**
for license plate localization with a **Neural-Network-based OCR** engine for
character recognition, wrapped in a **Tkinter GUI** for a Vehicle Fine Management
System.

> Original project: *Detection of License Plates Through Genetic Algorithms and
> Neural Networks* — G. Pullaiah College of Engineering and Technology (GPCET),
> Kurnool. Department of ECE, 2022–2026.
> Authors: T. Veerendra Prasad, V. Narayana, S. Mahaboob Basha.
> Guide: Dr. B. Ravi Chandra.

This repo re-implements the MATLAB prototype described in the project report in
pure Python so it can be run, extended, and version-controlled on GitHub.

---

## ✨ Features

- **Image preprocessing** — grayscale conversion, Canny edge detection.
- **Genetic Algorithm plate localization** — chromosomes `[x, y, w, h]` evolved
  with selection, one-point crossover, and mutation. Fitness combines:
  - Edge density
  - Aspect-ratio score (plates ≈ 3.5–5:1)
  - Intensity variance
- **Neural-network OCR** — uses [EasyOCR](https://github.com/JaidedAI/EasyOCR)
  (a CRNN + attention deep-learning OCR engine) to read the alphanumeric plate
  text from the localized region — the modern, pip-installable equivalent of the
  "Neural Network OCR" module described in the report.
- **Vehicle Fine Management GUI** — Tkinter desktop app to upload a vehicle
  image, run GA + NN detection, auto-fill owner details from a local SQLite
  database, issue fines, mark challans paid, and view all records.
- **Performance graphs** — regenerates the GA fitness convergence, edge-density
  vs. accuracy, and execution-time plots from the report (`src/plots.py`).

## 📁 Project Structure

```
alpr-ga-nn/
├── README.md
├── requirements.txt
├── main.py                  # entry point — launches the GUI
├── data/
│   ├── vehicle_db.json      # seed data (matches the report's sample DB)
│   └── sample_images/       # put test vehicle photos here
├── models/                  # (optional) place custom-trained OCR weights here
├── output/                  # generated plots / detected-plate crops land here
└── src/
    ├── __init__.py
    ├── preprocessing.py     # grayscale + edge detection
    ├── genetic_algorithm.py # GA-based plate localization
    ├── ocr_engine.py        # NN-based character recognition (EasyOCR)
    ├── database.py          # SQLite vehicle / fine management
    ├── gui.py                # Tkinter Vehicle Fine Management System
    └── plots.py              # regenerates the report's performance graphs
```

## 🛠️ Setup

```bash
git clone https://github.com/<your-username>/alpr-ga-nn.git
cd alpr-ga-nn
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** EasyOCR downloads its detection/recognition weights (~100 MB) the
> first time it runs, and pulls in `torch`. If you're on a low-resource machine
> and just want to test the GA localization step without OCR, you can comment
> out the `ocr_engine` import in `src/gui.py` — the GA module has zero heavy
> dependencies (just `numpy` + `opencv-python`).

## ▶️ Running

```bash
python main.py
```

This opens the Vehicle Fine Management GUI:

1. **Upload Vehicle Image** — choose a car photo.
2. **Detect License Plate (GA + NN)** — runs the Genetic Algorithm to localize
   the plate, draws the bounding box, then runs OCR to read the plate text.
2. If the plate matches a record in `data/vehicle_db.json`, the owner's
   details auto-fill; otherwise fill them in manually.
3. **Issue Fine** — pick a violation, enter the amount, and save.
4. **Pay Challan** — clears the outstanding fine for the selected plate.
5. **Show All Records** — refreshes the records table.

## 📊 Regenerating the report's graphs

```bash
python src/plots.py
```

Outputs `ga_fitness_convergence.png`, `edge_density_vs_accuracy.png`, and
`execution_time_per_generation.png` into `output/`.

## 🧪 Using the GA module standalone

```python
from src.preprocessing import load_image, to_grayscale, canny_edges
from src.genetic_algorithm import GeneticPlateLocator

img = load_image("data/sample_images/car1.jpg")
gray = to_grayscale(img)
edges = canny_edges(gray)

locator = GeneticPlateLocator(image_shape=gray.shape, population_size=40,
                               generations=60)
best_box, fitness_history = locator.run(gray, edges)
print("Best bounding box [x, y, w, h]:", best_box)
```

## 🔮 Future Work (from the original report)

- Swap EasyOCR for a custom-trained CNN+CTC / transformer OCR model.
- Real-time video stream processing (OpenCV `VideoCapture` + GPU acceleration).
- Cloud-hosted violation database instead of local SQLite.
- Multilingual / regional plate format support.
- Deployment on edge devices (Raspberry Pi / Jetson Nano).

## 📄 License

MIT — see [LICENSE](LICENSE). Feel free to fork and build on this for your own
coursework, just keep the attribution to the original authors.
