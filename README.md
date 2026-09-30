# ☀️ Solar Panel Defect Detection AI

An end-to-end computer vision application for automated solar panel health monitoring and defect detection. Built with **Streamlit**, **YOLOv8 / YOLO26**, and **ONNX Runtime**, this platform provides real-time detection of panel defects, including surface cracks and soiling/dust accumulation, alongside automated PDF report generation.

---

## 🌟 Key Features

- 🎯 **Automated Defect Identification**: Accurately detects and classifies solar panel defects:
  - **Cracked**: Surface cracks, micro-fractures, and structural panel damages.
  - **Dusty**: Light to heavy soiling, dust layer accumulation, and surface contamination.
  - **Panel**: Photovoltaic solar panel boundaries.
- ⚡ **Dual Inference Backend**: Choose between standard **PyTorch (`.pt`)** and high-speed **ONNX Runtime (`.onnx`)** execution engine (~19.7 CPU FPS).
- 📊 **Model Metrics & Calibrated Thresholds**: Integrates benchmark evaluations (mAP50 = **87.3%** on test split) with adjustable class confidence sliders for precision tuning.
- 📑 **PDF Inspection Report Generation**: Automatically generates downloadable, multi-page inspection summary reports formatted with **ReportLab**, including detection counts, severity classifications, and annotated image previews.
- 🖼️ **Interactive Inspection Dashboard**: Analyze uploaded images or select from built-in sample test cases (clean panel, light dust, heavy dust, single crack, multiple cracks).

---

## 📁 Repository Structure

```text
solar-panel-defect-detection/
├── models/
│   ├── b2_best.pt          # PyTorch YOLO model weights
│   └── b2_best.onnx        # Exported ONNX model weights
├── test_images/            # Sample panel images for testing
│   ├── clean_panel.jpg
│   ├── crack.jpg
│   ├── dust.jpg
│   └── ...
├── app.py                  # Main Streamlit web application
├── requirements.txt        # Project dependencies
├── .gitignore              # Ignored files and directories
└── README.md               # Project documentation
```

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/Deadpool20x/solar-panel-defect-detection.git
cd solar-panel-defect-detection
```

### 3. Set Up Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 📊 Model Performance & Benchmarks

| Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Validation mAP50** | `86.6%` | Leakage-free dataset resplit |
| **Test mAP50** | `87.3%` | Overall test set accuracy |
| **Dusty Class mAP50** | `79.0%` | Specific soiling detection benchmark |
| **Inference Speed** | `~19.7 FPS` | CPU reference benchmark |

---

## 🛠️ Built With

- **[Streamlit](https://streamlit.io/)** - Interactive web UI framework
- **[Ultralytics YOLO](https://docs.ultralytics.com/)** - Object detection model framework
- **[ONNX Runtime](https://onnxruntime.ai/)** - Optimized cross-platform inference engine
- **[ReportLab](https://www.reportlab.com/)** - PDF document generation library
- **[OpenCV](https://opencv.org/) & [PIL](https://pillow.readthedocs.io/)** - Image processing utilities

---

## 📄 License & Attribution

Distributed under the MIT License. See `LICENSE` for more information.  
Repository maintained by [Deadpool20x](https://github.com/Deadpool20x).
