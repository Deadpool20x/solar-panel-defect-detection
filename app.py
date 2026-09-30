from __future__ import annotations

import io
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as PDFImage,
    PageBreak,
)
from ultralytics import YOLO


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Solar Panel Inspection AI",
    page_icon="☀️",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_NAME = "YOLO26n B2"

CLASS_NAMES = {
    0: "cracked",
    1: "dusty",
    2: "panel",
}

DEFECT_CLASSES = {
    "cracked",
    "dusty",
}

PT_MODEL_PATH = Path("models/b2_best.pt")
ONNX_MODEL_PATH = Path("models/b2_best.onnx")


@dataclass(frozen=True)
class B2Metrics:
    """Measured B2 experiment metrics — retrained on the leakage-free
    (group + visual-duplicate resplit) dataset. Replaces the earlier
    numbers, which were measured on a split with confirmed train/test
    leakage and are no longer valid."""

    validation_map50: float
    test_map50: float
    test_dusty_map50: float  # dusty-only, test split — only 28 images, wider uncertainty than the other classes
    reference_cpu_fps: float


METRICS = B2Metrics(
    validation_map50=0.866,
    test_map50=0.873,
    test_dusty_map50=0.790,
    reference_cpu_fps=19.7,
)

# Run inference broadly, then apply our calibrated class thresholds.
INTERNAL_CONFIDENCE = 0.01


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .inspection-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .inspection-subtitle {
            color: #8b8f98;
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
        }

        .section-title {
            font-size: 1.55rem;
            font-weight: 700;
            margin-top: 1.2rem;
            margin-bottom: 0.8rem;
        }

        .status-good {
            padding: 0.9rem 1rem;
            border-radius: 0.55rem;
            background: rgba(16, 185, 129, 0.18);
            color: #34d399;
            font-weight: 600;
        }

        .status-warning {
            padding: 0.9rem 1rem;
            border-radius: 0.55rem;
            background: rgba(245, 158, 11, 0.18);
            color: #fbbf24;
            font-weight: 600;
        }

        .report-note {
            color: #9ca3af;
            font-size: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL PATH
# ============================================================

def get_model_path(engine: str) -> Path:
    if engine == "ONNX":
        return ONNX_MODEL_PATH

    if engine == "PyTorch":
        return PT_MODEL_PATH

    raise ValueError(f"Unsupported engine: {engine}")


# ============================================================
# ENVIRONMENT VALIDATION
# ============================================================

def validate_environment(engine: str) -> None:
    model_path = get_model_path(engine)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found:\n\n{model_path}"
        )

    if engine == "ONNX":

        try:
            import onnx  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Package 'onnx' is missing.\n\n"
                "Install it with:\n"
                "python -m pip install onnx"
            ) from exc

        try:
            import onnxruntime  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Package 'onnxruntime' is missing.\n\n"
                "Install it with:\n"
                "python -m pip install onnxruntime"
            ) from exc


# ============================================================
# MODEL LOAD / WARMUP
# ============================================================

def warmup_model(model: YOLO) -> None:
    dummy = Image.new(
        "RGB",
        (640, 640),
        (0, 0, 0),
    )

    model.predict(
        source=dummy,
        imgsz=640,
        conf=INTERNAL_CONFIDENCE,
        verbose=False,
    )


@st.cache_resource(show_spinner=False)
def load_model(engine: str) -> YOLO:

    validate_environment(engine)

    path = get_model_path(engine)

    model = YOLO(
        str(path),
        task="detect",
    )

    loaded_names = {
        int(k): str(v).lower()
        for k, v in model.names.items()
    }

    if loaded_names != CLASS_NAMES:
        raise RuntimeError(
            "Loaded model is not the expected B2 model.\n\n"
            f"Expected: {CLASS_NAMES}\n"
            f"Loaded:   {loaded_names}"
        )

    warmup_model(model)

    return model


# ============================================================
# CLASS-SPECIFIC THRESHOLDS
# ============================================================

def get_class_thresholds(
    user_confidence: float,
) -> dict[str, float]:
    """
    Calibrated from the B2 validation threshold experiment.

    At UI confidence 0.25:
        cracked = 0.15
        dusty   = 0.25
        panel   = 0.25
    """

    return {
        "cracked": max(
            0.01,
            user_confidence * 0.60,
        ),
        "dusty": user_confidence,
        "panel": user_confidence,
    }


# ============================================================
# INFERENCE
# ============================================================

def run_inference(
    model: YOLO,
    image: Image.Image,
    user_confidence: float,
    imgsz: int,
) -> tuple[Any, float]:

    start = time.perf_counter()

    results = model.predict(
        source=image,
        imgsz=imgsz,
        conf=INTERNAL_CONFIDENCE,
        verbose=False,
    )

    elapsed = time.perf_counter() - start

    if not results:
        raise RuntimeError(
            "Model returned no prediction result."
        )

    result = results[0]

    thresholds = get_class_thresholds(
        user_confidence
    )

    if result.boxes is not None:

        keep_indices = []

        for i, box in enumerate(result.boxes):

            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            class_name = str(
                result.names.get(
                    cls_id,
                    CLASS_NAMES.get(
                        cls_id,
                        f"class_{cls_id}",
                    ),
                )
            ).lower()

            threshold = thresholds.get(
                class_name,
                user_confidence,
            )

            if conf >= threshold:
                keep_indices.append(i)

        if keep_indices:
            result.boxes = result.boxes[keep_indices]
        else:
            result.boxes = None

    return result, elapsed


# ============================================================
# DETECTION EXTRACTION
# ============================================================

def extract_detections(
    result: Any,
) -> list[dict[str, Any]]:

    detections = []

    if result.boxes is None:
        return detections

    for index, box in enumerate(result.boxes):

        cls_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        class_name = str(
            result.names.get(
                cls_id,
                CLASS_NAMES.get(
                    cls_id,
                    f"class_{cls_id}",
                ),
            )
        )

        detections.append(
            {
                "Detection": index + 1,
                "Class": class_name,
                "Confidence": round(confidence, 3),
                "X1": round(x1, 1),
                "Y1": round(y1, 1),
                "X2": round(x2, 1),
                "Y2": round(y2, 1),
            }
        )

    return detections


# ============================================================
# IMAGE RENDERING
# ============================================================

def make_annotated_image(
    result: Any,
) -> Image.Image:

    plotted = result.plot()

    rgb = plotted[..., ::-1]

    return Image.fromarray(
        np.asarray(rgb).astype(np.uint8)
    )


# ============================================================
# CLASS COUNTS
# ============================================================

def count_classes(
    detections: list[dict[str, Any]],
) -> dict[str, int]:

    counts = {
        "cracked": 0,
        "dusty": 0,
        "panel": 0,
    }

    for detection in detections:

        name = detection["Class"].lower()

        if name in counts:
            counts[name] += 1

    return counts


# ============================================================
# STATUS TEXT
# ============================================================

def get_status_text(
    counts: dict[str, int],
) -> str:

    defects = (
        counts["cracked"]
        + counts["dusty"]
    )

    if defects == 0:
        return (
            "No trained defect detected above "
            "the selected confidence thresholds."
        )

    parts = []

    if counts["cracked"] > 0:
        parts.append(
            f"{counts['cracked']} cracked"
        )

    if counts["dusty"] > 0:
        parts.append(
            f"{counts['dusty']} dusty"
        )

    return (
        "Potential maintenance issue detected: "
        + ", ".join(parts)
        + "."
    )


def show_status(
    counts: dict[str, int],
) -> None:

    defects = (
        counts["cracked"]
        + counts["dusty"]
    )

    if defects == 0:

        st.markdown(
            """
            <div class="status-good">
                ✅ No trained defect detected above the selected thresholds.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "This does not guarantee that the panel is defect-free. "
            "The model only detects the trained visual classes."
        )

    else:

        st.markdown(
            f"""
            <div class="status-warning">
                ⚠️ {get_status_text(counts)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "Review the detected regions and confidence values before action."
        )


# ============================================================
# PDF HELPERS
# ============================================================

def pil_to_png_bytes(
    image: Image.Image,
) -> io.BytesIO:

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    buffer.seek(0)

    return buffer


def build_pdf_report(
    original_image: Image.Image,
    annotated_image: Image.Image,
    detections: list[dict[str, Any]],
    counts: dict[str, int],
    engine: str,
    model_path: Path,
    confidence: float,
    class_thresholds: dict[str, float],
    inference_size: int,
    elapsed: float,
    fps: float,
    source_filename: str,
) -> bytes:

    output = io.BytesIO()

    doc = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="Solar Panel Inspection Report",
        author="Solar Panel Inspection AI",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        textColor=colors.grey,
        spaceAfter=16,
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
    )

    story = []

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Solar Panel Inspection AI",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Automated Visual Inspection Report",
            subtitle_style,
        )
    )

    # --------------------------------------------------------
    # GENERAL INFORMATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Inspection Information",
            heading_style,
        )
    )

    information = [
        ["Date / Time", timestamp],
        ["Source Image", source_filename],
        ["Model", MODEL_NAME],
        ["Inference Engine", engine],
        ["Model File", model_path.name],
        ["Inference Size", f"{inference_size} x {inference_size}"],
        ["UI Confidence", f"{confidence:.2f}"],
        [
            "Cracked Threshold",
            f"{class_thresholds['cracked']:.2f}",
        ],
        [
            "Dusty Threshold",
            f"{class_thresholds['dusty']:.2f}",
        ],
        [
            "Panel Threshold",
            f"{class_thresholds['panel']:.2f}",
        ],
    ]

    info_table = Table(
        information,
        colWidths=[
            48 * mm,
            125 * mm,
        ],
    )

    info_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor("#eeeeee"),
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#cccccc"),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, -1),
                    "Helvetica",
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8.5,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )

    story.append(info_table)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Inspection Summary",
            heading_style,
        )
    )

    latency_ms = elapsed * 1000

    summary = [
        ["Panels", str(counts["panel"])],
        ["Cracked", str(counts["cracked"])],
        ["Dusty", str(counts["dusty"])],
        ["Latency", f"{latency_ms:.1f} ms"],
        ["FPS", f"{fps:.1f}"],
        ["Status", get_status_text(counts)],
    ]

    summary_table = Table(
        summary,
        colWidths=[
            48 * mm,
            125 * mm,
        ],
    )

    summary_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor("#eeeeee"),
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#cccccc"),
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8.5,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )

    story.append(summary_table)

    # --------------------------------------------------------
    # DETECTIONS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Detection Details",
            heading_style,
        )
    )

    if detections:

        table_data = [
            [
                "Detection",
                "Class",
                "Confidence",
                "X1",
                "Y1",
                "X2",
                "Y2",
            ]
        ]

        for detection in detections:

            table_data.append(
                [
                    str(detection["Detection"]),
                    str(detection["Class"]),
                    f"{detection['Confidence']:.3f}",
                    f"{detection['X1']:.1f}",
                    f"{detection['Y1']:.1f}",
                    f"{detection['X2']:.1f}",
                    f"{detection['Y2']:.1f}",
                ]
            )

        detection_table = Table(
            table_data,
            repeatRows=1,
            colWidths=[
                18 * mm,
                28 * mm,
                25 * mm,
                20 * mm,
                20 * mm,
                20 * mm,
                20 * mm,
            ],
        )

        detection_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#dddddd"),
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.4,
                        colors.HexColor("#bbbbbb"),
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold",
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        7.5,
                    ),
                    (
                        "ALIGN",
                        (2, 1),
                        (-1, -1),
                        "RIGHT",
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE",
                    ),
                ]
            )
        )

        story.append(detection_table)

    else:

        story.append(
            Paragraph(
                "No detections passed the selected class thresholds.",
                body_style,
            )
        )

    # --------------------------------------------------------
    # IMAGES
    # --------------------------------------------------------

    story.append(PageBreak())

    story.append(
        Paragraph(
            "Inspection Images",
            heading_style,
        )
    )

    original_buffer = pil_to_png_bytes(
        original_image
    )

    annotated_buffer = pil_to_png_bytes(
        annotated_image
    )

    original_pdf_image = PDFImage(
        original_buffer,
        width=82 * mm,
        height=60 * mm,
        kind="proportional",
    )

    annotated_pdf_image = PDFImage(
        annotated_buffer,
        width=82 * mm,
        height=60 * mm,
        kind="proportional",
    )

    image_table = Table(
        [
            [
                Paragraph(
                    "<b>Original</b>",
                    body_style,
                ),
                Paragraph(
                    "<b>AI Detection</b>",
                    body_style,
                ),
            ],
            [
                original_pdf_image,
                annotated_pdf_image,
            ],
        ],
        colWidths=[
            88 * mm,
            88 * mm,
        ],
    )

    image_table.setStyle(
        TableStyle(
            [
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
            ]
        )
    )

    story.append(image_table)

    # --------------------------------------------------------
    # MODEL BENCHMARK INFORMATION
    # --------------------------------------------------------

    story.append(
        Spacer(
            1,
            8,
        )
    )

    story.append(
        Paragraph(
            "Model Benchmark Information",
            heading_style,
        )
    )

    benchmark_data = [
        [
            "Validation mAP50",
            f"{METRICS.validation_map50:.3f}",
        ],
        [
            "Test mAP50",
            f"{METRICS.test_map50:.3f}",
        ],
        [
            "Test mAP50 - dusty class only (n=28 images)",
            f"{METRICS.test_dusty_map50:.3f}",
        ],
        [
            "Reference CPU benchmark",
            f"~{METRICS.reference_cpu_fps:.1f} FPS",
        ],
        [
            "Current inspection speed",
            f"{fps:.1f} FPS",
        ],
    ]

    benchmark_table = Table(
        benchmark_data,
        colWidths=[
            60 * mm,
            113 * mm,
        ],
    )

    benchmark_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor("#eeeeee"),
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#cccccc"),
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8.5,
                ),
            ]
        )
    )

    story.append(
        benchmark_table
    )

    story.append(
        Spacer(
            1,
            10,
        )
    )

    story.append(
        Paragraph(
            "Note: model benchmark values are dataset evaluation "
            "metrics, while current inspection speed is measured "
            "on the local application hardware.",
            body_style,
        )
    )

    story.append(
        Spacer(
            1,
            10,
        )
    )

    story.append(
        Paragraph(
            "Generated by Solar Panel Inspection AI.",
            body_style,
        )
    )

    # --------------------------------------------------------
    # BUILD
    # --------------------------------------------------------

    doc.build(story)

    output.seek(0)

    return output.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## Detection Settings")

    engine = st.radio(
        "Inference Engine",
        ["ONNX", "PyTorch"],
        index=0,
    )

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
    )

    inference_size = st.selectbox(
        "Inference size",
        [320, 416, 512, 640, 768, 960],
        index=3,
    )

    class_thresholds = get_class_thresholds(
        confidence
    )

    st.divider()

    st.markdown("## Model Information")

    st.write(
        f"**{MODEL_NAME} - {engine}**"
    )

    st.write(
        f"Model file: **{get_model_path(engine).name}**"
    )

    st.caption(
        "Classes: cracked / dusty / panel"
    )

    st.write(
        f"Validation mAP50: **{METRICS.validation_map50:.3f}**"
    )

    st.write(
        f"Test mAP50: **{METRICS.test_map50:.3f}**"
    )

    with st.expander(
        "Active class thresholds",
        expanded=False,
    ):

        st.write(
            f"Cracked: **{class_thresholds['cracked']:.2f}**"
        )

        st.write(
            f"Dusty: **{class_thresholds['dusty']:.2f}**"
        )

        st.write(
            f"Panel: **{class_thresholds['panel']:.2f}**"
        )

    try:

        model = load_model(engine)

        st.success(
            f"{engine} B2 engine ready"
        )

    except Exception as exc:

        st.error(
            str(exc)
        )

        st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="inspection-title">'
    "Solar Panel Inspection AI"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="inspection-subtitle">'
    "AI-assisted solar-panel visual inspection using YOLO26n B2"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a solar-panel image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
)

if uploaded_file is None:

    st.info(
        "Upload a solar-panel image to begin inspection."
    )

    st.stop()


# ============================================================
# LOAD IMAGE
# ============================================================

try:

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")

except Exception as exc:

    st.error(
        f"Could not read image: {exc}"
    )

    st.stop()


# ============================================================
# RUN INFERENCE
# ============================================================

try:

    result, elapsed = run_inference(
        model=model,
        image=original_image,
        user_confidence=confidence,
        imgsz=inference_size,
    )

except Exception as exc:

    st.error(
        f"Inference failed:\n\n{exc}"
    )

    st.stop()


# ============================================================
# PROCESS RESULTS
# ============================================================

detections = extract_detections(
    result
)

counts = count_classes(
    detections
)

annotated_image = make_annotated_image(
    result
)

fps = (
    1.0 / elapsed
    if elapsed > 0
    else 0.0
)


# ============================================================
# RESULT IMAGES
# ============================================================

st.markdown(
    '<div class="section-title">'
    "Inspection Result"
    "</div>",
    unsafe_allow_html=True,
)

left, right = st.columns(
    2,
    gap="large",
)

with left:

    st.subheader("Original")

    st.image(
        original_image,
        width="stretch",
    )

with right:

    st.subheader(
        f"AI Detection - {engine}"
    )

    st.image(
        annotated_image,
        width="stretch",
    )


# ============================================================
# SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">'
    "Inspection Summary"
    "</div>",
    unsafe_allow_html=True,
)

cols = st.columns(5)

cols[0].metric(
    "Panels",
    counts["panel"],
)

cols[1].metric(
    "Cracks",
    counts["cracked"],
)

cols[2].metric(
    "Dusty",
    counts["dusty"],
)

cols[3].metric(
    "Latency",
    f"{elapsed * 1000:.1f} ms",
)

cols[4].metric(
    "FPS",
    f"{fps:.1f}",
)


# ============================================================
# STATUS
# ============================================================

st.markdown(
    '<div class="section-title">'
    "Inspection Status"
    "</div>",
    unsafe_allow_html=True,
)

show_status(counts)


# ============================================================
# DETECTION TABLE
# ============================================================

st.markdown(
    '<div class="section-title">'
    "Detection Details"
    "</div>",
    unsafe_allow_html=True,
)

if detections:

    df = pd.DataFrame(
        detections
    )

    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
    )

else:

    st.info(
        "No detections passed the selected class thresholds."
    )


# ============================================================
# PDF REPORT
# ============================================================

st.markdown(
    '<div class="section-title">'
    "Inspection Report"
    "</div>",
    unsafe_allow_html=True,
)

st.caption(
    "Generate a PDF containing this inspection's results and images."
)

try:

    pdf_bytes = build_pdf_report(
        original_image=original_image,
        annotated_image=annotated_image,
        detections=detections,
        counts=counts,
        engine=engine,
        model_path=get_model_path(engine),
        confidence=confidence,
        class_thresholds=class_thresholds,
        inference_size=inference_size,
        elapsed=elapsed,
        fps=fps,
        source_filename=uploaded_file.name,
    )

    report_name = (
        Path(uploaded_file.name).stem
        + "_inspection_report.pdf"
    )

    st.download_button(
        label="Download Inspection Report (PDF)",
        data=pdf_bytes,
        file_name=report_name,
        mime="application/pdf",
        width="stretch",
    )

except Exception as exc:

    st.error(
        f"Could not generate PDF report: {exc}"
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "Model / Benchmark Information",
    expanded=False,
):

    st.write(
        f"**Model:** {MODEL_NAME}"
    )

    st.write(
        f"**Inference engine:** {engine}"
    )

    st.write(
        f"**Model file:** `{get_model_path(engine)}`"
    )

    st.write(
        "**Classes:** cracked, dusty, panel"
    )

    st.write(
        f"**Validation mAP50:** {METRICS.validation_map50:.3f}"
    )

    st.write(
        f"**Test mAP50:** {METRICS.test_map50:.3f}"
    )

    st.write(
        f"**Test mAP50 (dusty only, n=28):** {METRICS.test_dusty_map50:.3f}"
    )

    st.write(
        f"**Reference CPU benchmark:** ~{METRICS.reference_cpu_fps:.1f} FPS"
    )

    st.write(
        f"**Current inference:** {elapsed * 1000:.1f} ms "
        f"({fps:.1f} FPS)"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Solar Panel Inspection AI - YOLO26n B2 - "
    "3 classes: cracked / dusty / panel"
)