import time
from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Solar Panel Inspection AI",
    page_icon="☀️",
    layout="wide",
)


# ============================================================
# MODEL PATHS
# ============================================================

PT_MODEL_PATH = Path("models/b2_best.pt")
ONNX_MODEL_PATH = Path("models/b2_best.onnx")


# ============================================================
# MODEL LOADER + WARM-UP
# ============================================================

@st.cache_resource
def load_model(engine: str):
    """
    Load the selected B1 model once and warm it up.

    PyTorch:
        models/best.pt

    ONNX:
        models/best.onnx

    The warm-up prevents the first real inference from
    including most of the model/runtime initialization cost.
    """

    # --------------------------------------------------------
    # Load selected model
    # --------------------------------------------------------

    if engine == "PyTorch":

        if not PT_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"PyTorch model not found: {PT_MODEL_PATH}"
            )

        model = YOLO(
            str(PT_MODEL_PATH),
            task="detect",
        )

    elif engine == "ONNX":

        if not ONNX_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {ONNX_MODEL_PATH}"
            )

        model = YOLO(
            str(ONNX_MODEL_PATH),
            task="detect",
        )

    else:
        raise ValueError(
            f"Unknown inference engine: {engine}"
        )

    # --------------------------------------------------------
    # Warm-up inference
    # --------------------------------------------------------

    dummy_image = Image.new(
        "RGB",
        (640, 640),
        color=(0, 0, 0),
    )

    if engine == "ONNX":

        model.predict(
            source=dummy_image,
            imgsz=640,
            conf=0.25,
            device="cpu",
            verbose=False,
        )

    else:

        model.predict(
            source=dummy_image,
            imgsz=640,
            conf=0.25,
            verbose=False,
        )

    return model


# ============================================================
# HEADER
# ============================================================

st.title("☀️ Solar Panel Inspection AI")

st.caption(
    "Edge-AI visual inspection for solar-panel defects"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Detection Settings")

    # --------------------------------------------------------
    # Engine selection
    # --------------------------------------------------------

    engine = st.radio(
        "Inference Engine",
        ["PyTorch", "ONNX"],
        index=0,
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
    )

    # --------------------------------------------------------
    # Image size
    # --------------------------------------------------------

    image_size = st.selectbox(
        "Inference size",
        [640],
        index=0,
    )

    st.divider()

    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    st.write("### Model Information")

    if engine == "PyTorch":

        st.write("**YOLO26n B2 — PyTorch**")

        st.caption(
            "Primary accuracy model"
        )

        st.write(
            "Test mAP50: 0.748"
        )

        st.write(
            "Model file: b2_best.pt"
        )

    else:

        st.write("**YOLO26n B2 — ONNX**")

        st.caption(
            "Edge deployment model"
        )

        st.write(
            "Test mAP50: 0.705"
        )

        st.write(
            "Reference CPU benchmark: ~21 FPS"
        )

        st.write(
            "Model file: b2_best.onnx"
        )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    with st.spinner(
        f"Loading {engine} inference engine..."
    ):

        model = load_model(engine)

    st.sidebar.success(
        f"{engine} engine ready"
    )

except Exception as e:

    st.error(
        f"Model loading failed:\n\n{e}"
    )

    st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a solar-panel image",
    type=["jpg", "jpeg", "png"],
)


if uploaded_file is None:

    st.info(
        "Upload a solar-panel image to begin inspection."
    )

    st.stop()


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(
    uploaded_file
).convert("RGB")


# ============================================================
# RUN REAL INFERENCE
# ============================================================

with st.spinner(
    f"Running {engine} inference..."
):

    start = time.perf_counter()

    if engine == "ONNX":

        results = model.predict(
            source=image,
            imgsz=image_size,
            conf=confidence,
            device="cpu",
            verbose=False,
        )

    else:

        results = model.predict(
            source=image,
            imgsz=image_size,
            conf=confidence,
            verbose=False,
        )

    elapsed = (
        time.perf_counter() - start
    )


# FPS based on the complete prediction call
fps = (
    1.0 / elapsed
    if elapsed > 0
    else 0.0
)

result = results[0]


# ============================================================
# ANNOTATED IMAGE
# ============================================================

annotated = result.plot()

# Ultralytics returns BGR.
# Convert BGR -> RGB for Streamlit.
annotated_rgb = annotated[:, :, ::-1]


# ============================================================
# INSPECTION RESULT
# ============================================================

st.subheader("Inspection Result")

col1, col2 = st.columns(2)

with col1:

    st.markdown("**Original**")

    st.image(
        image,
        width="stretch",
    )


with col2:

    st.markdown(
        f"**AI Detection — {engine}**"
    )

    st.image(
        annotated_rgb,
        width="stretch",
    )


# ============================================================
# DETECTION DATA
# ============================================================

boxes = result.boxes

total_detections = len(boxes)


class_counts = {
    "panel": 0,
    "cracked": 0,
    "dusty": 0,
    "bird_drop": 0,
}


detections = []


for i in range(total_detections):

    class_id = int(
        boxes.cls[i].item()
    )

    class_name = model.names[
        class_id
    ]

    conf = float(
        boxes.conf[i].item()
    )

    xyxy = boxes.xyxy[
        i
    ].tolist()

    # --------------------------------------------------------
    # Count classes
    # --------------------------------------------------------

    if class_name in class_counts:

        class_counts[
            class_name
        ] += 1

    # --------------------------------------------------------
    # Store detection details
    # --------------------------------------------------------

    detections.append(
        {
            "Detection": i + 1,
            "Class": class_name,
            "Confidence": f"{conf:.3f}",
            "X1": round(xyxy[0], 1),
            "Y1": round(xyxy[1], 1),
            "X2": round(xyxy[2], 1),
            "Y2": round(xyxy[3], 1),
        }
    )


# ============================================================
# INSPECTION SUMMARY
# ============================================================

st.subheader("Inspection Summary")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Panels",
    class_counts["panel"]
)

c2.metric(
    "Cracks",
    class_counts["cracked"]
)

c3.metric(
    "Dusty",
    class_counts["dusty"]
)

c4.metric(
    "Latency",
    f"{elapsed * 1000:.1f} ms"
)

c5.metric(
    "FPS",
    f"{fps:.1f}"
)


# ============================================================
# INSPECTION STATUS
# ============================================================

st.subheader("Inspection Status")


has_defect = (
    class_counts["cracked"] > 0
    or class_counts["dusty"] > 0
    or class_counts["bird_drop"] > 0
)


if has_defect:

    st.warning(
        "⚠️ Potential maintenance issue detected."
    )

else:

    st.success(
        "✅ No defect was detected above "
        "the selected confidence threshold."
    )

    st.caption(
        "This does not guarantee that the panel is "
        "defect-free. The model only detects the "
        "trained visual classes."
    )


# ============================================================
# DETECTION DETAILS
# ============================================================

st.subheader("Detection Details")


if detections:

    st.dataframe(
        detections,
        width="stretch",
        hide_index=True,
    )

else:

    st.info(
        "No detections above the selected "
        "confidence threshold."
    )


# ============================================================
# OPTIONAL ENGINE INFORMATION
# ============================================================

with st.expander("Model / Benchmark Information"):

    if engine == "PyTorch":

        st.write(
            "PyTorch B1 is the primary accuracy model."
        )

        st.write(
            "Held-out test mAP50: **0.7483**"
        )

        st.write(
            "Held-out test mAP50-95: **0.5577**"
        )

        st.write(
            "Model file: **best.pt**"
        )

    else:

        st.write(
            "ONNX B1 is the edge deployment model."
        )

        st.write(
            "Held-out test mAP50: **0.7049**"
        )

        st.write(
            "Held-out test mAP50-95: **0.5238**"
        )

        st.write(
            "Model file: **best.onnx**"
        )

        st.write(
            "Kaggle Intel Xeon reference benchmark: "
            "**~19.7 FPS average**"
        )

        st.write(
            "ONNX model size: **9.35 MB**"
        )