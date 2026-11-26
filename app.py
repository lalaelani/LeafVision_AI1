import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# =========================================================
# 1. KONFIGURASI APLIKASI
# =========================================================
st.set_page_config(
    page_title="LeafVision AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "leafvision_cnn_38_classes.keras"

# class_names1.json adalah file daftar kelas terbaru.
# class_names.json tetap disediakan sebagai fallback HANYA jika isinya 38 kelas.
CLASS_PATHS = [
    BASE_DIR / "metadata" / "class_names1.json",
    BASE_DIR / "metadata" / "class_names.json",
]

IMAGE_SIZE = (160, 160)
NUM_CLASSES = 38
MODEL_ACCURACY = "92,09%"
CONFIDENCE_WARNING = 55.0
MARGIN_WARNING = 8.0

# =========================================================
# 2. TAMPILAN
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #F5F8F5; }
    [data-testid="stSidebar"] { background: #143629; }
    [data-testid="stSidebar"] * { color: #F1F7F2; }
    .main-title { color: #173D2D; font-size: 38px; font-weight: 800; margin-bottom: 5px; }
    .subtitle { color: #718477; font-size: 15px; margin-bottom: 25px; }
    .hero {
        background: linear-gradient(120deg, #143D2B, #286B49);
        padding: 35px; border-radius: 22px; color: white; margin-bottom: 25px;
    }
    .hero h1 { color: white; font-size: 34px; font-weight: 800; }
    .hero p { color: #E1F0E5; font-size: 16px; }
    .metric-card {
        background: white; padding: 22px; border-radius: 17px;
        border: 1px solid #E4ECE5; min-height: 120px;
    }
    .metric-label { color: #788A7C; font-size: 14px; }
    .metric-value { color: #173D2D; font-size: 27px; font-weight: 800; }
    .result-card {
        background: #E5F3E9; padding: 22px; border-radius: 18px;
        border-left: 5px solid #27834F; margin-top: 15px;
    }
    .info-card {
        background: white; padding: 20px; border-radius: 16px;
        border: 1px solid #E4ECE5; margin-bottom: 12px;
    }
    div.stButton > button {
        background: #246B43; color: white; border: none;
        border-radius: 11px; font-weight: 700; padding: 10px 18px;
    }
    div.stButton > button:hover { background: #174B30; color: white; }
    [data-testid="stFileUploader"] {
        background: white; border: 1px dashed #83A88D;
        border-radius: 15px; padding: 15px;
    }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 3. MEMUAT MODEL DAN NAMA KELAS
# =========================================================
def normalize_class_data(data):
    """Ubah JSON list/dictionary menjadi list nama kelas tanpa mengubah urutannya."""
    if isinstance(data, list):
        classes = data
    elif isinstance(data, dict):
        # Format umum: {"0": "kelas_a", "1": "kelas_b", ...}
        if all(str(i) in data for i in range(len(data))):
            classes = [data[str(i)] for i in range(len(data))]
        elif all(isinstance(v, str) for v in data.values()):
            # Pertahankan urutan key pada JSON jika dictionary bukan berindeks.
            classes = list(data.values())
        else:
            raise ValueError("Format dictionary pada JSON nama kelas tidak dikenali.")
    else:
        raise ValueError("Isi JSON nama kelas harus berupa list atau dictionary.")

    classes = [str(item).strip() for item in classes]

    if len(classes) != NUM_CLASSES:
        raise ValueError(
            f"File memuat {len(classes)} kelas, sedangkan model membutuhkan {NUM_CLASSES}."
        )
    if any(not item for item in classes):
        raise ValueError("Ada nama kelas kosong pada file JSON.")
    if len(set(classes)) != NUM_CLASSES:
        raise ValueError("Ada nama kelas duplikat pada file JSON.")

    return classes


@st.cache_data
def load_classes():
    errors = []
    for path in CLASS_PATHS:
        if not path.exists():
            errors.append(f"{path.name}: file tidak ditemukan")
            continue

        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            classes = normalize_class_data(data)
            return classes, path.name
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    raise RuntimeError(
        "Tidak ditemukan file daftar kelas yang valid (harus tepat 38 kelas). "
        + " | ".join(errors)
    )


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}. "
            "Pastikan file leafvision_cnn_38_classes.keras berada di folder models."
        )

    # compile=False diperlukan karena aplikasi hanya melakukan inferensi.
    loaded_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    if len(loaded_model.input_shape) != 4:
        raise ValueError(f"Bentuk input model tidak dikenali: {loaded_model.input_shape}")

    if tuple(loaded_model.input_shape[1:3]) != IMAGE_SIZE:
        raise ValueError(
            f"Ukuran input model {loaded_model.input_shape[1:3]}, "
            f"tetapi aplikasi disetel ke {IMAGE_SIZE}."
        )

    if loaded_model.input_shape[-1] not in (3, None):
