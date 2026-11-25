
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import pandas as pd

from PIL import Image
from pathlib import Path
from datetime import datetime

# =====================================================
# 1. KONFIGURASI
# =====================================================

st.set_page_config(
    page_title="LeafVision AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "leafvision_cnn_augmentation.keras"
CLASS_PATH = BASE_DIR / "metadata" / "class_names.json"

IMAGE_SIZE = (160, 160)

# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #F5F8F5;
}

[data-testid="stSidebar"] {
    background: #143629;
}

[data-testid="stSidebar"] * {
    color: #F1F7F2;
}

.main-title {
    color: #173D2D;
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    color: #718477;
    font-size: 15px;
    margin-bottom: 25px;
}

.hero {
    background: linear-gradient(120deg, #143D2B, #286B49);
    padding: 35px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 34px;
    font-weight: 800;
}

.hero p {
    color: #E1F0E5;
    font-size: 16px;
}

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 17px;
    border: 1px solid #E4ECE5;
    min-height: 120px;
}

.metric-label {
    color: #788A7C;
    font-size: 14px;
}

.metric-value {
    color: #173D2D;
    font-size: 27px;
    font-weight: 800;
}

.result-card {
    background: #E5F3E9;
    padding: 22px;
    border-radius: 18px;
    border-left: 5px solid #27834F;
    margin-top: 15px;
}

.info-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #E4ECE5;
    margin-bottom: 12px;
}

div.stButton > button {
    background: #246B43;
    color: white;
    border: none;
    border-radius: 11px;
    font-weight: 700;
    padding: 10px 18px;
}

div.stButton > button:hover {
    background: #174B30;
    color: white;
}

[data-testid="stFileUploader"] {
    background: white;
    border: 1px dashed #83A88D;
    border-radius: 15px;
    padding: 15px;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD MODEL DAN CLASS NAMES
# =====================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


@st.cache_data
def load_classes():
    if not CLASS_PATH.exists():
        raise FileNotFoundError(
            f"class_names.json tidak ditemukan: {CLASS_PATH}"
        )

    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        classes = json.load(file)

    if not isinstance(classes, list) or len(classes) != 15:
        raise ValueError(
            "class_names.json harus berisi daftar 15 kelas."
        )

    return classes


try:
    model = load_model()
    class_names = load_classes()

    if model.output_shape[-1] != len(class_names):
        st.error(
            "Jumlah output model tidak sama dengan jumlah kelas."
        )
        st.stop()

except Exception as error:
    st.error(f"Gagal memuat model atau kelas: {error}")
    st.stop()


# =====================================================
# 4. INFORMASI PENYAKIT
# =====================================================

DISEASE_INFO = {
    "bacterial_spot": {
        "name": "Bacterial Spot",
        "cause": "Infeksi bakteri pada jaringan daun.",
        "symptoms": "Bercak kecil gelap pada daun, terkadang dikelilingi area kekuningan.",
        "handling": "Gunakan bibit sehat, hindari penyiraman dari atas daun, dan buang bagian tanaman yang terinfeksi."
    },
    "early_blight": {
        "name": "Early Blight",
        "cause": "Penyakit yang umumnya disebabkan jamur Alternaria.",
        "symptoms": "Bercak cokelat dengan pola lingkaran konsentris.",
        "handling": "Buang daun terinfeksi, jaga sirkulasi udara, dan hindari kelembapan berlebih."
    },
    "late_blight": {
        "name": "Late Blight",
        "cause": "Penyakit akibat Phytophthora infestans.",
        "symptoms": "Bercak cokelat gelap yang dapat berkembang cepat pada kondisi lembap.",
        "handling": "Pisahkan tanaman terinfeksi dan gunakan metode pengendalian sesuai rekomendasi pertanian."
    },
    "leaf_mold": {
        "name": "Leaf Mold",
        "cause": "Infeksi jamur Passalora fulva.",
        "symptoms": "Bercak kekuningan pada bagian atas daun dan pertumbuhan seperti beludru di bagian bawah.",
        "handling": "Tingkatkan ventilasi dan kurangi kelembapan di sekitar tanaman."
    },
    "septoria_leaf_spot": {
        "name": "Septoria Leaf Spot",
        "cause": "Infeksi jamur Septoria lycopersici.",
        "symptoms": "Bercak kecil bulat dengan bagian tengah abu-abu atau kecokelatan.",
        "handling": "Buang daun terinfeksi dan hindari percikan air dari tanah."
    },
    "spider_mites": {
        "name": "Spider Mites",
        "cause": "Serangan tungau yang mengisap cairan daun.",
        "symptoms": "Daun berbintik kekuningan, kusam, dan terkadang terdapat jaring halus.",
        "handling": "Periksa bagian bawah daun dan lakukan pengendalian tungau yang sesuai."
    },
    "target_spot": {
        "name": "Target Spot",
        "cause": "Penyakit jamur Corynespora cassiicola.",
        "symptoms": "Bercak cokelat yang dapat membentuk pola lingkaran.",
        "handling": "Jaga jarak tanam dan kurangi kelembapan daun."
    },
    "yellow_leaf_curl": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "cause": "Infeksi virus yang dapat ditularkan oleh kutu kebul.",
        "symptoms": "Daun menggulung, menguning, dan pertumbuhan tanaman terhambat.",
        "handling": "Kendalikan kutu kebul dan singkirkan tanaman terinfeksi sesuai praktik budidaya."
    },
    "tomato_mosaic": {
        "name": "Tomato Mosaic Virus",
        "cause": "Infeksi virus mosaik pada tanaman tomat.",
        "symptoms": "Pola warna hijau muda dan hijau tua menyerupai mosaik.",
        "handling": "Gunakan benih sehat, sanitasi peralatan, dan hindari kontak dengan tanaman terinfeksi."
    }
}


def get_disease_info(label):
    normalized = label.lower()

    if "healthy" in normalized:
        return {
            "name": "Healthy",
            "cause": "Model mengklasifikasikan daun sebagai sehat.",
            "symptoms": "Tidak ditemukan indikasi penyakit berdasarkan kelas prediksi.",
            "handling": "Pertahankan perawatan tanaman dan lakukan pemantauan rutin."
        }

    # Pencocokan nama kelas yang spesifik
    if "yellowleaf_curl" in normalized:
        key = "yellow_leaf_curl"
    elif "mosaic_virus" in normalized:
        key = "tomato_mosaic"
    elif "spider_mites" in normalized:
        key = "spider_mites"
    elif "septoria" in normalized:
        key = "septoria_leaf_spot"
    elif "bacterial_spot" in normalized:
        key = "bacterial_spot"
    elif "early_blight" in normalized:
        key = "early_blight"
    elif "late_blight" in normalized:
        key = "late_blight"
    elif "leaf_mold" in normalized:
        key = "leaf_mold"
    elif "target_spot" in normalized:
        key = "target_spot"
    else:
        key = None

    if key and key in DISEASE_INFO:
        return DISEASE_INFO[key]

    return {
        "name": label.replace("_", " "),
        "cause": "Informasi penyebab belum tersedia.",
        "symptoms": "Periksa kondisi daun secara langsung.",
        "handling": "Lakukan pemeriksaan lanjutan dengan ahli pertanian."
    }


# =====================================================
# 5. FUNGSI PREDIKSI
# =====================================================

def predict_image(image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    top_indices = np.argsort(probabilities)[::-1][:3]

    return {
        "label": class_names[int(top_indices[0])],
        "confidence": float(probabilities[int(top_indices[0])]) * 100,
        "top_indices": [int(index) for index in top_indices],
        "probabilities": probabilities.tolist()
    }


# =====================================================
# 6. SESSION STATE
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "result" not in st.session_state:
    st.session_state.result = None

if "navigation" not in st.session_state:
    st.session_state.navigation = "Dashboard"


def go_to_detection():
    st.session_state.navigation = "Deteksi Penyakit"


# =====================================================
# 7. SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("# 🌿 LeafVision AI")
    st.caption("SMART PLANT DISEASE DETECTION")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Deteksi Penyakit",
            "Informasi Penyakit",
            "Riwayat Deteksi",
            "Tentang Model"
        ],
        key="navigation"
    )

    st.divider()

    st.caption("LeafVision AI v1.0")
    st.caption("CNN-Based Plant Disease Classification")
    st.caption("GEMASTIK 2026")


# =====================================================
# 8. DASHBOARD
# =====================================================

if page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>Kenali Kesehatan Tanamanmu 🌱</h1>
        <p>
        Deteksi kemungkinan penyakit daun tanaman menggunakan
        teknologi Artificial Intelligence dan Convolutional Neural Network.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Jumlah Kelas</div>
            <div class="metric-value">15 Kelas</div>
            <p>Tanaman dan penyakit</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Model AI</div>
            <div class="metric-value">CNN</div>
            <p>Dengan data augmentation</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Test Accuracy</div>
            <div class="metric-value">96.54%</div>
            <p>Hasil evaluasi pengujian</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.markdown("### Mulai Deteksi")

    st.write(
        "Unggah foto daun tanaman untuk mendapatkan hasil "
        "klasifikasi dan informasi kemungkinan penyakit."
    )

    st.button(
        "Mulai Deteksi Sekarang →",
        on_click=go_to_detection
    )


# =====================================================
# 9. DETEKSI PENYAKIT
# =====================================================

elif page == "Deteksi Penyakit":

    st.markdown(
        '<div class="main-title">Deteksi Penyakit</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Unggah gambar daun untuk memulai analisis AI.</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload foto daun tanaman",
        type=["jpg", "jpeg", "png"],
        key="leaf_upload"
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(
                image,
                caption="Foto daun yang diunggah",
                use_container_width=True
            )

        with col2:
            st.markdown("### Siap untuk dianalisis?")
            st.write(
                "Pastikan bagian daun terlihat jelas "
                "dan gambar tidak terlalu buram."
            )

            if st.button(
                "🔍 Analisis Gambar",
                type="primary",
                use_container_width=True
            ):

                with st.spinner("AI sedang menganalisis gambar..."):

                    try:
                        prediction = predict_image(image)

                        result = {
                            **prediction,
                            "image_name": uploaded_file.name,
                            "time": datetime.now().strftime(
                                "%d-%m-%Y %H:%M:%S"
                            )
                        }

                        st.session_state.result = result
                        st.session_state.history.insert(0, result)

                    except Exception as error:
                        st.error(f"Prediksi gagal: {error}")

    # Tampilkan hasil yang tersimpan
    if st.session_state.result is not None:

        result = st.session_state.result
        info = get_disease_info(result["label"])

        st.divider()
        st.markdown("## Hasil Analisis")

        st.markdown(
            f"""
            <div class="result-card">
                <div style="font-size:14px;color:#477557;">
                    HASIL KLASIFIKASI
                </div>
                <h2 style="color:#174B30;">
                    {info['name']}
                </h2>
                <h2 style="color:#246B43;">
                    {result['confidence']:.2f}%
                </h2>
                <div style="color:#477557;">
                    Confidence Score
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### Top 3 Prediksi")

        for index in result["top_indices"]:

            label = class_names[index]
            score = result["probabilities"][index] * 100

            st.write(f"**{label.replace('_', ' ')}**")

            st.progress(
                min(max(score / 100, 0.0), 1.0)
            )

            st.caption(f"{score:.2f}%")

        st.markdown("### Informasi Penyakit")

        st.info(f"**Penyebab:** {info['cause']}")
        st.write(f"**Gejala:** {info['symptoms']}")
        st.write(f"**Saran penanganan:** {info['handling']}")

        st.warning(
            "Hasil merupakan prediksi AI, bukan diagnosis pasti. "
            "Lakukan pemeriksaan langsung untuk memastikan kondisi tanaman."
        )


# =====================================================
# 10. INFORMASI PENYAKIT
# =====================================================

elif page == "Informasi Penyakit":

    st.markdown(
        '<div class="main-title">Informasi Penyakit</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Kenali gejala dan penanganan tanaman.</div>',
        unsafe_allow_html=True
    )

    for label in class_names:

        info = get_disease_info(label)

        with st.expander(label.replace("_", " ")):

            st.write(f"**Nama:** {info['name']}")
            st.write(f"**Penyebab:** {info['cause']}")
            st.write(f"**Gejala:** {info['symptoms']}")
            st.write(f"**Penanganan:** {info['handling']}")


# =====================================================
# 11. RIWAYAT DETEKSI
# =====================================================

elif page == "Riwayat Deteksi":

    st.markdown(
        '<div class="main-title">Riwayat Deteksi</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Daftar hasil analisis selama sesi penggunaan.</div>',
        unsafe_allow_html=True
    )

    if st.session_state.history:

        history_df = pd.DataFrame([
            {
                "Waktu": item["time"],
                "Nama Gambar": item["image_name"],
                "Hasil": item["label"].replace("_", " "),
                "Confidence": f"{item['confidence']:.2f}%"
            }
            for item in st.session_state.history
        ])

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button("Hapus Riwayat"):

            st.session_state.history = []
            st.session_state.result = None

            st.rerun()

    else:
        st.info(
            "Belum ada riwayat deteksi. "
            "Silakan lakukan deteksi terlebih dahulu."
        )


# =====================================================
# 12. TENTANG MODEL
# =====================================================

elif page == "Tentang Model":

    st.markdown(
        '<div class="main-title">Tentang LeafVision AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Teknologi di balik sistem klasifikasi penyakit tanaman.</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Convolutional Neural Network")

    st.write(
        "LeafVision AI menggunakan arsitektur Convolutional Neural Network "
        "untuk mengklasifikasikan gambar daun tanaman ke dalam 15 kelas."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <b>Model</b>
            <h2 style="color:#246B43;">CNN</h2>
            <p>Dengan data augmentation</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <b>Test Accuracy</b>
            <h2 style="color:#246B43;">96.54%</h2>
            <p>Hasil evaluasi model</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Spesifikasi Model")

    st.write("- Input gambar: 160 × 160 piksel")
    st.write("- Batch size training: 16")
    st.write("- Jumlah kelas: 15")
    st.write("- Optimizer: Adam")
    st.write("- Loss: Sparse Categorical Crossentropy")
    st.write("- Augmentation: Flip, Rotation, Zoom, Contrast")

    st.markdown("### Daftar Kelas")

    for number, name in enumerate(class_names, start=1):
        st.write(f"{number}. {name.replace('_', ' ')}")


# =====================================================
# 13. FOOTER
# =====================================================

st.divider()

st.markdown("""
<div style="
    text-align:center;
    color:#829187;
    padding:15px;
">
    <b>🌿 LeafVision AI</b><br>
    AI-Based Plant Disease Detection System<br>
    GEMASTIK 2026
</div>
""", unsafe_allow_html=True)
