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
        raise ValueError("Model tidak menggunakan input gambar RGB 3 kanal.")

    if loaded_model.output_shape[-1] != NUM_CLASSES:
        raise ValueError(
            f"Model menghasilkan {loaded_model.output_shape[-1]} output, "
            f"bukan {NUM_CLASSES}."
        )

    return loaded_model


try:
    model = load_model()
    class_names, class_file_used = load_classes()

    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            f"Output model ({model.output_shape[-1]}) tidak sama dengan "
            f"jumlah nama kelas ({len(class_names)})."
        )

except Exception as error:
    st.error(f"Gagal memuat model atau daftar kelas: {error}")
    st.info(
        "Struktur yang dibutuhkan: models/leafvision_cnn_38_classes.keras "
        "dan metadata/class_names1.json (38 kelas)."
    )
    st.stop()

# =========================================================
# 4. INFORMASI PENYAKIT
# =========================================================
DISEASE_INFO = {
    "apple_scab": ("Apple Scab", "Penyakit jamur pada tanaman apel.", "Bercak gelap atau zaitun pada daun.", "Buang bagian terinfeksi dan jaga kebersihan kebun."),
    "black_rot": ("Black Rot", "Penyakit jamur yang dapat menyerang tanaman buah.", "Bercak nekrotik atau jaringan daun mengering.", "Pangkas bagian terinfeksi dan hindari kelembapan berlebih."),
    "cedar_apple_rust": ("Cedar Apple Rust", "Penyakit karat pada apel.", "Bercak kekuningan hingga jingga pada daun.", "Pantau gejala dan lakukan pengelolaan tanaman sesuai rekomendasi setempat."),
    "powdery_mildew": ("Powdery Mildew", "Penyakit jamur embun tepung.", "Lapisan putih seperti tepung pada permukaan daun.", "Tingkatkan sirkulasi udara dan hindari kondisi terlalu lembap."),
    "cercospora": ("Cercospora / Gray Leaf Spot", "Penyakit bercak daun pada jagung.", "Bercak memanjang atau tidak beraturan pada daun.", "Gunakan benih sehat, rotasi tanaman, dan kelola sisa tanaman."),
    "common_rust": ("Common Rust", "Penyakit karat pada jagung.", "Pustula kecil berwarna cokelat kemerahan.", "Gunakan varietas tahan dan pantau gejala."),
    "northern_leaf_blight": ("Northern Leaf Blight", "Penyakit hawar daun pada jagung.", "Lesi memanjang berwarna cokelat keabu-abuan.", "Gunakan varietas tahan dan lakukan rotasi tanaman."),
    "esca": ("Esca / Black Measles", "Gangguan penyakit kompleks pada tanaman anggur.", "Perubahan warna dan nekrosis pada daun.", "Pangkas bagian sakit dengan alat bersih dan minta pemeriksaan ahli."),
    "leaf_blight": ("Leaf Blight", "Penyakit hawar daun.", "Area daun menguning, kecokelatan, atau mengering.", "Buang jaringan rusak dan jaga sirkulasi udara."),
    "huanglongbing": ("Huanglongbing / Citrus Greening", "Penyakit sistemik pada tanaman jeruk.", "Daun dapat belang tidak simetris dan tanaman tampak merana.", "Konfirmasi melalui pemeriksaan pertanian dan kendalikan vektor sesuai panduan."),
    "bacterial_spot": ("Bacterial Spot", "Penyakit bercak daun yang disebabkan bakteri.", "Bercak kecil gelap, kadang dikelilingi area kekuningan.", "Gunakan bibit sehat, hindari penyiraman dari atas daun, dan sanitasi alat."),
    "early_blight": ("Early Blight", "Penyakit yang umumnya disebabkan jamur Alternaria.", "Bercak cokelat yang dapat membentuk lingkaran konsentris.", "Buang daun terinfeksi, jaga sirkulasi udara, dan hindari kelembapan berlebih."),
    "late_blight": ("Late Blight", "Penyakit akibat Phytophthora infestans.", "Bercak cokelat gelap yang dapat berkembang cepat pada kondisi lembap.", "Pisahkan tanaman bergejala dan minta rekomendasi pengendalian."),
    "leaf_mold": ("Leaf Mold", "Penyakit jamur pada daun, umum pada tomat.", "Bercak kekuningan di atas daun dan pertumbuhan seperti beludru di bawah daun.", "Tingkatkan ventilasi dan kurangi kelembapan."),
    "septoria": ("Septoria Leaf Spot", "Penyakit jamur Septoria.", "Bercak kecil bulat dengan bagian tengah abu-abu atau kecokelatan.", "Buang daun terinfeksi dan hindari percikan air dari tanah."),
    "spider_mites": ("Spider Mites", "Serangan tungau yang mengisap cairan daun.", "Daun berbintik kekuningan dan terkadang terdapat jaring halus.", "Periksa bagian bawah daun dan lakukan pengendalian tungau yang sesuai."),
    "target_spot": ("Target Spot", "Penyakit bercak daun.", "Bercak cokelat dengan pola lingkaran atau sasaran.", "Jaga jarak tanam, sirkulasi udara, dan kebersihan daun."),
    "yellow_leaf_curl": ("Tomato Yellow Leaf Curl Virus", "Penyakit virus yang dapat ditularkan kutu kebul.", "Daun menggulung, menguning, dan pertumbuhan terhambat.", "Kendalikan kutu kebul dan singkirkan tanaman terinfeksi sesuai praktik budidaya."),
    "tomato_mosaic": ("Tomato Mosaic Virus", "Penyakit virus mosaik pada tomat.", "Pola warna hijau muda dan hijau tua menyerupai mosaik.", "Gunakan benih sehat dan sanitasi peralatan."),
    "leaf_scorch": ("Leaf Scorch", "Gangguan yang menyebabkan jaringan daun mengalami nekrosis.", "Tepi atau bagian daun tampak kering dan kecokelatan.", "Periksa kondisi air, akar, dan faktor lingkungan."),
}

def get_disease_info(label):
    normalized = label.lower()

    if "healthy" in normalized:
        return {
            "name": "Healthy / Sehat",
            "cause": "Model mengklasifikasikan gambar ke kelas daun sehat.",
            "symptoms": "Pola gambar paling sesuai dengan kelas sehat pada dataset.",
            "handling": "Pertahankan perawatan dan pemantauan rutin. Hasil AI bukan jaminan tanaman bebas penyakit.",
        }

    key_map = [
        ("yellow_leaf_curl", "yellow_leaf_curl"),
        ("yellow_leaf_curl_virus", "yellow_leaf_curl"),
        ("yellowleaf", "yellow_leaf_curl"),
        ("mosaic_virus", "tomato_mosaic"),
        ("tomato_mosaic", "tomato_mosaic"),
        ("spider_mites", "spider_mites"),
        ("septoria", "septoria"),
        ("bacterial_spot", "bacterial_spot"),
        ("early_blight", "early_blight"),
        ("late_blight", "late_blight"),
        ("leaf_mold", "leaf_mold"),
        ("target_spot", "target_spot"),
        ("powdery_mildew", "powdery_mildew"),
        ("apple_scab", "apple_scab"),
        ("black_rot", "black_rot"),
        ("cedar_apple_rust", "cedar_apple_rust"),
        ("gray_leaf_spot", "cercospora"),
        ("cercospora", "cercospora"),
        ("common_rust", "common_rust"),
        ("northern_leaf_blight", "northern_leaf_blight"),
        ("esca", "esca"),
        ("leaf_blight", "leaf_blight"),
        ("huanglongbing", "huanglongbing"),
        ("citrus_greening", "huanglongbing"),
        ("leaf_scorch", "leaf_scorch"),
    ]

    for token, key in key_map:
        if token in normalized and key in DISEASE_INFO:
            name, cause, symptoms, handling = DISEASE_INFO[key]
            return {
                "name": name,
                "cause": cause,
                "symptoms": symptoms,
                "handling": handling,
            }

    return {
        "name": readable_class(label),
        "cause": "Informasi khusus untuk kelas ini belum tersedia.",
        "symptoms": "Periksa gejala pada daun dan tanaman secara langsung.",
        "handling": "Jangan menentukan perlakuan hanya dari prediksi AI; lakukan pemeriksaan lanjutan.",
    }


def readable_class(label):
    plant, separator, condition = label.partition("___")
    if not separator:
        return label.replace("_", " ").replace(",", "")
    plant = plant.replace("_", " ").replace(",", "")
    condition = condition.replace("_", " ").replace(",", "")
    return f"{plant} — {condition}"


# =========================================================
# 5. PREPROCESSING DAN PREDIKSI
# =========================================================
def prepare_image(uploaded_image):
    """
    Preprocessing inferensi:
    EXIF transpose -> RGB -> resize 160x160 -> float32 -> /255.
    Tidak menambahkan crop/filter yang tidak dipakai saat training.
    """
    image = ImageOps.exif_transpose(uploaded_image).convert("RGB")
    original_size = image.size

    resized = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image, image_array, original_size


def image_quality_notes(image, original_size):
    notes = []
    width, height = original_size

    if min(width, height) < 160:
        notes.append("Resolusi foto rendah. Coba ambil foto lebih dekat dan fokus pada daun.")

    if max(width, height) / max(1, min(width, height)) > 2.5:
        notes.append("Foto terlalu memanjang. Usahakan daun menjadi objek utama dan terlihat utuh.")

    gray = np.asarray(image.convert("L").resize((256, 256)), dtype=np.float32)
    sharpness_measure = (
        np.var(gray[1:, :] - gray[:-1, :])
        + np.var(gray[:, 1:] - gray[:, :-1])
    )
    if sharpness_measure < 18:
        notes.append("Foto mungkin buram. Ambil ulang dengan fokus kamera pada permukaan daun.")

    brightness = float(np.mean(gray))
    if brightness < 35:
        notes.append("Foto terlalu gelap. Gunakan pencahayaan yang cukup.")
    elif brightness > 225:
        notes.append("Foto terlalu terang. Hindari pantulan cahaya langsung.")

    return notes


def predict_image(image_array):
    raw_output = model.predict(image_array, verbose=0)
    scores = np.asarray(raw_output[0], dtype=np.float64)

    if scores.ndim != 1 or len(scores) != NUM_CLASSES:
        raise ValueError("Bentuk output prediksi tidak sesuai dengan 38 kelas.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Model menghasilkan nilai prediksi yang tidak valid.")

    # Model menggunakan softmax; normalisasi defensif untuk tampilan top-3.
    if np.any(scores < 0) or not np.isclose(np.sum(scores), 1.0, atol=0.03):
        raise ValueError(
            "Output model bukan distribusi softmax yang diharapkan. "
            "Periksa layer output model."
        )

    top_indices = np.argsort(scores)[::-1][:3]
    first = float(scores[top_indices[0]] * 100)
    second = float(scores[top_indices[1]] * 100)
    margin = first - second

    return {
        "label": class_names[int(top_indices[0])],
        "confidence": first,
        "margin": margin,
        "top_indices": [int(i) for i in top_indices],
        "scores": scores.tolist(),
        "uncertain": first < CONFIDENCE_WARNING or margin < MARGIN_WARNING,
    }


# =========================================================
# 6. SESSION STATE
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None
if "last_upload_id" not in st.session_state:
    st.session_state.last_upload_id = None
if "navigation" not in st.session_state:
    st.session_state.navigation = "Dashboard"


def go_to_detection():
    st.session_state.navigation = "Deteksi Penyakit"


# =========================================================
# 7. SIDEBAR
# =========================================================
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
            "Tentang Model",
        ],
        key="navigation",
    )

    st.divider()
    st.caption("LeafVision AI | CNN 38 Classes")
    st.caption("AI-Based Plant Disease Classification")


# =========================================================
# 8. DASHBOARD
# =========================================================
if page == "Dashboard":
    st.markdown(
        """
        <div class="hero">
            <h1>Kenali Kesehatan Tanamanmu 🌱</h1>
            <p>Unggah foto daun untuk memperoleh prediksi kelas tanaman dan kemungkinan
            penyakit menggunakan teknologi Artificial Intelligence dan CNN.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-label">Jumlah Kelas</div>
            <div class="metric-value">{len(class_names)} Kelas</div>
            <p>Tanaman dan kondisi daun</p></div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """<div class="metric-card"><div class="metric-label">Model AI</div>
            <div class="metric-value">CNN</div>
            <p>Convolutional Neural Network</p></div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-label">Test Accuracy</div>
            <div class="metric-value">{MODEL_ACCURACY}</div>
            <p>Evaluasi model 38 kelas</p></div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("### Mulai Deteksi")
    st.write(
        "Gunakan foto satu daun yang terlihat jelas, tidak buram, "
        "memiliki pencahayaan cukup, dan memenuhi sebagian besar area foto."
    )
    st.button("Mulai Deteksi Sekarang →", on_click=go_to_detection)


# =========================================================
# 9. DETEKSI PENYAKIT
# =========================================================
elif page == "Deteksi Penyakit":
    st.markdown('<div class="main-title">Deteksi Penyakit</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Unggah foto daun yang jelas untuk memulai analisis AI.</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Tips foto: gunakan satu daun sebagai objek utama, dekatkan kamera secukupnya, "
        "pastikan fokus tajam, cahaya merata, dan hindari latar belakang ramai. "
        "Model mengenali pola dari dataset, bukan melakukan diagnosis laboratorium."
    )

    uploaded_file = st.file_uploader(
        "Upload foto daun tanaman",
        type=["jpg", "jpeg", "png"],
        key="leaf_upload",
    )

    if uploaded_file is not None:
        try:
            raw_image = Image.open(uploaded_file)
            image, image_array, original_size = prepare_image(raw_image)

            # Jika pengguna mengganti foto, hasil foto sebelumnya tidak ditampilkan lagi.
            upload_id = f"{uploaded_file.name}-{uploaded_file.size}"
            if st.session_state.last_upload_id != upload_id:
                st.session_state.result = None
                st.session_state.last_upload_id = upload_id

            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image, caption="Foto daun yang diunggah", use_container_width=True)
                st.caption(f"Resolusi asli: {original_size[0]} × {original_size[1]} piksel")

            with col2:
                st.markdown("### Pemeriksaan Foto")
                notes = image_quality_notes(image, original_size)
                if notes:
                    for note in notes:
                        st.warning(note)
                else:
                    st.success("Pemeriksaan dasar gambar cukup baik. Pastikan objek utama memang daun.")

                if st.button(
                    "🔍 Analisis Gambar",
                    type="primary",
                    use_container_width=True,
                ):
                    with st.spinner("AI sedang menganalisis gambar..."):
                        prediction = predict_image(image_array)
                        info = get_disease_info(prediction["label"])

                        result = {
                            **prediction,
                            "image_name": uploaded_file.name,
                            "time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                            "info_name": info["name"],
                        }
                        st.session_state.result = result
                        st.session_state.history.insert(0, result)
                        st.rerun()

        except Exception as error:
            st.error(f"Gambar tidak dapat diproses: {error}")

    if st.session_state.result is not None:
        result = st.session_state.result
        info = get_disease_info(result["label"])

        st.divider()
        st.markdown("## Hasil Analisis")

        if result["uncertain"]:
            st.warning(
                "Prediksi belum cukup meyakinkan. Coba foto lain yang lebih jelas. "
                "Jangan gunakan hasil ini sebagai satu-satunya dasar diagnosis."
            )
        else:
            st.success("Model menemukan satu kelas yang paling sesuai pada gambar.")

        st.markdown(
            f"""
            <div class="result-card">
                <div style="font-size:14px;color:#477557;">PREDIKSI KELAS TERATAS</div>
                <h2 style="color:#174B30;">{info['name']}</h2>
                <h2 style="color:#246B43;">{result['confidence']:.2f}%</h2>
                <div style="color:#477557;">
                    Skor keluaran model, bukan probabilitas diagnosis klinis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Top 3 Prediksi")
        for index in result["top_indices"]:
            label = class_names[index]
            score = result["scores"][index] * 100
            st.write(f"**{readable_class(label)}**")
            st.progress(float(np.clip(score / 100, 0.0, 1.0)))
            st.caption(f"{score:.2f}%")

        st.caption(f"Selisih skor peringkat pertama dan kedua: {result['margin']:.2f} poin persentase.")

        st.markdown("### Informasi Kelas Teratas")
        st.info(f"**Penyebab/keterangan:** {info['cause']}")
        st.write(f"**Gejala umum:** {info['symptoms']}")
        st.write(f"**Saran:** {info['handling']}")

        st.warning(
            "Hasil merupakan prediksi model berdasarkan pola gambar dalam dataset. "
            "Model dapat keliru, terutama pada foto dengan latar ramai, objek selain daun, "
            "atau kondisi yang tidak termasuk kelas pelatihan."
        )


# =========================================================
# 10. INFORMASI PENYAKIT
# =========================================================
elif page == "Informasi Penyakit":
    st.markdown('<div class="main-title">Informasi Penyakit</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Daftar kelas yang dapat dikenali model.</div>',
        unsafe_allow_html=True,
    )

    search_text = st.text_input("Cari tanaman atau kondisi daun", placeholder="Contoh: tomato, apple, healthy")
    filtered_classes = [
        label for label in class_names
        if search_text.lower() in readable_class(label).lower()
    ]

    if not filtered_classes:
        st.info("Tidak ada kelas yang cocok dengan pencarian.")
    else:
        for label in filtered_classes:
            info = get_disease_info(label)
            with st.expander(readable_class(label)):
                st.write(f"**Nama kelas:** {readable_class(label)}")
                st.write(f"**Penyebab/keterangan:** {info['cause']}")
                st.write(f"**Gejala umum:** {info['symptoms']}")
                st.write(f"**Saran penanganan:** {info['handling']}")


# =========================================================
# 11. RIWAYAT DETEKSI
# =========================================================
elif page == "Riwayat Deteksi":
    st.markdown('<div class="main-title">Riwayat Deteksi</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Daftar hasil analisis selama sesi penggunaan.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.history:
        history_df = pd.DataFrame(
            [
                {
                    "Waktu": item["time"],
                    "Nama Gambar": item["image_name"],
                    "Hasil": readable_class(item["label"]),
                    "Skor": f"{item['confidence']:.2f}%",
                    "Catatan": "Perlu pemeriksaan ulang" if item["uncertain"] else "Prediksi teratas",
                }
                for item in st.session_state.history
            ]
        )
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        if st.button("Hapus Riwayat"):
            st.session_state.history = []
            st.session_state.result = None
            st.rerun()
    else:
        st.info("Belum ada riwayat deteksi. Silakan lakukan deteksi terlebih dahulu.")


# =========================================================
# 12. TENTANG MODEL
# =========================================================
elif page == "Tentang Model":
    st.markdown('<div class="main-title">Tentang LeafVision AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Teknologi di balik sistem klasifikasi penyakit tanaman.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Convolutional Neural Network")
    st.write(
        "LeafVision AI menggunakan arsitektur Convolutional Neural Network (CNN) "
        "untuk mengklasifikasikan gambar ke dalam 38 kelas tanaman dan kondisi daun."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """<div class="info-card"><b>Model</b><h2 style="color:#246B43;">CNN</h2>
            <p>Dengan data augmentation</p></div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="info-card"><b>Test Accuracy</b>
            <h2 style="color:#246B43;">{MODEL_ACCURACY}</h2>
            <p>Hasil evaluasi model 38 kelas</p></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("### Spesifikasi Model")
    st.write("- Ukuran input: 160 × 160 piksel")
    st.write("- Jumlah kelas: 38")
    st.write("- Optimizer saat training: Adam")
    st.write("- Loss: Sparse Categorical Crossentropy")
    st.write("- Augmentation saat training: Flip, Rotation, Zoom, Contrast")
    st.write("- Normalisasi input: nilai piksel dibagi 255")
    st.write("- Test accuracy: 92,09%")
    st.caption(
        "Daftar kelas dimuat dari file JSON. File yang digunakan: "
        f"{class_file_used}. Urutan kelas pada file tersebut harus sama dengan urutan "
        "kelas saat model dilatih. Performa pada foto lapangan dapat berbeda."
    )

    st.markdown("### Daftar Kelas")
    for number, name in enumerate(class_names, start=1):
        st.write(f"{number}. {readable_class(name)}")


# =========================================================
# 13. FOOTER
# =========================================================
st.divider()
st.markdown(
    """
    <div style="text-align:center;color:#829187;padding:15px;">
        <b>🌿 LeafVision AI</b><br>
        AI-Based Plant Disease Detection System
    </div>
    """,
    unsafe_allow_html=True,
)
