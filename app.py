import streamlit as st
import tensorflow as tf
import numpy as np
import json
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
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
MODEL_PATH = BASE_DIR / "models" / "leafvision_cnn_38_classes.keras"
CLASS_PATH = BASE_DIR / "metadata" / "class_names.json"

IMAGE_SIZE = (160, 160)
NUM_CLASSES = 38
MODEL_ACCURACY = "92,09%"
CONFIDENCE_WARNING = 55.0

# =====================================================
# 2. CUSTOM CSS
# =====================================================
st.markdown("""
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
""", unsafe_allow_html=True)

# =====================================================
# 3. LOAD MODEL DAN NAMA KELAS
# =====================================================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model 38 kelas tidak ditemukan di: {MODEL_PATH}. "
            "Pastikan file leafvision_cnn_38_classes.keras sudah diunggah ke folder models."
        )
    loaded = tf.keras.models.load_model(MODEL_PATH, compile=False)
    if loaded.input_shape[1:3] != IMAGE_SIZE:
        raise ValueError(
            f"Ukuran input model adalah {loaded.input_shape[1:3]}, "
            f"sedangkan aplikasi menggunakan {IMAGE_SIZE}."
        )
    return loaded


@st.cache_data
def load_classes():
    if not CLASS_PATH.exists():
        raise FileNotFoundError(f"class_names.json tidak ditemukan: {CLASS_PATH}")

    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        classes = json.load(file)

    # Mendukung JSON berbentuk list maupun dictionary berindeks.
    if isinstance(classes, dict):
        try:
            classes = [classes[str(i)] for i in range(len(classes))]
        except (KeyError, TypeError):
            classes = list(classes.values())

    if not isinstance(classes, list) or len(classes) != NUM_CLASSES:
        raise ValueError(
            f"class_names.json harus berisi tepat {NUM_CLASSES} nama kelas; "
            f"yang terbaca: {len(classes) if isinstance(classes, list) else 'format tidak valid'}."
        )

    return [str(name) for name in classes]


try:
    model = load_model()
    class_names = load_classes()
    if model.output_shape[-1] != len(class_names):
        st.error(
            f"Output model ({model.output_shape[-1]}) tidak sama dengan "
            f"jumlah nama kelas ({len(class_names)}). Periksa model dan class_names.json."
        )
        st.stop()
except Exception as error:
    st.error(f"Gagal memuat model atau daftar kelas: {error}")
    st.info(
        "Periksa struktur repositori: models/leafvision_cnn_38_classes.keras "
        "dan metadata/class_names.json."
    )
    st.stop()

# =====================================================
# 4. INFORMASI PENYAKIT
# =====================================================
DISEASE_INFO = {
    "apple_scab": ("Apple Scab", "Penyakit jamur pada tanaman apel.", "Bercak gelap atau zaitun pada daun; gejala perlu dikonfirmasi pada tanaman.", "Buang bagian tanaman yang terinfeksi dan jaga kebersihan kebun."),
    "black_rot": ("Black Rot", "Penyakit jamur yang dapat menyerang beberapa tanaman buah.", "Bercak nekrotik atau jaringan daun mengering.", "Pangkas bagian terinfeksi dan hindari kelembapan berlebih."),
    "cedar_apple_rust": ("Cedar Apple Rust", "Penyakit karat pada apel yang berkaitan dengan inang cedar/juniper.", "Bercak kekuningan hingga jingga pada daun.", "Pantau gejala dan lakukan pengelolaan inang sesuai rekomendasi setempat."),
    "powdery_mildew": ("Powdery Mildew", "Penyakit jamur embun tepung.", "Lapisan putih seperti tepung pada permukaan daun.", "Tingkatkan sirkulasi udara dan hindari kondisi terlalu lembap."),
    "cercospora": ("Cercospora / Gray Leaf Spot", "Penyakit bercak daun akibat kelompok jamur Cercospora.", "Bercak memanjang atau tidak beraturan pada daun.", "Gunakan benih sehat, rotasi tanaman, dan kelola sisa tanaman."),
    "common_rust": ("Common Rust", "Penyakit karat pada jagung.", "Pustula kecil berwarna cokelat kemerahan pada daun.", "Gunakan varietas tahan dan pantau perkembangan gejala."),
    "northern_leaf_blight": ("Northern Leaf Blight", "Penyakit hawar daun pada jagung.", "Lesi memanjang berwarna cokelat keabu-abuan.", "Gunakan varietas tahan dan lakukan rotasi tanaman."),
    "esca": ("Esca / Black Measles", "Gangguan penyakit kompleks pada tanaman anggur.", "Pola perubahan warna dan nekrosis pada daun.", "Pangkas bagian sakit dengan alat bersih dan minta pemeriksaan ahli."),
    "leaf_blight": ("Leaf Blight", "Penyakit hawar daun.", "Area daun menguning, kecokelatan, atau mengering.", "Buang jaringan yang rusak dan jaga sirkulasi udara."),
    "huanglongbing": ("Huanglongbing / Citrus Greening", "Penyakit sistemik pada tanaman jeruk.", "Daun dapat belang tidak simetris dan tanaman tampak merana.", "Konfirmasi melalui pemeriksaan pertanian; kendalikan vektor sesuai panduan setempat."),
    "bacterial_spot": ("Bacterial Spot", "Penyakit bercak daun yang disebabkan bakteri.", "Bercak kecil gelap, kadang dengan area kekuningan di sekitarnya.", "Gunakan bibit sehat, hindari penyiraman dari atas daun, dan sanitasi alat."),
    "early_blight": ("Early Blight", "Penyakit yang umumnya disebabkan jamur Alternaria.", "Bercak cokelat yang dapat membentuk lingkaran konsentris.", "Buang daun terinfeksi, jaga sirkulasi udara, dan hindari kelembapan berlebih."),
    "late_blight": ("Late Blight", "Penyakit akibat Phytophthora infestans.", "Bercak cokelat gelap yang dapat berkembang cepat pada kondisi lembap.", "Pisahkan tanaman bergejala dan minta rekomendasi pengendalian yang sesuai."),
    "leaf_mold": ("Leaf Mold", "Penyakit jamur pada daun, umum pada tomat.", "Bercak kekuningan di atas daun dan pertumbuhan seperti beludru di bawah daun.", "Tingkatkan ventilasi dan kurangi kelembapan di sekitar tanaman."),
    "septoria": ("Septoria Leaf Spot", "Penyakit jamur Septoria.", "Bercak kecil bulat dengan bagian tengah abu-abu atau kecokelatan.", "Buang daun terinfeksi dan hindari percikan air dari tanah."),
    "spider_mites": ("Spider Mites", "Serangan tungau yang mengisap cairan daun.", "Daun berbintik kekuningan, kusam, dan terkadang terdapat jaring halus.", "Periksa bagian bawah daun dan lakukan pengendalian tungau yang sesuai."),
    "target_spot": ("Target Spot", "Penyakit bercak daun yang dapat disebabkan Corynespora cassiicola.", "Bercak cokelat dengan pola lingkaran atau sasaran.", "Jaga jarak tanam, sirkulasi udara, dan kebersihan daun."),
    "yellow_leaf_curl": ("Tomato Yellow Leaf Curl Virus", "Penyakit virus yang dapat ditularkan kutu kebul.", "Daun menggulung, menguning, dan pertumbuhan tanaman terhambat.", "Kendalikan kutu kebul dan singkirkan tanaman terinfeksi sesuai praktik budidaya."),
    "tomato_mosaic": ("Tomato Mosaic Virus", "Penyakit virus mosaik pada tomat.", "Pola warna hijau muda dan hijau tua menyerupai mosaik.", "Gunakan benih sehat dan sanitasi peralatan."),
    "leaf_scorch": ("Leaf Scorch", "Gangguan yang menyebabkan jaringan daun mengalami nekrosis.", "Tepi atau bagian daun tampak kering dan kecokelatan.", "Periksa kondisi air, akar, dan faktor lingkungan; konfirmasi penyebabnya."),
    "cedar_rust": ("Rust Disease", "Penyakit karat pada daun.", "Bercak atau pustula berwarna kuning-jingga hingga cokelat.", "Buang bagian yang sangat terinfeksi dan jaga sirkulasi udara."),
}

def get_disease_info(label):
    normalized = label.lower()
    if "healthy" in normalized:
        return {
            "name": "Healthy / Sehat",
            "cause": "Model mengklasifikasikan gambar ke kelas daun sehat.",
            "symptoms": "Pada citra yang diberikan, model tidak menemukan pola yang paling sesuai dengan kelas penyakit dalam dataset.",
            "handling": "Pertahankan perawatan tanaman dan lakukan pemantauan rutin. Hasil AI bukan jaminan tanaman bebas penyakit."
        }

    key_map = [
        ("yellow_leaf_curl", "yellow_leaf_curl"),
        ("yellowleaf", "yellow_leaf_curl"),
        ("mosaic_virus", "tomato_mosaic"),
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
            return {"name": name, "cause": cause, "symptoms": symptoms, "handling": handling}

    readable = label.replace("___", " — ").replace("_", " ").replace(",", "")
    return {
        "name": readable,
        "cause": "Informasi khusus untuk kelas ini belum tersedia di aplikasi.",
        "symptoms": "Periksa gejala pada daun dan tanaman secara langsung.",
        "handling": "Jangan menentukan perlakuan hanya dari prediksi AI; lakukan pemeriksaan lanjutan."
    }

def readable_class(label):
    plant, sep, condition = label.partition("___")
    if not sep:
        return label.replace("_", " ")
    plant = plant.replace("_", " ").replace(",", "")
    condition = condition.replace("_", " ").replace(",", "")
    return f"{plant} — {condition}"

# =====================================================
# 5. PEMROSESAN GAMBAR DAN PREDIKSI
# =====================================================
def prepare_image(uploaded_image):
    # Koreksi orientasi foto dari kamera/ponsel berdasarkan metadata EXIF.
    image = ImageOps.exif_transpose(uploaded_image).convert("RGB")
    original_size = image.size

    # Jangan melakukan crop/segmentasi/peningkatan warna yang tidak digunakan
    # saat training. Resize dan normalisasi harus konsisten dengan pipeline training.
    resized = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image, image_array, original_size

def image_quality_notes(image, original_size):
    notes = []
    width, height = original_size

    if min(width, height) < 160:
        notes.append("Resolusi foto cukup kecil. Gunakan foto dengan daun lebih dekat dan tajam.")
    if max(width, height) / max(1, min(width, height)) > 2.5:
        notes.append("Foto sangat memanjang. Usahakan daun berada di tengah dan terlihat utuh.")

    gray = np.asarray(image.convert("L").resize((256, 256)), dtype=np.float32)
    edge_strength = np.var(
        gray[1:, :] - gray[:-1, :]
    ) + np.var(
        gray[:, 1:] - gray[:, :-1]
    )
    if edge_strength < 18:
        notes.append("Foto mungkin kurang tajam atau terlalu buram. Coba ambil ulang dengan fokus pada daun.")

    brightness = float(np.mean(gray))
    if brightness < 35:
        notes.append("Foto terlihat terlalu gelap. Gunakan pencahayaan yang cukup.")
    elif brightness > 225:
        notes.append("Foto terlihat terlalu terang. Hindari pantulan cahaya langsung.")

    return notes

def predict_image(image_array):
    probabilities = np.asarray(model.predict(image_array, verbose=0)[0], dtype=np.float64)

    if probabilities.ndim != 1 or len(probabilities) != len(class_names):
        raise ValueError("Bentuk output prediksi tidak sesuai dengan daftar kelas.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model menghasilkan nilai prediksi yang tidak valid.")

    top_indices = np.argsort(probabilities)[::-1][:3]
    confidence = float(probabilities[int(top_indices[0])] * 100)
    second_confidence = float(probabilities[int(top_indices[1])] * 100)
    margin = confidence - second_confidence

    return {
        "label": class_names[int(top_indices[0])],
        "confidence": confidence,
        "margin": margin,
        "top_indices": [int(index) for index in top_indices],
        "probabilities": probabilities.tolist(),
        "uncertain": confidence < CONFIDENCE_WARNING or margin < 8.0
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
        ["Dashboard", "Deteksi Penyakit", "Informasi Penyakit", "Riwayat Deteksi", "Tentang Model"],
        key="navigation"
    )
    st.divider()
    st.caption("LeafVision AI | CNN 38 Classes")
    st.caption("AI-Based Plant Disease Classification")

# =====================================================
# 8. DASHBOARD
# =====================================================
if page == "Dashboard":
    st.markdown("""
    <div class="hero">
        <h1>Kenali Kesehatan Tanamanmu 🌱</h1>
        <p>Unggah foto daun untuk memperoleh prediksi kelas tanaman dan kemungkinan penyakit
        menggunakan teknologi Artificial Intelligence dan Convolutional Neural Network.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Jumlah Kelas</div>
        <div class="metric-value">{len(class_names)} Kelas</div><p>Tanaman dan kondisi daun</p></div>""",
        unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card"><div class="metric-label">Model AI</div>
        <div class="metric-value">CNN</div><p>Convolutional Neural Network</p></div>""",
        unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Test Accuracy</div>
        <div class="metric-value">{MODEL_ACCURACY}</div><p>Evaluasi model 38 kelas</p></div>""",
        unsafe_allow_html=True)

    st.write("")
    st.markdown("### Mulai Deteksi")
    st.write("Gunakan foto satu daun yang terlihat jelas, tidak buram, dan memiliki pencahayaan cukup.")
    st.button("Mulai Deteksi Sekarang →", on_click=go_to_detection)

# =====================================================
# 9. DETEKSI PENYAKIT
# =====================================================
elif page == "Deteksi Penyakit":
    st.markdown('<div class="main-title">Deteksi Penyakit</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Unggah foto daun yang jelas untuk memulai analisis AI.</div>',
                unsafe_allow_html=True)

    st.info(
        "Tips foto: gunakan satu daun sebagai objek utama, dekatkan kamera secukupnya, "
        "pastikan fokus tajam, cahaya merata, dan hindari latar belakang yang terlalu ramai. "
        "Model mengenali pola yang dipelajari dari dataset, bukan melakukan diagnosis laboratorium."
    )

    uploaded_file = st.file_uploader(
        "Upload foto daun tanaman",
        type=["jpg", "jpeg", "png"],
        key="leaf_upload"
    )

    if uploaded_file is not None:
        try:
            raw_image = Image.open(uploaded_file)
            image, image_array, original_size = prepare_image(raw_image)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image, caption="Foto daun yang diunggah", use_container_width=True)
                st.caption(f"Resolusi asli: {original_size[0]} × {original_size[1]} piksel")

            with col2:
                st.markdown("### Sebelum dianalisis")
                notes = image_quality_notes(image, original_size)
                if notes:
                    for note in notes:
                        st.warning(note)
                else:
                    st.success("Kualitas dasar gambar terlihat cukup baik. Pastikan objek utama memang daun.")

                if st.button("🔍 Analisis Gambar", type="primary", use_container_width=True):
                    with st.spinner("AI sedang menganalisis gambar..."):
                        prediction = predict_image(image_array)
                        info = get_disease_info(prediction["label"])
                        result = {
                            **prediction,
                            "image_name": uploaded_file.name,
                            "time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                            "info_name": info["name"]
                        }
                        st.session_state.result = result
                        st.session_state.history.insert(0, result)

        except Exception as error:
            st.error(f"Gambar tidak dapat diproses: {error}")

    if st.session_state.result is not None:
        result = st.session_state.result
        info = get_disease_info(result["label"])

        st.divider()
        st.markdown("## Hasil Analisis")

        if result["uncertain"]:
            st.warning(
                "Hasil prediksi belum cukup meyakinkan. Periksa kembali foto dan coba unggah "
                "gambar daun yang lebih jelas. Jangan langsung menggunakan hasil ini sebagai diagnosis."
            )
        else:
            st.success("Model menemukan satu kelas yang paling sesuai pada gambar.")

        st.markdown(
            f"""
            <div class="result-card">
                <div style="font-size:14px;color:#477557;">PREDIKSI KELAS TERATAS</div>
                <h2 style="color:#174B30;">{info['name']}</h2>
                <h2 style="color:#246B43;">{result['confidence']:.2f}%</h2>
                <div style="color:#477557;">Skor keluaran model (bukan probabilitas diagnosis klinis)</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### Top 3 Prediksi")
        for index in result["top_indices"]:
            label = class_names[index]
            score = result["probabilities"][index] * 100
            st.write(f"**{readable_class(label)}**")
            st.progress(float(np.clip(score / 100, 0.0, 1.0)))
            st.caption(f"{score:.2f}%")

        st.markdown("### Informasi Kelas Teratas")
        st.info(f"**Penyebab/keterangan:** {info['cause']}")
        st.write(f"**Gejala umum:** {info['symptoms']}")
        st.write(f"**Saran:** {info['handling']}")
        st.warning(
            "Hasil merupakan prediksi model berdasarkan kemiripan pola gambar dengan dataset. "
            "Model dapat keliru, terutama pada foto dengan kualitas buruk, objek selain daun, "
            "atau kondisi tanaman yang tidak termasuk kelas pelatihan."
        )

# =====================================================
# 10. INFORMASI PENYAKIT
# =====================================================
elif page == "Informasi Penyakit":
    st.markdown('<div class="main-title">Informasi Penyakit</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Daftar kelas yang dapat dikenali model.</div>',
                unsafe_allow_html=True)
    for label in class_names:
        info = get_disease_info(label)
        with st.expander(readable_class(label)):
            st.write(f"**Nama kelas:** {readable_class(label)}")
            st.write(f"**Penyebab/keterangan:** {info['cause']}")
            st.write(f"**Gejala umum:** {info['symptoms']}")
            st.write(f"**Saran penanganan:** {info['handling']}")

# =====================================================
# 11. RIWAYAT DETEKSI
# =====================================================
elif page == "Riwayat Deteksi":
    st.markdown('<div class="main-title">Riwayat Deteksi</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Daftar hasil analisis selama sesi penggunaan.</div>',
                unsafe_allow_html=True)

    if st.session_state.history:
        history_df = pd.DataFrame([
            {
                "Waktu": item["time"],
                "Nama Gambar": item["image_name"],
                "Hasil": readable_class(item["label"]),
                "Skor": f"{item['confidence']:.2f}%",
                "Catatan": "Perlu pemeriksaan ulang" if item["uncertain"] else "Prediksi teratas"
            }
            for item in st.session_state.history
        ])
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        if st.button("Hapus Riwayat"):
            st.session_state.history = []
            st.session_state.result = None
            st.rerun()
    else:
        st.info("Belum ada riwayat deteksi. Silakan lakukan deteksi terlebih dahulu.")

# =====================================================
# 12. TENTANG MODEL
# =====================================================
elif page == "Tentang Model":
    st.markdown('<div class="main-title">Tentang LeafVision AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Teknologi di balik sistem klasifikasi penyakit tanaman.</div>',
                unsafe_allow_html=True)

    st.markdown("### Convolutional Neural Network")
    st.write(
        "LeafVision AI menggunakan arsitektur Convolutional Neural Network (CNN) "
        "untuk mengklasifikasikan gambar ke dalam 38 kelas tanaman dan kondisi daun."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="info-card"><b>Model</b><h2 style="color:#246B43;">CNN</h2>
        <p>Dengan data augmentation</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="info-card"><b>Test Accuracy</b>
        <h2 style="color:#246B43;">{MODEL_ACCURACY}</h2><p>Hasil evaluasi model 38 kelas</p></div>""",
        unsafe_allow_html=True)

    st.markdown("### Spesifikasi Model")
    st.write("- Ukuran input: 160 × 160 piksel")
    st.write("- Jumlah kelas: 38")
    st.write("- Optimizer saat training: Adam")
    st.write("- Loss: Sparse Categorical Crossentropy")
    st.write("- Augmentation saat training: Flip, Rotation, Zoom, Contrast")
    st.write("- Normalisasi input: nilai piksel dibagi 255")
    st.write("- Test accuracy: 92,09%")
    st.caption(
        "Angka evaluasi berasal dari pengujian pada dataset yang digunakan dalam proyek. "
        "Performa pada foto lapangan dapat berbeda."
    )

    st.markdown("### Daftar Kelas")
    for number, name in enumerate(class_names, start=1):
        st.write(f"{number}. {readable_class(name)}")

# =====================================================
# 13. FOOTER
# =====================================================
st.divider()
st.markdown("""
<div style="text-align:center;color:#829187;padding:15px;">
    <b>🌿 LeafVision AI</b><br>
    AI-Based Plant Disease Detection System
</div>
""", unsafe_allow_html=True)
