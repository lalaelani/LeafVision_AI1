
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "leafvision_cnn_augmentation.keras"
CLASS_PATH = BASE_DIR / "metadata" / "class_names.json"

st.set_page_config(
    page_title="LeafVision AI",
    page_icon="🌿",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

@st.cache_data
def load_classes():
    with open(CLASS_PATH, "r") as f:
        return json.load(f)

st.title("🌿 LeafVision AI")
st.subheader("Deteksi Penyakit Tanaman Menggunakan CNN")
st.write(
    "Unggah foto daun tanaman untuk mengetahui kondisi "
    "dan kemungkinan penyakit yang terdeteksi."
)

try:
    model = load_model()
    class_names = load_classes()
except Exception as e:
    st.error(f"Gagal memuat model atau daftar kelas: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Pilih foto daun tanaman",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Foto daun yang diunggah",
        use_container_width=True
    )

    if st.button("🔍 Deteksi Penyakit", type="primary"):
        with st.spinner("Sedang menganalisis gambar..."):
            resized_image = image.resize((160, 160))
            image_array = np.array(
                resized_image, dtype=np.float32
            ) / 255.0

            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(
                image_array, verbose=0
            )[0]

            top_indices = np.argsort(predictions)[::-1][:3]

            best_index = int(top_indices[0])
            best_class = class_names[best_index]
            confidence = float(predictions[best_index]) * 100

        st.success("Analisis gambar selesai!")

        st.subheader("Hasil Deteksi")
        st.write("**Kelas terdeteksi:**")
        st.markdown(f"### {best_class.replace('_', ' ')}")
        st.metric("Tingkat keyakinan model", f"{confidence:.2f}%")

        st.subheader("3 Prediksi Teratas")

        for index in top_indices:
            label = class_names[int(index)].replace("_", " ")
            score = float(predictions[index]) * 100
            st.write(f"**{label}** — {score:.2f}%")
            st.progress(min(max(score / 100, 0.0), 1.0))

        st.caption(
            "Catatan: Persentase merupakan tingkat keyakinan "
            "prediksi model, bukan jaminan diagnosis."
        )

st.divider()
st.caption("LeafVision AI | Plant Disease Detection using CNN")
