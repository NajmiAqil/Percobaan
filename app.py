from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).with_name("random_forest_regressor_model.pkl")


@st.cache_resource(show_spinner="Memuat model...")
def load_model(model_path: str):
    return joblib.load(model_path)


def get_feature_names(model) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return [str(name) for name in model.feature_names_in_]

    feature_count = getattr(model, "n_features_in_", None)
    if feature_count is None:
        raise ValueError("Model tidak menyediakan nama atau jumlah fitur input.")
    return [f"feature_{index + 1}" for index in range(feature_count)]


def prepare_input(data: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    if data.empty:
        raise ValueError("Data belum memiliki baris untuk diprediksi.")

    missing = [name for name in feature_names if name not in data.columns]
    if missing:
        raise ValueError("Kolom wajib belum tersedia: " + ", ".join(missing))

    numeric_data = data.loc[:, feature_names].copy()
    for column in feature_names:
        numeric_data[column] = pd.to_numeric(numeric_data[column], errors="coerce")

    invalid_columns = numeric_data.columns[numeric_data.isna().any()].tolist()
    if invalid_columns:
        raise ValueError(
            "Nilai kosong atau bukan angka ditemukan pada: "
            + ", ".join(invalid_columns)
        )

    if not np.isfinite(numeric_data.to_numpy(dtype=float)).all():
        raise ValueError("Input tidak boleh berisi nilai tak hingga.")

    return numeric_data


def show_predictions(model, data: pd.DataFrame, feature_names: list[str], key: str):
    try:
        model_input = prepare_input(data, feature_names)
        predictions = np.asarray(model.predict(model_input))
        if predictions.ndim == 1:
            prediction_columns = pd.DataFrame({"hasil_prediksi": predictions})
        elif predictions.ndim == 2:
            prediction_columns = pd.DataFrame(
                predictions,
                columns=[
                    f"hasil_prediksi_{index + 1}"
                    for index in range(predictions.shape[1])
                ],
            )
        else:
            raise ValueError("Bentuk output prediksi model tidak didukung.")

        results = pd.concat(
            [model_input.reset_index(drop=True), prediction_columns], axis=1
        )
        st.dataframe(results, use_container_width=True)
        st.download_button(
            "Unduh hasil CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="hasil_prediksi.csv",
            mime="text/csv",
            key=f"download_{key}",
        )
    except Exception as error:
        st.error(f"Prediksi gagal: {error}")


st.set_page_config(page_title="Prediksi Random Forest", layout="wide")
st.title("Prediksi Random Forest")
st.caption("Unggah CSV atau masukkan data, lalu jalankan prediksi model.")

if not MODEL_PATH.is_file():
    st.error(f"File model tidak ditemukan: {MODEL_PATH.name}")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
    feature_names = get_feature_names(model)
except Exception as error:
    st.error(f"Model tidak dapat dimuat: {error}")
    st.stop()

st.caption(f"Model siap | {len(feature_names)} fitur input")
st.info(
    "Gunakan kolom numerik dengan nama persis seperti template. "
    "Kolom kategori pada model ini sudah berupa fitur one-hot encoded."
)

template = pd.DataFrame(columns=feature_names)
st.download_button(
    "Unduh template CSV",
    data=template.to_csv(index=False).encode("utf-8"),
    file_name="template_input.csv",
    mime="text/csv",
)

upload_tab, manual_tab = st.tabs(["Unggah CSV", "Input tabel"])

with upload_tab:
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_data = pd.read_csv(uploaded_file)
            st.dataframe(uploaded_data.head(10), use_container_width=True)

            missing_columns = [
                name for name in feature_names if name not in uploaded_data.columns
            ]
            extra_columns = [
                name for name in uploaded_data.columns if name not in feature_names
            ]
            if missing_columns:
                st.error("Kolom wajib belum tersedia: " + ", ".join(missing_columns))
            else:
                if extra_columns:
                    st.warning(
                        "Kolom tambahan akan diabaikan: " + ", ".join(extra_columns)
                    )
                if st.button("Prediksi CSV", type="primary"):
                    show_predictions(model, uploaded_data, feature_names, "csv")
        except Exception as error:
            st.error(f"File CSV tidak dapat dibaca: {error}")

with manual_tab:
    st.caption(
        "Isi semua fitur dengan nilai numerik. Baris awal bernilai 0 sebagai placeholder."
    )
    manual_data = st.data_editor(
        pd.DataFrame([{name: 0.0 for name in feature_names}]),
        num_rows="dynamic",
        use_container_width=True,
        key="manual_input",
    )
    if st.button("Prediksi tabel", type="primary"):
        show_predictions(model, manual_data, feature_names, "manual")

st.caption(
    "Catatan: file pickle/joblib hanya aman dimuat jika berasal dari sumber tepercaya."
)