from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).with_name("random_forest_regressor_model.pkl")

INPUT_SECTIONS = {
    "Profil": [
        "age",
        "income",
        "household_size",
        "dependents",
        "bmi",
        "education",
    ],
    "Kesehatan": [
        "visits_last_year",
        "hospitalizations_last_3yrs",
        "days_hospitalized_last_3yrs",
        "medication_count",
        "systolic_bp",
        "diastolic_bp",
        "ldl",
        "hba1c",
        "chronic_count",
        "alcohol_freq",
        "hypertension",
        "diabetes",
        "asthma",
        "copd",
        "cardiovascular_disease",
        "cancer_history",
        "kidney_disease",
        "liver_disease",
        "arthritis",
        "mental_health",
    ],
    "Polis": [
        "deductible",
        "copay",
        "policy_term_years",
        "policy_changes_last_2yrs",
        "provider_quality",
        "risk_score",
        "network_tier",
        "is_high_risk",
    ],
    "Klaim & tindakan": [
        "annual_premium",
        "monthly_premium",
        "claims_count",
        "avg_claim_amount",
        "total_claims_paid",
        "proc_imaging_count",
        "proc_surgery_count",
        "proc_physio_count",
        "proc_consult_count",
        "proc_lab_count",
        "had_major_procedure",
    ],
}

ONE_HOT_GROUPS = {
    "region": ("Wilayah", "region_"),
    "marital_status": ("Status pernikahan", "marital_status_"),
    "employment_status": ("Status pekerjaan", "employment_status_"),
    "plan_type": ("Jenis paket", "plan_type_"),
}

ONE_HOT_SECTIONS = {
    "region": "Profil",
    "marital_status": "Profil",
    "employment_status": "Profil",
    "plan_type": "Polis",
}

FEATURE_LABELS = {
    "age": "Usia",
    "income": "Pendapatan",
    "household_size": "Jumlah anggota rumah tangga",
    "dependents": "Jumlah tanggungan",
    "bmi": "BMI",
    "education": "Pendidikan (kode model)",
    "visits_last_year": "Kunjungan tahun lalu",
    "hospitalizations_last_3yrs": "Rawat inap (3 tahun terakhir)",
    "days_hospitalized_last_3yrs": "Hari rawat inap (3 tahun terakhir)",
    "medication_count": "Jumlah obat",
    "systolic_bp": "Tekanan darah sistolik",
    "diastolic_bp": "Tekanan darah diastolik",
    "ldl": "LDL",
    "hba1c": "HbA1c",
    "chronic_count": "Jumlah kondisi kronis",
    "alcohol_freq": "Frekuensi konsumsi alkohol (kode model)",
    "hypertension": "Hipertensi",
    "diabetes": "Diabetes",
    "asthma": "Asma",
    "copd": "COPD",
    "cardiovascular_disease": "Penyakit kardiovaskular",
    "cancer_history": "Riwayat kanker",
    "kidney_disease": "Penyakit ginjal",
    "liver_disease": "Penyakit hati",
    "arthritis": "Artritis",
    "mental_health": "Kondisi kesehatan mental",
    "deductible": "Deductible",
    "copay": "Copay",
    "policy_term_years": "Masa polis (tahun)",
    "policy_changes_last_2yrs": "Perubahan polis (2 tahun terakhir)",
    "provider_quality": "Kualitas provider (skor model)",
    "risk_score": "Skor risiko model",
    "network_tier": "Tingkat jaringan (kode model)",
    "is_high_risk": "Ditandai berisiko tinggi",
    "annual_premium": "Premi tahunan",
    "monthly_premium": "Premi bulanan",
    "claims_count": "Jumlah klaim",
    "avg_claim_amount": "Rata-rata nilai klaim",
    "total_claims_paid": "Total klaim dibayar",
    "proc_imaging_count": "Prosedur pencitraan",
    "proc_surgery_count": "Prosedur operasi",
    "proc_physio_count": "Prosedur fisioterapi",
    "proc_consult_count": "Prosedur konsultasi",
    "proc_lab_count": "Prosedur laboratorium",
    "had_major_procedure": "Pernah menjalani prosedur besar",
}

INTEGER_FEATURES = {
    "age",
    "household_size",
    "dependents",
    "visits_last_year",
    "hospitalizations_last_3yrs",
    "days_hospitalized_last_3yrs",
    "medication_count",
    "chronic_count",
    "policy_term_years",
    "policy_changes_last_2yrs",
    "claims_count",
    "proc_imaging_count",
    "proc_surgery_count",
    "proc_physio_count",
    "proc_consult_count",
    "proc_lab_count",
}

BINARY_FEATURES = {
    "hypertension",
    "diabetes",
    "asthma",
    "copd",
    "cardiovascular_disease",
    "cancer_history",
    "kidney_disease",
    "liver_disease",
    "arthritis",
    "mental_health",
    "is_high_risk",
    "had_major_procedure",
}


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


def input_label(feature_name: str) -> str:
    return FEATURE_LABELS.get(feature_name, feature_name.replace("_", " ").title())


def render_numeric_input(feature_name: str):
    is_integer = feature_name in INTEGER_FEATURES
    step = 1 if is_integer else 0.1
    value = st.number_input(
        input_label(feature_name),
        value=None,
        placeholder="Masukkan nilai",
        step=step,
        format="%d" if is_integer else "%.4f",
        key=f"manual_{feature_name}",
    )
    return value


def render_one_hot_input(prefix: str, label: str, feature_names: list[str]):
    options = [name for name in feature_names if name.startswith(prefix)]
    if not options:
        return {}

    display_options = {
        name: name.removeprefix(prefix).replace("_", " ").title()
        for name in options
    }
    selected = st.selectbox(
        label,
        options=[None, *options],
        index=0,
        format_func=lambda option: (
            "Pilih..." if option is None else display_options[option]
        ),
        key=f"manual_{prefix.rstrip('_')}",
    )
    return {name: float(name == selected) for name in options}, selected


def render_manual_form(feature_names: list[str]) -> dict[str, float] | None:
    feature_set = set(feature_names)
    values = {}
    missing_selections = []

    with st.form("manual_prediction"):
        profile_tab, health_tab, policy_tab, claims_tab = st.tabs(
            ["Profil", "Kesehatan", "Polis", "Klaim & tindakan"]
        )

        for tab, section_name in zip(
            [profile_tab, health_tab, policy_tab, claims_tab], INPUT_SECTIONS
        ):
            with tab:
                st.subheader(section_name)
                section_features = [
                    feature
                    for feature in INPUT_SECTIONS[section_name]
                    if feature in feature_set
                ]
                columns = st.columns(2)
                for index, feature in enumerate(section_features):
                    with columns[index % 2]:
                        if feature in BINARY_FEATURES:
                            selected_value = st.selectbox(
                                input_label(feature),
                                options=["Pilih...", "Ya", "Tidak"],
                                index=0,
                                key=f"manual_{feature}",
                            )
                            if selected_value == "Pilih...":
                                missing_selections.append(input_label(feature))
                            else:
                                values[feature] = float(selected_value == "Ya")
                        else:
                            values[feature] = render_numeric_input(feature)

                for group_key, (group_label, prefix) in ONE_HOT_GROUPS.items():
                    if (
                        ONE_HOT_SECTIONS[group_key] == section_name
                        and any(name.startswith(prefix) for name in feature_names)
                    ):
                        group_values, selected = render_one_hot_input(
                            prefix, group_label, feature_names
                        )
                        values.update(group_values)
                        if selected is None:
                            missing_selections.append(group_label)

                if section_name == "Kesehatan":
                    st.caption(
                        "Skor/kode mengikuti encoding saat pelatihan model. "
                        "Gunakan nilai numerik yang sama dengan data training."
                    )
                elif section_name == "Polis":
                    st.caption(
                        "Nilai nominal dan skor tidak diberi satuan karena "
                        "metadata model tidak menyimpannya."
                    )

        submitted = st.form_submit_button(
            "Jalankan prediksi", type="primary", use_container_width=True
        )

    if not submitted:
        return None

    empty_inputs = [
        input_label(name)
        for name, value in values.items()
        if value is None or pd.isna(value)
    ]
    if empty_inputs or missing_selections:
        missing = list(dict.fromkeys(empty_inputs + missing_selections))
        st.error("Lengkapi semua input sebelum prediksi: " + ", ".join(missing))
        return None

    return values


st.set_page_config(page_title="Prediksi Random Forest", layout="wide")
st.title("Prediksi Model Asuransi")
st.caption("Prediksi menggunakan model Random Forest")

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

template = pd.DataFrame(columns=feature_names)
st.download_button(
    "Unduh template CSV",
    data=template.to_csv(index=False).encode("utf-8"),
    file_name="template_input.csv",
    mime="text/csv",
)

upload_tab, manual_tab = st.tabs(["Unggah CSV", "Input terarah"])

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
        "Isi data aktual pada setiap bagian. Nilai kosong tidak akan dianggap nol."
    )
    manual_values = render_manual_form(feature_names)
    if manual_values is not None:
        manual_data = pd.DataFrame([manual_values], columns=feature_names)
        show_predictions(model, manual_data, feature_names, "manual")

st.caption(
    "Catatan: file pickle/joblib hanya aman dimuat jika berasal dari sumber tepercaya."
)