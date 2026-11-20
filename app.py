
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import requests
import pandas as pd
import numpy as np


# =====================================================
# KONFIGURASI APLIKASI
# =====================================================

st.set_page_config(
    page_title="Bearing Diagnosis System",
    page_icon="⚙️",
    layout="wide"
)

API_URL = os.getenv(
    "API_URL",
    "https://lid-implement-throwing.ngrok-free.dev"
).rstrip("/")

PREDICT_URL = f"{API_URL}/predict"
HEALTH_URL = f"{API_URL}/health"

REQUIRED_COLUMNS = [
    "posisi_sensor",
    "kecepatan_rpm",
    "beban_kN",
    "suhu_housing_C",
    "rms_g",
    "puncak_g",
    "kurtosis",
    "crest_factor",
    "energi_bpfo",
    "energi_bpfi",
    "energi_bsf",
    "frekuensi_dominan_hz"
]

NUMERIC_COLUMNS = [
    "kecepatan_rpm",
    "beban_kN",
    "suhu_housing_C",
    "rms_g",
    "puncak_g",
    "kurtosis",
    "crest_factor",
    "energi_bpfo",
    "energi_bpfi",
    "energi_bsf",
    "frekuensi_dominan_hz"
]

NON_NEGATIVE_COLUMNS = [
    "kecepatan_rpm",
    "beban_kN",
    "rms_g",
    "puncak_g",
    "crest_factor",
    "energi_bpfo",
    "energi_bpfi",
    "energi_bsf",
    "frekuensi_dominan_hz"
]

VALID_SENSOR_POSITIONS = [
    "drive_end",
    "fan_end"
]


# =====================================================
# FUNGSI BACKEND
# =====================================================

def check_api_connection():
    try:
        response = requests.get(
            HEALTH_URL,
            timeout=5
        )

        response.raise_for_status()
        return True

    except requests.exceptions.RequestException:
        return False


def predict_single(input_data):
    response = requests.post(
        PREDICT_URL,
        json=input_data,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


# =====================================================
# FUNGSI VALIDASI CSV
# =====================================================

def validate_csv(df):

    df_process = df.copy()

    # Membersihkan nama kolom
    df_process.columns = (
        df_process.columns.astype(str).str.strip()
    )

    # Memastikan semua kolom wajib tersedia
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df_process.columns
    ]

    if missing_columns:
        return None, None, None, missing_columns

    # Membersihkan teks posisi sensor
    df_process["posisi_sensor"] = (
        df_process["posisi_sensor"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # Konversi angka desimal koma menjadi titik
    for col in NUMERIC_COLUMNS:

        values = (
            df_process[col]
            .astype("string")
            .str.strip()
            .str.replace(",", ".", regex=False)
        )

        df_process[col] = pd.to_numeric(
            values,
            errors="coerce"
        )

    errors = []
    valid_indices = []

    # Pemeriksaan setiap baris
    for index, row in df_process.iterrows():

        row_errors = []

        # Pemeriksaan nilai kosong
        for col in REQUIRED_COLUMNS:

            if pd.isna(row[col]):
                row_errors.append(
                    f"{col} kosong atau tidak valid"
                )

        # Pemeriksaan posisi sensor
        if row["posisi_sensor"] not in VALID_SENSOR_POSITIONS:

            row_errors.append(
                "posisi_sensor harus drive_end atau fan_end"
            )

        # Pemeriksaan nilai negatif
        for col in NON_NEGATIVE_COLUMNS:

            value = row[col]

            if pd.notna(value) and value < 0:

                row_errors.append(
                    f"{col} bernilai negatif"
                )

        # Pemeriksaan angka tak hingga
        for col in NUMERIC_COLUMNS:

            value = row[col]

            if pd.notna(value) and not np.isfinite(value):

                row_errors.append(
                    f"{col} bukan angka terbatas"
                )

        if row_errors:

            errors.append({
                "baris": index + 2,
                "id_rekaman": (
                    df_process.at[index, "id_rekaman"]
                    if "id_rekaman" in df_process.columns
                    else "-"
                ),
                "kesalahan": "; ".join(row_errors)
            })

        else:

            valid_indices.append(index)

    # Memisahkan data
    df_valid = df_process.loc[
        valid_indices
    ].copy()

    df_invalid = df_process.drop(
        valid_indices
    ).copy()

    error_df = pd.DataFrame(errors)

    return df_valid, df_invalid, error_df, []


# =====================================================
# FUNGSI PREDIKSI CSV
# =====================================================

def predict_csv_row(index, row):

    input_data = {}

    for col in REQUIRED_COLUMNS:

        if col == "posisi_sensor":
            input_data[col] = str(row[col])

        else:
            input_data[col] = float(row[col])

    try:

        result = predict_single(input_data)

        probabilities = result.get(
            "probabilities", {}
        )

        recommendation = result.get(
            "recommendation", {}
        )

        confidence = max(
            probabilities.values(),
            default=0
        )

        output = {
            "nomor_baris": index + 2,
            "prediksi": result.get("prediction"),
            "confidence": round(
                float(confidence) * 100, 2
            ),
            "status": recommendation.get(
                "status", "-"
            )
        }

        # Mempertahankan identitas data
        if "id_rekaman" in row.index:

            output["id_rekaman"] = row["id_rekaman"]

        # Mempertahankan label aktual
        if "kondisi" in row.index:

            output["kondisi_aktual"] = row["kondisi"]

        # Menambahkan probabilitas setiap kelas
        for label, probability in probabilities.items():

            output[
                f"probabilitas_{label}"
            ] = round(
                float(probability) * 100, 2
            )

        return output

    except requests.exceptions.RequestException as e:

        return {
            "nomor_baris": index + 2,
            "id_rekaman": row.get("id_rekaman", "-"),
            "prediksi": "Gagal",
            "confidence": 0,
            "status": str(e)
        }

    except (ValueError, TypeError, KeyError) as e:

        return {
            "nomor_baris": index + 2,
            "id_rekaman": row.get("id_rekaman", "-"),
            "prediksi": "Gagal",
            "confidence": 0,
            "status": f"Respons API tidak valid: {e}"
        }


# =====================================================
# HEADER APLIKASI
# =====================================================

st.title("⚙️ Bearing Diagnosis System")

st.write(
    "Sistem diagnosis kerusakan bearing menggunakan "
    "Machine Learning."
)

st.divider()


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("Informasi Sistem")

    st.write("**Backend:** FastAPI")

    st.write("**Machine Learning:** Bearing Diagnosis Model")

    st.write("**API URL:**")

    st.code(API_URL)

    st.divider()

    if check_api_connection():

        st.success("FastAPI Connected")

    else:

        st.error("FastAPI Disconnected")

    st.caption(
        "Pastikan FastAPI berjalan sebelum melakukan diagnosis."
    )


# =====================================================
# TAB APLIKASI
# =====================================================

tab_manual, tab_csv = st.tabs([
    "Diagnosis Manual",
    "Upload CSV"
])


# =====================================================
# TAB 1: DIAGNOSIS MANUAL
# =====================================================

with tab_manual:

    st.header("Diagnosis Bearing Manual")

    st.write(
        "Masukkan parameter pengukuran bearing "
        "untuk mendapatkan hasil diagnosis."
    )

    with st.form("manual_diagnosis_form"):

        col1, col2 = st.columns(2)

        with col1:

            posisi_sensor = st.selectbox(
                "Posisi Sensor",
                VALID_SENSOR_POSITIONS
            )

            kecepatan_rpm = st.number_input(
                "Kecepatan (RPM)",
                min_value=0.0,
                value=2547.0
            )

            beban_kN = st.number_input(
                "Beban (kN)",
                min_value=0.0,
                value=2.79
            )

            suhu_housing_C = st.number_input(
                "Suhu Housing (°C)",
                value=53.8
            )

            rms_g = st.number_input(
                "RMS (g)",
                min_value=0.0,
                value=0.5168
            )

            puncak_g = st.number_input(
                "Puncak (g)",
                min_value=0.0,
                value=1.1658
            )

        with col2:

            kurtosis = st.number_input(
                "Kurtosis",
                value=0.039
            )

            crest_factor = st.number_input(
                "Crest Factor",
                min_value=0.0,
                value=2.256
            )

            energi_bpfo = st.number_input(
                "Energi BPFO",
                min_value=0.0,
                value=0.1400,
                format="%.4f"
            )

            energi_bpfi = st.number_input(
                "Energi BPFI",
                min_value=0.0,
                value=0.0436,
                format="%.4f"
            )

            energi_bsf = st.number_input(
                "Energi BSF",
                min_value=0.0,
                value=0.0629,
                format="%.4f"
            )

            frekuensi_dominan_hz = st.number_input(
                "Frekuensi Dominan (Hz)",
                min_value=0.0,
                value=40.9
            )

        submitted = st.form_submit_button(
            "Diagnosis Bearing",
            type="primary",
            use_container_width=True
        )

    if submitted:

        input_data = {
            "posisi_sensor": posisi_sensor,
            "kecepatan_rpm": kecepatan_rpm,
            "beban_kN": beban_kN,
            "suhu_housing_C": suhu_housing_C,
            "rms_g": rms_g,
            "puncak_g": puncak_g,
            "kurtosis": kurtosis,
            "crest_factor": crest_factor,
            "energi_bpfo": energi_bpfo,
            "energi_bpfi": energi_bpfi,
            "energi_bsf": energi_bsf,
            "frekuensi_dominan_hz": frekuensi_dominan_hz
        }

        try:

            with st.spinner("Melakukan diagnosis..."):

                result = predict_single(input_data)

            st.success("Diagnosis berhasil!")

            prediction = result.get("prediction")

            probabilities = result.get(
                "probabilities", {}
            )

            recommendation = result.get(
                "recommendation", {}
            )

            st.subheader("Hasil Diagnosis")

            st.metric(
                "Kondisi Bearing",
                str(prediction).replace("_", " ").title()
            )

            if probabilities:

                probability_df = pd.DataFrame(
                    list(probabilities.items()),
                    columns=["Kelas", "Probabilitas"]
                )

                probability_df["Probabilitas"] *= 100

                st.write("**Probabilitas setiap kelas**")

                st.bar_chart(
                    probability_df.set_index("Kelas")
                )

            st.subheader("Rekomendasi")

            st.write(
                recommendation.get("status", "-")
            )

            st.write("**Tindakan:**")

            for action in recommendation.get("action", []):

                st.write(f"- {action}")

            st.write("**Inspeksi:**")

            for inspection in recommendation.get("inspection", []):

                st.write(f"- {inspection}")

            spare_parts = recommendation.get(
                "spare_parts", []
            )

            if spare_parts:

                st.write("**Spare Parts:**")

                for part in spare_parts:

                    st.write(f"- {part}")

        except requests.exceptions.RequestException as e:

            st.error(
                f"Gagal menghubungi FastAPI: {e}"
            )


# =====================================================
# TAB 2: UPLOAD CSV
# =====================================================

with tab_csv:

    st.header("Diagnosis Bearing melalui CSV")

    st.write(
        "Unggah dataset CSV untuk melakukan diagnosis "
        "terhadap banyak data bearing sekaligus."
    )

    # -------------------------------------------------
    # TEMPLATE CSV
    # -------------------------------------------------

    st.subheader("Template CSV")

    template_data = {
        "posisi_sensor": ["drive_end"],
        "kecepatan_rpm": [2547.0],
        "beban_kN": [2.79],
        "suhu_housing_C": [53.8],
        "rms_g": [0.5168],
        "puncak_g": [1.1658],
        "kurtosis": [0.039],
        "crest_factor": [2.256],
        "energi_bpfo": [0.1400],
        "energi_bpfi": [0.0436],
        "energi_bsf": [0.0629],
        "frekuensi_dominan_hz": [40.9]
    }

    template_df = pd.DataFrame(template_data)

    st.download_button(
        "Download Template CSV",
        data=template_df.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="template_bearing.csv",
        mime="text/csv"
    )

    st.divider()

    # -------------------------------------------------
    # UPLOAD FILE
    # -------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload Dataset Bearing",
        type=["csv"],
        help="Upload file CSV dengan 12 kolom fitur wajib."
    )

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()

        file_hash = hashlib.md5(
            file_bytes
        ).hexdigest()

        try:

            df_upload = pd.read_csv(
                uploaded_file,
                sep=",",
                encoding="utf-8-sig"
            )

            df_upload.columns = (
                df_upload.columns.astype(str).str.strip()
            )

            st.subheader("Preview Dataset")

            st.dataframe(
                df_upload.head(10),
                use_container_width=True
            )

            st.info(
                f"Jumlah data: {len(df_upload):,} baris"
            )

            if len(df_upload) == 0:

                st.error("CSV tidak memiliki data.")

            else:

                # -----------------------------------------
                # VALIDASI DATASET
                # -----------------------------------------

                (
                    df_valid,
                    df_invalid,
                    error_df,
                    missing_columns
                ) = validate_csv(df_upload)

                if missing_columns:

                    st.error(
                        "Kolom berikut tidak ditemukan: "
                        + ", ".join(missing_columns)
                    )

                else:

                    st.subheader("Hasil Validasi Dataset")

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Total Data",
                        f"{len(df_upload):,}"
                    )

                    col2.metric(
                        "Data Valid",
                        f"{len(df_valid):,}"
                    )

                    col3.metric(
                        "Data Tidak Valid",
                        f"{len(df_invalid):,}"
                    )

                    # -------------------------------------
                    # DATA TIDAK VALID
                    # -------------------------------------

                    if not df_invalid.empty:

                        st.warning(
                            f"{len(df_invalid):,} baris tidak valid. "
                            "Data tersebut akan dilewati."
                        )

                        with st.expander(
                            "Lihat detail kesalahan"
                        ):

                            st.dataframe(
                                error_df.head(100),
                                use_container_width=True
                            )

                            st.caption(
                                f"Menampilkan maksimal 100 dari "
                                f"{len(error_df):,} baris bermasalah."
                            )

                        invalid_download = df_upload.loc[
                            df_invalid.index
                        ].copy()

                        st.download_button(
                            "Download Data Tidak Valid",
                            data=invalid_download.to_csv(
                                index=False
                            ).encode("utf-8-sig"),
                            file_name="data_bearing_tidak_valid.csv",
                            mime="text/csv"
                        )

                    else:

                        st.success(
                            "Semua data berhasil melewati validasi!"
                        )

                    # -------------------------------------
                    # PROSES DIAGNOSIS
                    # -------------------------------------

                    st.divider()

                    st.subheader("Proses Diagnosis")

                    if df_valid.empty:

                        st.error(
                            "Tidak ada data valid yang dapat diproses."
                        )

                    else:

                        st.info(
                            f"Sebanyak {len(df_valid):,} data "
                            "siap diproses."
                        )

                        run_diagnosis = st.button(
                            "Mulai Diagnosis CSV",
                            type="primary",
                            use_container_width=True,
                            key="run_csv_diagnosis"
                        )

                        if run_diagnosis:

                            if not check_api_connection():

                                st.error(
                                    "FastAPI tidak terhubung. "
                                    "Jalankan backend terlebih dahulu."
                                )

                            else:

                                results = []

                                progress_bar = st.progress(0)

                                status_text = st.empty()

                                total_rows = len(df_valid)

                                # Gunakan beberapa worker agar
                                # pemrosesan batch lebih cepat.
                                # Backend tetap menerima endpoint
                                # /predict satu baris per request.

                                max_workers = 64

                                with ThreadPoolExecutor(
                                    max_workers=max_workers
                                ) as executor:

                                    futures = {
                                        executor.submit(
                                            predict_csv_row,
                                            index,
                                            row
                                        ): index
                                        for index, row
                                        in df_valid.iterrows()
                                    }

                                    completed = 0

                                    for future in as_completed(futures):

                                        try:

                                            results.append(
                                                future.result()
                                            )

                                        except Exception as e:

                                            index = futures[future]

                                            results.append({
                                                "nomor_baris": index + 2,
                                                "prediksi": "Gagal",
                                                "confidence": 0,
                                                "status": str(e)
                                            })

                                        completed += 1

                                        progress_bar.progress(
                                            completed / total_rows
                                        )

                                        status_text.text(
                                            f"Memproses {completed:,} "
                                            f"dari {total_rows:,} data"
                                        )

                                # Urutkan hasil sesuai baris CSV
                                result_df = pd.DataFrame(
                                    results
                                ).sort_values(
                                    "nomor_baris"
                                ).reset_index(drop=True)

                                # Simpan hasil berdasarkan file
                                st.session_state[
                                    "csv_results"
                                ] = result_df

                                st.session_state[
                                    "csv_results_hash"
                                ] = file_hash

                                status_text.success(
                                    "Proses diagnosis selesai!"
                                )

                    # -------------------------------------
                    # TAMPILKAN HASIL
                    # -------------------------------------

                    if (
                        st.session_state.get(
                            "csv_results_hash"
                        ) == file_hash
                        and "csv_results" in st.session_state
                    ):

                        result_df = st.session_state[
                            "csv_results"
                        ]

                        st.divider()

                        st.subheader("Hasil Diagnosis CSV")

                        success_count = (
                            result_df["prediksi"] != "Gagal"
                        ).sum()

                        failed_count = (
                            len(result_df) - success_count
                        )

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Total Diproses",
                            f"{len(result_df):,}"
                        )

                        col2.metric(
                            "Berhasil",
                            f"{success_count:,}"
                        )

                        col3.metric(
                            "Gagal",
                            f"{failed_count:,}"
                        )

                        st.dataframe(
                            result_df,
                            use_container_width=True
                        )

                        # ---------------------------------
                        # DISTRIBUSI PREDIKSI
                        # ---------------------------------

                        st.subheader(
                            "Distribusi Hasil Diagnosis"
                        )

                        successful_results = result_df[
                            result_df["prediksi"] != "Gagal"
                        ]

                        prediction_counts = (
                            successful_results[
                                "prediksi"
                            ].value_counts()
                        )

                        if not prediction_counts.empty:

                            st.bar_chart(
                                prediction_counts
                            )

                        # ---------------------------------
                        # DOWNLOAD HASIL
                        # ---------------------------------

                        result_csv = result_df.to_csv(
                            index=False
                        ).encode("utf-8-sig")

                        st.download_button(
                            "Download Hasil Diagnosis CSV",
                            data=result_csv,
                            file_name="hasil_diagnosis_bearing.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

        except Exception as e:

            st.error(
                f"Terjadi kesalahan saat membaca CSV: {e}"
            )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Bearing Diagnosis System | Machine Learning Project"
)
