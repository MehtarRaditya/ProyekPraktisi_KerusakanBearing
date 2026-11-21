# Proyek Praktisi Diagnosis Bearing · Tim Lorem Ipsum

**Kasus:** [4 Diagnosis Bearing](https://github.com/nurhikam/mkb-ml-project/blob/main/kasus/04-diagnosis-bearing/brief.md) · **Aplikasi:** [tautan aplikasi](https://proyekpraktisikerusakanbearing-drwvxutgx7jgafczueygac.streamlit.app/) · **Video:** [tautan video](https://youtu.be/ZGGkjnfCaIQ)

| Anggota | NIM | Bagian yang dikerjakan |
|---|---|---|
| Mehtar Raditya Daniswara Susanto | 2400744 | API, Aplikasi, UI Streamlit, Model, Video Editting |
| Ghaisan Malik Al Jabbar | 2400333 | Model, Revisi Model, Uji Aplikasi, Data, Laporan |

## Masalah dan pengguna

Pengguna utama model ini adalah teknisi pemeliharaan mesin, operator industri, dan praktisi bidang mekatronika yang bertanggung jawab memantau kondisi bearing pada mesin berputar. Model Machine Learning ini membantu pengguna mengidentifikasi kondisi bearing, baik normal maupun mengalami kerusakan pada outer race, inner race, atau elemen bola, berdasarkan karakteristik data sensor dan parameter operasional. Hasil diagnosis dapat digunakan sebagai dasar pengambilan keputusan dalam menentukan kebutuhan inspeksi dan tindakan pemeliharaan mesin.

## Data dan pembersihan

| Masalah yang ditemukan | Jumlah baris | Cara menangani |
|---|---|---|
| Missing Values | 609 | Menggunakan df.isnull() |
| Duplicates | 220 | Menggunakan drop_duplicates |


## Hasil

Evaluasi model dilakukan menggunakan validation set dan test set dengan Macro F1-Score sebagai metrik utama. Logistic Regression digunakan sebagai model baseline, sedangkan Random Forest dipilih sebagai model final berdasarkan hasil perbandingan performa pada validation set.

| Model | Macro F1 *validation* | Macro F1 *test* |
|---|---:|---:|
| *Baseline*: Logistic Regression | 0.7256 | - |
| *Final Model*: Random Forest | 0.7312 | 0.7548 |

### Hasil Evaluasi Model Final

Model Random Forest memperoleh hasil evaluasi sebagai berikut:

| Metrik | Validation | Test |
|---|---:|---:|
| Accuracy | 0.8070 | 0.8203 |
| Precision (Macro) | 0.8212 | 0.8365 |
| Recall (Macro) | 0.6851 | 0.7107 |
| Macro F1-Score | 0.7312 | 0.7548 |

Random Forest terpilih sebagai model final karena menghasilkan Macro F1-Score tertinggi pada validation set dibandingkan model lainnya. Pada test set, model memperoleh Macro F1-Score sebesar 0.7548 dan Accuracy sebesar 0.8203.

Hasil tersebut menunjukkan bahwa model mampu melakukan klasifikasi kondisi bearing dengan performa yang cukup baik pada data pengujian. Namun, performa pengenalan setiap kelas masih bervariasi, sehingga pengembangan lebih lanjut diperlukan untuk meningkatkan kemampuan model dalam mendeteksi seluruh jenis kerusakan bearing.



### Threshold yang Digunakan

**Threshold:** Tidak menggunakan threshold probabilitas tetap (default multiclass prediction).

Model Random Forest menggunakan kelas dengan probabilitas prediksi tertinggi (*argmax*) untuk menentukan kondisi bearing, tanpa menetapkan ambang probabilitas khusus.

Pendekatan ini digunakan karena sistem bertujuan mengklasifikasikan bearing ke dalam empat kondisi, yaitu `normal`, `outer_race`, `inner_race`, dan `bola`. Setiap sampel akan diberikan kategori berdasarkan hasil prediksi dengan probabilitas tertinggi.

Dari sisi kebutuhan operasional, pendekatan ini memungkinkan setiap data memperoleh hasil diagnosis sehingga teknisi dapat menggunakan prediksi sebagai referensi awal untuk menentukan kebutuhan inspeksi dan pemeliharaan bearing.


## Cara menjalankan

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload   # API
streamlit run app.py            # aplikasi
```

## Struktur repositori

```
notebooks/      notebook dari EDA sampai penyimpanan model
data/sample     data mentah (dan data bersih jika perlu)
src/features.py fungsi pembuat fitur, dipakai saat pelatihan dan saat prediksi
artifacts/      model.joblib + metadata.json
app/            API (FastAPI) dan antarmuka (Streamlit)
```

## Keterbatasan

Model Machine Learning yang dikembangkan dalam proyek ini memiliki beberapa keterbatasan yang perlu diperhatikan sebelum digunakan untuk diagnosis kerusakan bearing.

1. **Ketergantungan pada Dataset:** Model hanya dapat memberikan prediksi yang relevan apabila data input memiliki karakteristik dan format yang sesuai dengan dataset yang digunakan selama pelatihan. Prediksi pada data dengan karakteristik yang jauh berbeda berpotensi menghasilkan kesalahan klasifikasi.

2. **Keterbatasan Variasi Data:** Dataset yang digunakan belum tentu mencakup seluruh variasi kondisi kerusakan bearing, jenis bearing, karakteristik mesin, dan lingkungan operasional di dunia industri. Oleh karena itu, kemampuan generalisasi model terhadap mesin atau bearing yang berbeda belum dapat dipastikan.

3. **Tidak Mendukung Prediksi Real-Time:** Model dikembangkan dan dievaluasi menggunakan dataset yang telah dikumpulkan sebelumnya. Model belum dirancang untuk melakukan pemantauan kondisi bearing secara langsung melalui sensor yang terhubung dengan mesin.

4. **Keterbatasan Kategori Diagnosis:** Model hanya mengklasifikasikan empat kategori kondisi bearing, yaitu normal, outer race, inner race, dan bola. Model belum dapat mengidentifikasi tingkat keparahan kerusakan, memprediksi sisa umur bearing (Remaining Useful Life), maupun menentukan waktu terjadinya kegagalan.

5. **Ketergantungan pada Kualitas Data Sensor:** Data sensor yang mengandung noise berlebihan, kesalahan pengukuran, nilai yang tidak valid, atau kondisi sensor yang tidak terkalibrasi dapat memengaruhi hasil prediksi. Preprocessing data tidak sepenuhnya menjamin bahwa seluruh gangguan pengukuran dapat diatasi.

6. **Keterbatasan Validasi di Lingkungan Industri:** Evaluasi model dilakukan menggunakan dataset yang tersedia dan belum membuktikan performa model pada pengujian langsung di lingkungan operasional industri. Oleh karena itu, hasil prediksi belum dapat dijadikan satu-satunya dasar untuk menentukan tindakan pemeliharaan mesin.

### Data yang Belum Tersedia

Beberapa jenis data yang masih diperlukan untuk meningkatkan kemampuan dan keandalan model antara lain:

- Data getaran bearing secara real-time dari mesin industri.
- Data dari berbagai jenis dan ukuran bearing.
- Data kerusakan dengan variasi tingkat keparahan, mulai dari kerusakan awal hingga kerusakan berat.
- Data dari berbagai kondisi operasional, seperti variasi kecepatan putaran, beban, dan temperatur.
- Data historis perkembangan kerusakan bearing untuk mendukung analisis prediksi sisa umur bearing.
- Data pengujian langsung pada mesin industri untuk memvalidasi kemampuan generalisasi model.

Dengan mempertimbangkan keterbatasan tersebut, model ini lebih tepat digunakan sebagai alat bantu klasifikasi awal kondisi bearing dan pendukung analisis pemeliharaan, bukan sebagai pengganti inspeksi teknis maupun sistem pemantauan kondisi mesin secara menyeluruh.

