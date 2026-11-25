# Aplikasi Prediksi Random Forest

Aplikasi Streamlit untuk menjalankan `random_forest_regressor_model.pkl` melalui unggah CSV atau tabel input.

## Menjalankan

```powershell
pip install -r requirements.txt
streamlit run app.py
```

File model harus berada satu folder dengan `app.py`. Unduh `template_input.csv` dari aplikasi untuk mendapatkan nama kolom dan urutan fitur yang tepat. Isi nilai numerik pada setiap kolom; fitur kategori pada model sudah diubah menjadi kolom one-hot encoded.

Nama target dan satuan hasil prediksi tidak tersedia di metadata model, jadi aplikasi menampilkan hasil dengan label umum `hasil_prediksi`.

**Keamanan:** file pickle/joblib dapat menjalankan kode saat dimuat. Jalankan aplikasi hanya dengan model yang Anda percayai.