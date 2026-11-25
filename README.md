# Aplikasi Prediksi Random Forest

Aplikasi Streamlit untuk menjalankan `random_forest_regressor_model.pkl` melalui unggah CSV atau form input terarah.

## Menjalankan

```powershell
pip install -r requirements.txt
streamlit run app.py
```

File model harus berada satu folder dengan `app.py`. Unduh `template_input.csv` dari aplikasi untuk mendapatkan nama kolom dan urutan fitur yang tepat. Form input terarah mengelompokkan data profil, kesehatan, polis, serta klaim dan tindakan. Nilai angka dimulai kosong dan harus diisi; fitur kategori one-hot ditampilkan sebagai pilihan.

Nama target dan satuan hasil prediksi tidak tersedia di metadata model, jadi aplikasi menampilkan hasil dengan label umum `hasil_prediksi`.

**Keamanan:** file pickle/joblib dapat menjalankan kode saat dimuat. Jalankan aplikasi hanya dengan model yang Anda percayai.