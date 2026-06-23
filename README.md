# Analisis Sentimen Instagram — Program MBG (Makan Bergizi Gratis)
Tugas Mata Kuliah Data Mining — Topik NLP

Repo ini berisi pipeline lengkap untuk mengumpulkan dan menganalisis opini publik
di Instagram terkait program Makan Bergizi Gratis (MBG), dari postingan resmi
**@badangizinasional.ri**.

## Daftar Isi
- [Struktur Folder](#struktur-folder)
- [Ringkasan Hasil](#ringkasan-hasil)
- [Cara Instalasi](#cara-instalasi)
- [Cara Menjalankan](#cara-menjalankan)
- [Catatan Metodologi](#catatan-metodologi)

---

## Struktur Folder

```
.
├── 00_setup_login_cookie.py      # Login Instagram via cookie browser (opsional, kalau mau scraping ulang)
├── 01_gabungkan_data_apify.py    # Gabungkan data mentah Apify (posts + comments)
├── 02_filtering_preprocessing.py # Filtering data + text preprocessing NLP
├── 03_analisis_sentimen.py       # Analisis sentimen pakai IndoBERT
├── 04_koreksi_sarkasme.py        # Koreksi kesalahan klasifikasi akibat sarkasme
├── 05_deteksi_emosi.py           # Deteksi 8 kategori emosi (lexicon-based)
├── 06_modeling_ml.py             # Modeling ML (Logistic Regression, Naive Bayes, SVM)
    07_ringkasan_evaluasi.py      # Ringkasan Evaluasi
├── audit_kata_positif.py         # Tool audit kata kunci negatif yang salah label
├── requirements.txt              # Daftar dependency Python
├── RINGKASAN_HASIL.md            # Ringkasan temuan utama (untuk diskusi kelompok)
└── data/
    ├── mentah/                   # Data asli hasil scraping Apify
    └── output/                   # Semua hasil tiap tahap proses + grafik (figures/)
```

## Ringkasan Hasil

Lihat **[RINGKASAN_HASIL.md](RINGKASAN_HASIL.md)** untuk ringkasan temuan utama
dalam bahasa yang mudah dipahami — cocok untuk bahan diskusi kelompok sebelum
menulis laporan lengkap.

Grafik hasil analisis ada di `data/output/figures/`.

---

## Cara Instalasi

Pastikan **Python 3.10+** sudah terinstall di komputer kamu.

### Windows

```powershell
# 1. Buka folder project di Command Prompt / PowerShell / Terminal VSCode
cd path\ke\folder\ini

# 2. Buat virtual environment
python -m venv venv

# 3. Aktifkan virtual environment
venv\Scripts\activate

# 4. Install dependency
pip install -r requirements.txt
```

### Linux / Mac

```bash
# 1. Buka folder project di terminal
cd path/ke/folder/ini

# 2. Buat virtual environment
python3 -m venv venv

# 3. Aktifkan virtual environment
source venv/bin/activate

# 4. Install dependency
pip install -r requirements.txt
```

> **Catatan:** Folder `venv/` **tidak ikut di-commit ke GitHub** (lihat `.gitignore`)
> karena virtual environment bersifat spesifik per sistem operasi — harus dibuat
> ulang di komputer masing-masing, tidak bisa di-copy antar OS.

### Kalau Instalasi PyTorch Lambat/Berat

Secara default, `pip install -r requirements.txt` akan menginstall versi PyTorch
standar yang bisa jadi besar (terutama jika otomatis menyertakan dukungan GPU
NVIDIA yang tidak diperlukan). Untuk versi yang lebih ringan (CPU-only, ±200MB):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers pandas PySastrawi langdetect wordcloud matplotlib seaborn scikit-learn
```

---

## Cara Menjalankan

Data sudah tersedia di folder `data/`, jadi **tidak perlu scraping ulang** untuk
sekadar melihat/melanjutkan analisis. Jalankan berurutan:

```bash
python 02_filtering_preprocessing.py
python 03_analisis_sentimen.py
python 04_koreksi_sarkasme.py
python 05_deteksi_emosi.py
python 06_modeling_ml.py
python 07_ringkasan_evaluasi.py
```

Setiap script akan mencetak progress dan ringkasan hasil di terminal, serta
menyimpan output (CSV dan grafik PNG) ke folder `data/output/`.

### Kalau Mau Scraping Ulang dari Awal (Opsional)

Membutuhkan akun Instagram dan akun Apify (ada free tier $5/bulan). Jalankan
`00_setup_login_cookie.py` dulu, lalu gunakan Apify Console untuk scraping
profil dan komentar — detail lengkap proses ini didiskusikan dalam laporan.

---

## Catatan Metodologi

1. **Sumber data**: scraping dilakukan dari profil resmi (bukan hashtag publik),
   karena endpoint hashtag Instagram API tidak stabil per pertengahan 2026.
2. **Sampling komentar**: diambil dari postingan dengan jumlah komentar
   terbanyak (bukan random sampling), sehingga representasinya condong ke
   postingan yang paling banyak memicu diskusi publik — ini adalah limitation
   yang perlu disebutkan di laporan.
3. **Koreksi sarkasme**: model sentimen umum (IndoBERT) terbukti salah
   mengklasifikasikan banyak komentar bersarkasme sebagai positif. Dibuat
   kamus koreksi berbasis pola kata kunci, divalidasi lewat audit sistematis
   terhadap 36 kata kunci negatif (`audit_kata_positif.py`). 34 dari 36
   berhasil dikoreksi mendekati 0% kesalahan; 2 kata tersisa ("hentikan",
   "tutup") bersifat ambigu secara linguistik dan dilaporkan transparan
   sebagai limitation, bukan dipaksa dikoreksi.
4. **Label untuk machine learning**: berasal dari hasil IndoBERT + koreksi
   sarkasme, bukan anotasi manual — sehingga akurasi model (Logistic
   Regression, Naive Bayes, Linear SVM) mengukur konsistensi pola linguistik
   yang dipelajari ulang, bukan kebenaran absolut terhadap ground truth manual.
