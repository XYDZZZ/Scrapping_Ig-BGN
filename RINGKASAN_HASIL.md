# Ringkasan Hasil — Analisis Sentimen Instagram Program MBG

## Apa yang Dikerjakan
Mengumpulkan dan menganalisis opini publik di Instagram terkait program **Makan Bergizi Gratis (MBG)**, dari akun resmi penyelenggara **@badangizinasional.ri**. Total data: **700 postingan** dan **1.910 komentar publik**.

## Tahapan Analisis
1. **Scraping** data Instagram (postingan + komentar) menggunakan tool Apify
2. **Filtering** — membuang data kosong/duplikat, menyaring hanya yang relevan dengan topik MBG (hasil: 603 postingan + 1.851 komentar)
3. **Text preprocessing (NLP)** — pembersihan teks, normalisasi kata tidak baku, penghapusan stopword, stemming (Sastrawi)
4. **Analisis sentimen** menggunakan model IndoBERT, dengan koreksi tambahan untuk kasus sarkasme (lihat catatan di bawah)
5. **Deteksi emosi** — selain positif/netral/negatif, juga dideteksi 8 jenis emosi spesifik (marah, takut, senang, dll)
6. **Machine learning modeling** — melatih 3 model (Logistic Regression, Naive Bayes, Linear SVM) untuk klasifikasi otomatis, plus analisis kata-kata paling berpengaruh

## Temuan Utama

**Sentimen Caption Resmi BGN** (532 caption):
- Netral 51.7% | Negatif 44.9% | Positif 3.4%
- Caption resmi banyak bernada negatif karena sering berisi *klarifikasi/bantahan* terhadap isu yang beredar (bukan promosi biasa)

**Sentimen Komentar Publik** (1.218 komentar):
- Positif 37.3% | Negatif 33.9% | Netral 28.8%
- Opini publik **terbelah cukup rata** — bukan dukungan besar, juga bukan penolakan besar

**Kata Paling Menandai Komentar Negatif:**
korupsi, racun/keracunan, koruptor, maling, "stop MBG"

**Emosi Dominan:**
Senang (321) dan Marah (199) — namun separuh komentar berkategori "senang" secara leksikal ternyata tetap berlabel sentimen negatif, menunjukkan banyak **sarkasme** ("makan racun gratis 😍")

**Performa Model Klasifikasi (Machine Learning):**
| Model | Akurasi | F1-score |
|---|---|---|
| Logistic Regression | 77.8% | 0.770 |
| Naive Bayes | 76.4% | 0.759 |
| **Linear SVM (terbaik)** | **80.0%** | **0.797** |

## Catatan Metodologi Penting (untuk Bagian Diskusi/Limitation)

1. **Sarkasme adalah tantangan utama.** Model sentimen umum (IndoBERT) awalnya salah mengklasifikasikan banyak komentar bersarkasme sebagai positif (misal "makan racun gratis 😍" karena kata "gratis" dan emoji positif). Dibuat kamus koreksi berbasis pola kata kunci, berhasil mengoreksi 235 dari 285 kasus teridentifikasi.
2. **Audit sistematis dilakukan** terhadap 36 kata kunci negatif — 34 berhasil dikoreksi mendekati 0% kesalahan, 2 kata ("hentikan", "tutup") tetap ambigu secara linguistik dan dilaporkan sebagai limitation, bukan dipaksa dikoreksi.
3. **Komentar diambil dari postingan dengan jumlah komentar terbanyak** (bukan random sampling), sehingga representasinya condong ke postingan yang paling banyak memicu diskusi publik.
4. **Label untuk machine learning** berasal dari hasil IndoBERT+koreksi, bukan anotasi manual — sehingga akurasi model mengukur konsistensi pola linguistik, bukan kebenaran absolut.

## File yang Dihasilkan
- 10 grafik visualisasi (distribusi sentimen, tren waktu, word cloud, emosi, confusion matrix, fitur penting)
- Dataset lengkap dalam format CSV (siap untuk analisis lanjutan)
- Seluruh script Python pipeline (reproducible)
