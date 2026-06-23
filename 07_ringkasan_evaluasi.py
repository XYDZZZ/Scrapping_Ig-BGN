"""
=============================================================
  TAHAP TAMBAHAN 3 — Ringkasan Evaluasi Komprehensif
  Data MBG (Makan Bergizi Gratis)
=============================================================

Merangkum SELURUH hasil pipeline (scraping -> filtering ->
preprocessing -> sentimen -> koreksi sarkasme -> emosi -> modeling ML)
menjadi satu laporan ringkasan terstruktur, siap dipakai sebagai
acuan utama saat menyusun laporan akhir tugas.

Tidak melakukan komputasi baru -- hanya membaca & merangkum file-file
hasil dari tahap-tahap sebelumnya.
"""

import pandas as pd
import os
import json

OUTPUT_DIR = "data/output"


def cek_file(path: str) -> bool:
    ada = os.path.exists(path)
    status = "✅" if ada else "❌"
    print(f"  {status} {path}")
    return ada


def main():
    print("=" * 70)
    print("  RINGKASAN EVALUASI KOMPREHENSIF — PIPELINE MBG")
    print("=" * 70)

    print("\n[1] CEK KELENGKAPAN FILE OUTPUT")
    print("-" * 70)
    files_check = {
        "Data gabungan & terfilter": f"{OUTPUT_DIR}/mbg_data_filtered.json",
        "Data preprocessing (cleaning, stemming)": f"{OUTPUT_DIR}/mbg_preprocessed_final.csv",
        "Hasil sentimen (sebelum koreksi sarkasme)": None,  # nama file ada timestamp, dicek manual
        "Hasil sentimen final (setelah koreksi sarkasme)": f"{OUTPUT_DIR}/mbg_sentimen_final_terkoreksi.csv",
        "Data dengan label emosi": f"{OUTPUT_DIR}/mbg_emosi.csv",
        "Perbandingan model ML": f"{OUTPUT_DIR}/model_comparison.csv",
        "Top komentar by likes": f"{OUTPUT_DIR}/top_komentar_like.csv",
    }

    status_lengkap = {}
    for nama, path in files_check.items():
        if path is None:
            continue
        status_lengkap[nama] = cek_file(path)

    # ── 2. Ringkasan data ──────────────────────
    print("\n[2] RINGKASAN VOLUME DATA")
    print("-" * 70)
    path_filtered = f"{OUTPUT_DIR}/mbg_data_filtered.json"
    if os.path.exists(path_filtered):
        with open(path_filtered, "r", encoding="utf-8") as f:
            data = json.load(f)
        total_post = len(data)
        total_komentar = sum(len(p.get("komentar", [])) for p in data)
        print(f"  Total postingan relevan : {total_post}")
        print(f"  Total komentar terambil : {total_komentar}")

    # ── 3. Ringkasan sentimen ──────────────────
    print("\n[3] RINGKASAN ANALISIS SENTIMEN (Setelah Koreksi Sarkasme)")
    print("-" * 70)
    path_sentimen = f"{OUTPUT_DIR}/mbg_sentimen_final_terkoreksi.csv"
    if os.path.exists(path_sentimen):
        df_sent = pd.read_csv(path_sentimen, encoding="utf-8-sig")
        for tipe in ["caption", "komentar"]:
            sub = df_sent[df_sent["tipe"] == tipe]
            if len(sub) == 0:
                continue
            print(f"\n  {tipe.upper()} (n={len(sub)}):")
            for label, count in sub["sentimen"].value_counts().items():
                pct = count / len(sub) * 100
                print(f"    {label:10s}: {count:4d} ({pct:.1f}%)")

        n_dikoreksi = df_sent.get("dikoreksi_sarkasme", pd.Series(dtype=bool)).sum()
        print(f"\n  Jumlah baris dikoreksi karena sarkasme: {n_dikoreksi}")
    else:
        print("  ⚠️  File belum ditemukan -- jalankan 03_analisis_sentimen.py dan 04_koreksi_sarkasme.py")

    # ── 4. Ringkasan emosi ─────────────────────
    print("\n[4] RINGKASAN ANALISIS EMOSI")
    print("-" * 70)
    path_emosi = f"{OUTPUT_DIR}/mbg_emosi.csv"
    if os.path.exists(path_emosi):
        df_emosi = pd.read_csv(path_emosi, encoding="utf-8-sig")
        print(df_emosi["emosi_dominan"].value_counts().to_string())
    else:
        print("  ⚠️  File belum ditemukan -- jalankan 05_deteksi_emosi.py")

    # ── 5. Ringkasan modeling ML ───────────────
    print("\n[5] RINGKASAN PERFORMA MODEL MACHINE LEARNING")
    print("-" * 70)
    path_model = f"{OUTPUT_DIR}/model_comparison.csv"
    if os.path.exists(path_model):
        df_model = pd.read_csv(path_model, index_col=0)
        print(df_model.to_string())
        model_terbaik = df_model["F1-Score"].idxmax()
        print(f"\n  🏆 Model terbaik (berdasarkan F1-Score): {model_terbaik}")
    else:
        print("  ⚠️  File belum ditemukan -- jalankan 06_modeling_ml.py")

    # ── 6. Top komentar ────────────────────────
    print("\n[6] TOP 5 KOMENTAR PALING BERPENGARUH (LIKE TERBANYAK)")
    print("-" * 70)
    path_top = f"{OUTPUT_DIR}/top_komentar_like.csv"
    if os.path.exists(path_top):
        df_top = pd.read_csv(path_top, encoding="utf-8-sig").head(5)
        for _, row in df_top.iterrows():
            teks = str(row.get("teks_asli", ""))[:80]
            print(f"  ❤️ {row['likes']:>4} | [{row['sentimen']:>8}] {teks}")
    else:
        print("  ⚠️  File belum ditemukan")

    # ── 7. Daftar visualisasi yang sudah ada ───
    print("\n[7] DAFTAR VISUALISASI YANG TERSEDIA")
    print("-" * 70)
    figure_dir = f"{OUTPUT_DIR}/figures"
    if os.path.exists(figure_dir):
        figs = sorted(os.listdir(figure_dir))
        for f in figs:
            print(f"  📊 {f}")
    else:
        print("  ⚠️  Folder figures belum ada")

    # ── 8. Limitation yang harus disebutkan di laporan ──
    print("\n[8] CATATAN LIMITATION UNTUK LAPORAN (WAJIB DISEBUTKAN)")
    print("-" * 70)
    print("""
  1. Data komentar diambil dari 130 postingan dengan jumlah komentar
     terbanyak (bukan random sampling dari seluruh 700 postingan),
     sehingga representasinya condong ke konten yang paling banyak
     mendapat reaksi/diskusi publik.

  2. Pengambilan komentar dibatasi otomatis ke maksimal 15 komentar
     per postingan oleh sistem scraping (Apify free tier), bukan
     batasan yang kami tentukan sendiri.

  3. Model sentimen IndoBERT menunjukkan kelemahan sistematis dalam
     mendeteksi SARKASME (contoh: "makan racun gratis" awalnya
     diklasifikasikan Positif). Koreksi berbasis kamus pola sarkasme
     berhasil memperbaiki sebagian kasus, namun TIDAK exhaustive --
     kasus sarkasme yang lebih implisit/kontekstual mungkin masih
     salah klasifikasi.

  4. Deteksi emosi menggunakan pendekatan leksikon berbasis kata
     kunci (bukan model machine learning terlatih khusus emosi),
     sehingga akurasinya bergantung pada kelengkapan kamus yang
     dibuat secara manual.

  5. Ditemukan 71 caption dengan atribusi kepemilikan akun yang tidak
     konsisten (kemungkinan akibat collaborative post atau keterbatasan
     tool scraping pihak ketiga) -- konten tetap dipertahankan karena
     relevan dengan topik, namun diberi anotasi transparan.
""")

    print("=" * 70)
    print("  RINGKASAN SELESAI — Siap dipakai sebagai acuan laporan akhir")
    print("=" * 70)


if __name__ == "__main__":
    main()
