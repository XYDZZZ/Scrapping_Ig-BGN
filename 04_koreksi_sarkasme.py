"""
=============================================================
  Koreksi Sarkasme — Analisis Sentimen MBG
=============================================================

LATAR BELAKANG:
IndoBERT (mdhugol/indonesia-bert-sentiment-classification) terbukti
salah mengklasifikasikan komentar yang mengandung SARKASME sebagai
Positif, padahal maknanya negatif/kritik. Contoh: "makan racun gratis 😍"
diklasifikasikan Positif karena kata "gratis" dan emoji 😍 secara
leksikal berasosiasi positif, padahal seluruh kalimat adalah sindiran.

PENDEKATAN:
1. Bangun kamus pola/kata kunci sarkasme yang umum dipakai dalam
   komentar Bahasa Indonesia terkait kritik sosial-politik.
2. Untuk baris yang (a) cocok dengan pola sarkasme DAN (b) diklasifikasi
   Positif oleh IndoBERT -> override label jadi Negatif, DITANDAI
   sebagai 'dikoreksi_sarkasme' = True (transparan, bisa diaudit).
3. Baris yang tetap terindikasi salah tapi TIDAK match kamus -> 
   dibiarkan apa adanya, ini bagian dari limitation penelitian yang
   dilaporkan secara terbuka, BUKAN disembunyikan.

PENTING: Ini bukan usaha untuk "memperbaiki" hasil supaya terlihat
sempurna, melainkan untuk mengoreksi kesalahan SISTEMATIS yang sudah
teridentifikasi jelas (sarkasme leksikal), sambil tetap transparan
soal cakupan dan keterbatasan koreksi ini.
"""

import pandas as pd
import re

# ─────────────────────────────────────────────
# KAMUS POLA SARKASME
# ─────────────────────────────────────────────
# Setiap pola: kombinasi kata/frasa yang SECARA KUAT mengindikasikan
# sarkasme/ironi ketika muncul bersamaan dalam satu kalimat,
# WALAUPUN kalimat tsb mengandung kata berkonotasi "positif"
# (gratis, bagus, enak, dst) atau emoji positif.

POLA_SARKASME = [
    # (regex pola, penjelasan)
    # -- Berdasarkan audit sistematis kata negatif leksikal pada komentar nyata --
    # CATATAN: pola tanpa \b di awal/akhir kata akar (racun, tipu, dst) sengaja
    # dipakai supaya tetap menangkap kata berimbuhan Bahasa Indonesia seperti
    # 'keracunan', 'ketipu', 'tertipu' -- \b(racun)\b TIDAK match 'keracunan'
    # karena prefix 'ke-' dan suffix '-an' menempel tanpa spasi.
    (r'racun', "mengandung akar kata 'racun' (termasuk 'keracunan', dll)"),
    (r'korup', "mengandung akar kata 'korup' (korupsi/koruptor/korup)"),
    (r'\b(maling|pencuri|mencuri)\b|tipu|menipu|penipuan', "tudingan pencurian/penipuan (termasuk 'tertipu', 'ketipu')"),
    (r'bohong|dusta', "tudingan kebohongan (termasuk berimbuhan)"),
    (r'becus', "frasa 'tidak/gak becus'"),
    (r'busuk|basi|kadaluarsa|jamuran', "kata kondisi makanan rusak/busuk"),
    (r'\bbubar(in|kan)?\b', "seruan membubarkan program"),
    (r'\bstop\s*mbg\b', "seruan stop MBG secara spesifik"),
    (r'\btutup\s*mbg\b|\bmbg\b.*\btutup\b', "seruan tutup MBG secara spesifik"),
    (r'\bgizi\b.*\bmaling\b|\bmaling\b.*\bgizi\b', "tudingan 'maling gizi'"),
    (r'malingberkedok|berkedokgizi|malingkedok', "hashtag/frasa gabungan 'maling berkedok gizi'"),
    (r'\bmaling\s*kedok\b|\bkedok\s*maling\b', "tudingan 'maling berkedok'"),
    (r'amburadul|\bkacau\b|\bparah\b|berantakan|memalukan', "kritik kondisi negatif eksplisit"),
    (r'tanggung\s*jawab', "seruan minta tanggung jawab (konteks menuntut)"),
    (r'(terima\s*kasih|makasih|thanks).*(korup|tipu|bohong)', "ucapan terima kasih + kata negatif (sarkasme)"),
    (r'(korup|tipu|bohong).*(terima\s*kasih|makasih|thanks)', "kata negatif + ucapan terima kasih (sarkasme)"),
    (r'wkwk.{0,15}(korup|racun|tipu|bohong)', "tertawa sinis + kata negatif"),
    (r'(korup|racun|tipu|bohong).{0,15}wkwk', "kata negatif + tertawa sinis"),
    (r'\bketahuan\b.*\b(juga|akhirnya)\b', "sindiran 'ketahuan juga'"),
    (r'\bmantul\b.*(korup|tipu|bohong|gak\s*becus)', "pujian palsu + kritik"),
    (r'\bfak\s*yu\b|\bfuck\s*you\b|\banjing\b|\bbangsat\b|\bkontol\b|\bgoblok\b|\btolol\b|\bbajingan\b', "makian eksplisit"),
]

# Kata-kata yang secara leksikal positif (untuk dokumentasi, bukan dipakai langsung)
KATA_POSITIF_LEKSIKAL = ["gratis", "enak", "bagus", "hebat", "mantap", "keren", "mantul"]


def deteksi_sarkasme(teks: str) -> tuple:
    """
    Cek apakah teks cocok dengan salah satu pola sarkasme.
    Return: (True/False, penjelasan_pola_yang_match atau None)
    """
    if not isinstance(teks, str) or not teks.strip():
        return False, None

    teks_lower = teks.lower()
    for pola, penjelasan in POLA_SARKASME:
        if re.search(pola, teks_lower):
            return True, penjelasan
    return False, None


def koreksi_sarkasme(df: pd.DataFrame, kolom_teks: str = "teks_asli") -> pd.DataFrame:
    """
    Terapkan koreksi sarkasme ke DataFrame hasil analisis sentimen.
    Menambahkan kolom:
      - terindikasi_sarkasme (bool)
      - pola_sarkasme_match (str, penjelasan pola yang cocok)
      - dikoreksi_sarkasme (bool) -- True jika label benar-benar diubah
      - sentimen_asli_sebelum_koreksi (str) -- simpan label asli IndoBERT
    """
    df = df.copy()

    hasil_deteksi = df[kolom_teks].apply(deteksi_sarkasme)
    df["terindikasi_sarkasme"] = hasil_deteksi.apply(lambda x: x[0])
    df["pola_sarkasme_match"] = hasil_deteksi.apply(lambda x: x[1])

    # Simpan label asli sebelum dikoreksi (untuk transparansi/audit)
    df["sentimen_asli_sebelum_koreksi"] = df["sentimen"]

    # Override HANYA jika: terindikasi sarkasme DAN label asli = Positif
    mask_dikoreksi = df["terindikasi_sarkasme"] & (df["sentimen"] == "Positif")
    df.loc[mask_dikoreksi, "sentimen"] = "Negatif"
    df["dikoreksi_sarkasme"] = mask_dikoreksi

    return df


def laporkan_hasil_koreksi(df: pd.DataFrame):
    """Cetak ringkasan transparan soal apa yang dikoreksi dan apa yang masih jadi limitation."""
    n_terindikasi = df["terindikasi_sarkasme"].sum()
    n_dikoreksi = df["dikoreksi_sarkasme"].sum()
    n_terindikasi_tapi_sudah_benar = n_terindikasi - n_dikoreksi

    print("=" * 70)
    print("  LAPORAN TRANSPARANSI — KOREKSI SARKASME")
    print("=" * 70)
    print(f"\nTotal baris terindikasi mengandung pola sarkasme : {n_terindikasi}")
    print(f"  - Berhasil dikoreksi (asalnya salah jadi Positif) : {n_dikoreksi}")
    print(f"  - Sudah benar sejak awal (bukan Positif)          : {n_terindikasi_tapi_sudah_benar}")

    if n_dikoreksi > 0:
        print(f"\nContoh baris yang DIKOREKSI (Positif -> Negatif):")
        contoh = df[df["dikoreksi_sarkasme"]].head(5)
        for _, row in contoh.iterrows():
            print(f"  - \"{str(row['teks_asli'])[:70]}\"")
            print(f"    Pola terdeteksi: {row['pola_sarkasme_match']}")

    # Limitation: baris yang KEMUNGKINAN masih salah tapi tidak match kamus
    # (tidak bisa dipastikan otomatis -- ini perlu disebutkan sebagai limitation umum)
    n_positif_setelah_koreksi = (df["sentimen"] == "Positif").sum()
    print(f"\n[ℹ️] CATATAN LIMITATION:")
    print(f"  Setelah koreksi, masih ada {n_positif_setelah_koreksi} baris berlabel Positif.")
    print(f"  Sebagian dari baris ini BISA SAJA masih mengandung sarkasme yang")
    print(f"  tidak tercakup oleh kamus pola yang dibuat (kamus ini tidak exhaustive).")
    print(f"  Ini adalah limitation yang melekat pada pendekatan rule-based sederhana")
    print(f"  dan keterbatasan model sentimen umum dalam mendeteksi sarkasme kontekstual.")


if __name__ == "__main__":
    import glob
    import os

    # Cari file hasil sentimen terbaru
    # PENTING: selalu mulai dari hasil ASLI IndoBERT (sebelum koreksi sarkasme),
    # supaya setiap kali kamus sarkasme diperbarui, koreksi dihitung ulang dari
    # awal -- bukan menumpuk di atas hasil koreksi versi sebelumnya.
    semua_file = sorted(glob.glob("data/output/mbg_sentimen_*.csv"))
    files = [f for f in semua_file if "final_terkoreksi" not in f]
    if not files:
        files = semua_file  # fallback kalau memang belum ada file asli lain
    if not files:
        raise FileNotFoundError("Tidak ada file mbg_sentimen_*.csv ditemukan. Jalankan dulu 03_analisis_sentimen.py")

    path = files[-1]
    print(f"[📂] Membaca: {path}\n")
    df = pd.read_csv(path, encoding="utf-8-sig")

    df_koreksi = koreksi_sarkasme(df, kolom_teks="teks_asli")
    laporkan_hasil_koreksi(df_koreksi)

    # Simpan hasil final
    output_path = "data/output/mbg_sentimen_final_terkoreksi.csv"
    df_koreksi.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[💾] Hasil final (dengan koreksi sarkasme) disimpan: {output_path}")

    # Ringkasan ulang distribusi sentimen SETELAH koreksi
    print("\n" + "=" * 50)
    print("  DISTRIBUSI SENTIMEN SETELAH KOREKSI SARKASME")
    print("=" * 50)
    for tipe in ["caption", "komentar"]:
        sub = df_koreksi[df_koreksi["tipe"] == tipe]
        if len(sub) == 0:
            continue
        print(f"\n📌 {tipe.upper()} (n={len(sub)}):")
        for label, count in sub["sentimen"].value_counts().items():
            pct = count / len(sub) * 100
            print(f"   {label:10s}: {count:4d} ({pct:.1f}%)")
