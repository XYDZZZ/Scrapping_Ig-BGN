"""
=============================================================
  TAHAP FILTERING & PREPROCESSING DATA — Sebelum Analisis Sentimen
  Tugas Mata Kuliah: Data Mining — Topik: NLP
  Data: Instagram MBG (Makan Bergizi Gratis)
=============================================================

Pipeline ini WAJIB dijalankan sebelum 02_analisis_sentimen.py
Urutan file: 01_scrape -> 01b_filtering (file ini) -> 02_analisis_sentimen

Tahapan dalam script ini:
  1. FILTERING (level data)
     1a. Buang caption/komentar kosong
     1b. Buang duplikat (shortcode/teks sama)
     1c. Filter bahasa (hanya Bahasa Indonesia)
     1d. Filter relevansi topik (mengandung kata kunci MBG)
     1e. Deteksi & buang akun bot/spam

  2. TEXT PREPROCESSING (level teks, NLP)
     2a. Cleaning (URL, mention, hashtag, simbol, angka)
     2b. Case folding (lowercase)
     2c. Normalisasi kata alay/slang -> baku
     2d. Tokenisasi
     2e. Stopword removal
     2f. Stemming (Sastrawi)
     2g. Filter panjang token minimum
"""

import json
import re
import os
import glob
import pandas as pd
from datetime import datetime
from collections import Counter

# pip install PySastrawi langdetect pandas
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from langdetect import detect, LangDetectException

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────

OUTPUT_DIR = "data/output"

# Kata kunci wajib ada (minimal salah satu) agar dianggap relevan dengan topik MBG
KATA_KUNCI_RELEVAN = [
    "mbg", "makan bergizi", "bergizi gratis", "makan siang gratis",
    "program makan", "gizi gratis", "dapur mbg", "sppg",
]

# Jumlah minimum kata setelah cleaning agar teks tidak dianggap noise
MIN_JUMLAH_KATA = 3

# Batas jumlah post per akun dalam dataset — di atas ini dicurigai bot/spam
BATAS_POST_PER_AKUN = 8

# Kamus normalisasi kata tidak baku -> baku (bisa terus ditambah)
KAMUS_NORMALISASI = {
    "yg": "yang", "ga": "tidak", "gak": "tidak", "nggak": "tidak",
    "tdk": "tidak", "udh": "sudah", "udah": "sudah", "sdh": "sudah",
    "blm": "belum", "blum": "belum", "dgn": "dengan", "dr": "dari",
    "utk": "untuk", "buat": "untuk", "krn": "karena", "karna": "karena",
    "jd": "jadi", "jdi": "jadi", "bgt": "banget", "bener": "benar",
    "emg": "memang", "emang": "memang", "knp": "kenapa", "gmn": "bagaimana",
    "gimana": "bagaimana", "tp": "tapi", "tapi": "tapi", "klo": "kalau",
    "kalo": "kalau", "sm": "sama", "sama2": "sama-sama", "bgmn": "bagaimana",
    "anak2": "anak-anak", "skrg": "sekarang", "skg": "sekarang",
    "trs": "terus", "trus": "terus", "dpt": "dapat", "dapet": "dapat",
    "msh": "masih", "org": "orang", "ortu": "orang tua", "sklh": "sekolah",
    "sekolahan": "sekolah", "mantul": "bagus", "mantap": "bagus",
    "keren": "bagus", "gizi2": "gizi", "anjir": "", "anjay": "",
}


# ─────────────────────────────────────────────
# 1. LOAD DATA HASIL SCRAPING
# ─────────────────────────────────────────────

def load_data_mentah(output_dir: str) -> list:
    # Cek dulu file gabungan dari Apify (sudah lewat dedup & filter relevansi)
    path_filtered = os.path.join(output_dir, "mbg_data_filtered.json")
    if os.path.exists(path_filtered):
        print(f"[📂] Membaca data: {path_filtered}")
        with open(path_filtered, "r", encoding="utf-8") as f:
            return json.load(f)

    json_files = sorted(glob.glob(os.path.join(output_dir, "mbg_data_*.json")))
    if not json_files:
        json_files = sorted(glob.glob("mbg_data.json"))
    if not json_files:
        raise FileNotFoundError(
            "Tidak ada file data mentah ditemukan. Jalankan dulu 01_gabungkan_data_apify.py"
        )
    path = json_files[-1]
    print(f"[📂] Membaca data mentah: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# 2. TAHAP FILTERING (LEVEL DATA / POSTINGAN)
# ─────────────────────────────────────────────

def filter_kosong_dan_duplikat(data: list) -> list:
    """1a & 1b — Buang caption kosong dan duplikat shortcode."""
    seen = set()
    hasil = []
    for post in data:
        caption = (post.get("caption") or "").strip()
        if not caption:
            continue
        if post["shortcode"] in seen:
            continue
        seen.add(post["shortcode"])
        hasil.append(post)

    print(f"[🧹] Filter kosong & duplikat: {len(data)} -> {len(hasil)} postingan")
    return hasil


def deteksi_bahasa(teks: str) -> str:
    """Deteksi bahasa teks, return kode bahasa ('id', 'en', dst) atau 'unknown'."""
    try:
        # langdetect butuh teks minimal beberapa karakter
        teks_bersih = re.sub(r'[^\w\s]', ' ', teks)
        if len(teks_bersih.strip()) < 10:
            return "unknown"
        return detect(teks_bersih)
    except LangDetectException:
        return "unknown"


def filter_bahasa(data: list) -> list:
    """1c — Hanya pertahankan caption berbahasa Indonesia (atau tidak terdeteksi, biar tidak terlalu agresif)."""
    hasil = []
    for post in data:
        bahasa = deteksi_bahasa(post["caption"])
        post["bahasa_terdeteksi"] = bahasa
        # 'id' = Indonesia. 'unknown' tetap dipertahankan (caption sangat pendek, mis. cuma emoji+hashtag)
        if bahasa in ("id", "unknown"):
            hasil.append(post)

    print(f"[🧹] Filter bahasa: {len(data)} -> {len(hasil)} postingan (Bahasa Indonesia)")
    return hasil


def filter_relevansi_topik(data: list, kata_kunci: list) -> list:
    """1d — Hanya pertahankan post yang benar-benar membahas topik MBG."""
    hasil = []
    for post in data:
        teks_cek = post["caption"].lower()
        if any(kw in teks_cek for kw in kata_kunci):
            hasil.append(post)

    print(f"[🧹] Filter relevansi topik MBG: {len(data)} -> {len(hasil)} postingan")
    return hasil


def deteksi_dan_filter_bot(data: list, batas_post: int) -> list:
    """
    1e — Deteksi indikasi bot/spam sederhana:
    Jika satu akun muncul terlalu sering (di atas batas_post) dalam dataset,
    kemungkinan itu akun bisnis/bot yang posting berulang -> tandai, bukan langsung dihapus.
    (Untuk tugas kuliah, lebih aman MENANDAI lalu didiskusikan, daripada menghapus diam-diam)
    """
    jumlah_per_akun = Counter(post["pemilik"] for post in data)
    akun_dicurigai = {akun for akun, jumlah in jumlah_per_akun.items() if jumlah > batas_post}

    for post in data:
        post["dicurigai_bot"] = post["pemilik"] in akun_dicurigai

    n_dicurigai = sum(1 for p in data if p["dicurigai_bot"])
    print(f"[⚠️] Akun dengan post > {batas_post}x (dicurigai bot/bisnis): "
          f"{len(akun_dicurigai)} akun, {n_dicurigai} postingan ditandai")
    print("     (postingan ini DITANDAI, bukan dihapus — silakan putuskan apakah dikecualikan)")

    return data


# ─────────────────────────────────────────────
# 3. TEXT PREPROCESSING (LEVEL TEKS, UNTUK NLP)
# ─────────────────────────────────────────────

factory_stopword = StopWordRemoverFactory()
stopword_remover  = factory_stopword.create_stop_word_remover()

factory_stemmer = StemmerFactory()
stemmer         = factory_stemmer.create_stemmer()


def cleaning_teks(teks: str) -> str:
    """2a — Hapus URL, mention, hashtag, emoji/simbol, angka."""
    if not teks:
        return ""
    teks = re.sub(r'http\S+|www\.\S+', ' ', teks)                 # URL
    teks = re.sub(r'@\w+', ' ', teks)                              # mention
    teks = re.sub(r'#\w+', ' ', teks)                              # hashtag
    teks = re.sub(r'[\U0001F300-\U0001FAFF\U00002600-\U000027BF]', ' ', teks)  # emoji
    teks = re.sub(r'[^\w\s]', ' ', teks)                           # tanda baca/simbol
    teks = re.sub(r'\d+', ' ', teks)                               # angka
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks


def case_folding(teks: str) -> str:
    """2b — Lowercase semua teks."""
    return teks.lower()


def normalisasi_kata(teks: str, kamus: dict) -> str:
    """2c — Ganti kata tidak baku/slang menjadi kata baku berdasarkan kamus."""
    kata_list = teks.split()
    hasil = [kamus.get(kata, kata) for kata in kata_list]
    hasil = [k for k in hasil if k != ""]  # buang kata yang dipetakan ke string kosong
    return " ".join(hasil)


def tokenisasi(teks: str) -> list:
    """2d — Pecah teks menjadi token/kata."""
    return teks.split()


def hapus_stopword(teks: str) -> str:
    """2e — Hapus kata umum yang tidak bermakna (stopword) bahasa Indonesia."""
    return stopword_remover.remove(teks)


def stemming(teks: str) -> str:
    """2f — Ubah kata berimbuhan menjadi kata dasar (mis. 'membantu' -> 'bantu')."""
    return stemmer.stem(teks)


def preprocess_lengkap(teks: str) -> dict:
    """Jalankan seluruh pipeline preprocessing pada satu teks, simpan setiap tahap (untuk transparansi laporan)."""
    t0_asli       = teks or ""
    t1_cleaning   = cleaning_teks(t0_asli)
    t2_casefold   = case_folding(t1_cleaning)
    t3_normalisasi= normalisasi_kata(t2_casefold, KAMUS_NORMALISASI)
    t4_stopword   = hapus_stopword(t3_normalisasi)
    t5_stemming   = stemming(t4_stopword)
    token_akhir   = tokenisasi(t5_stemming)

    return {
        "teks_asli"        : t0_asli,
        "setelah_cleaning" : t1_cleaning,
        "setelah_casefold" : t2_casefold,
        "setelah_normalisasi": t3_normalisasi,
        "setelah_stopword" : t4_stopword,
        "setelah_stemming" : t5_stemming,
        "jumlah_token"     : len(token_akhir),
        "teks_final"       : t5_stemming,   # ini yang dipakai untuk model sentimen
    }


def filter_panjang_minimum(rows: list, min_kata: int) -> list:
    """2g — Buang teks yang setelah preprocessing terlalu pendek (noise, tidak informatif)."""
    hasil = [r for r in rows if r["preprocessing"]["jumlah_token"] >= min_kata]
    print(f"[🧹] Filter panjang minimum ({min_kata} kata): {len(rows)} -> {len(hasil)} item")
    return hasil


# ─────────────────────────────────────────────
# 4. JALANKAN SELURUH PIPELINE
# ─────────────────────────────────────────────

def proses_komentar(data: list) -> list:
    """Terapkan preprocessing yang sama untuk semua komentar di setiap post."""
    for post in data:
        komentar_valid = []
        for komentar in post.get("komentar", []):
            teks = (komentar.get("teks") or "").strip()
            if not teks:
                continue
            komentar["preprocessing"] = preprocess_lengkap(teks)
            komentar_valid.append(komentar)
        post["komentar"] = komentar_valid
    return data


def buat_dataframe_final(data: list) -> pd.DataFrame:
    """Satukan caption + komentar yang sudah lolos filter ke satu DataFrame siap pakai untuk analisis sentimen."""
    rows = []

    for post in data:
        rows.append({
            "tipe"            : "caption",
            "shortcode"       : post["shortcode"],
            "pemilik"         : post["pemilik"],
            "tanggal"         : post["tanggal"],
            "likes"           : post.get("likes", 0),
            "dicurigai_bot"   : post.get("dicurigai_bot", False),
            "bahasa"          : post.get("bahasa_terdeteksi", "unknown"),
            "hashtag_sumber"  : post.get("hashtag_sumber", ""),
            "preprocessing"   : post["preprocessing"],
        })
        for komentar in post.get("komentar", []):
            rows.append({
                "tipe"            : "komentar",
                "shortcode"       : post["shortcode"],
                "pemilik"         : komentar.get("username", ""),
                "tanggal"         : komentar.get("tanggal", ""),
                "likes"           : 0,
                "dicurigai_bot"   : False,
                "bahasa"          : "unknown",
                "hashtag_sumber"  : post.get("hashtag_sumber", ""),
                "preprocessing"   : komentar["preprocessing"],
            })

    rows = filter_panjang_minimum(rows, MIN_JUMLAH_KATA)

    # Flatten kolom preprocessing supaya enak dibaca di CSV/Excel
    df_rows = []
    for r in rows:
        flat = {k: v for k, v in r.items() if k != "preprocessing"}
        flat.update(r["preprocessing"])
        df_rows.append(flat)

    return pd.DataFrame(df_rows)


def simpan_hasil(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"mbg_preprocessed_{timestamp}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n[💾] Data hasil preprocessing tersimpan: {path}")
    print("     File ini siap dipakai sebagai INPUT untuk 02_analisis_sentimen.py")
    return path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  FILTERING & PREPROCESSING DATA — MBG Instagram (Tugas NLP)")
    print("=" * 65)

    # 1. Load
    data = load_data_mentah(OUTPUT_DIR)
    print(f"[📊] Total data mentah: {len(data)} postingan\n")

    # 2. FILTERING (level data)
    print("── TAHAP 1: FILTERING DATA ──────────────────")
    data = filter_kosong_dan_duplikat(data)
    data = filter_bahasa(data)
    data = filter_relevansi_topik(data, KATA_KUNCI_RELEVAN)
    data = deteksi_dan_filter_bot(data, BATAS_POST_PER_AKUN)

    # 3. PREPROCESSING TEKS (level kata, untuk NLP)
    print("\n── TAHAP 2: TEXT PREPROCESSING (NLP) ────────")
    for post in data:
        post["preprocessing"] = preprocess_lengkap(post["caption"])
    data = proses_komentar(data)
    print(f"[✅] Preprocessing selesai untuk {len(data)} caption + seluruh komentarnya")

    # 4. Gabungkan jadi 1 DataFrame final
    df_final = buat_dataframe_final(data)
    print(f"\n[📊] Total baris siap dianalisis (caption + komentar): {len(df_final)}")

    # 5. Simpan
    simpan_hasil(df_final, OUTPUT_DIR)

    print("\n[🎉] Tahap filtering & preprocessing selesai!")
    print("     Lanjutkan ke 02_analisis_sentimen.py untuk analisis sentimen.")
