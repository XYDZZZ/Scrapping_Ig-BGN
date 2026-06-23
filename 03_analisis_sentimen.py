"""
=============================================================
  Analisis Sentimen — Data MBG (Makan Bergizi Gratis)
  Model: IndoBERT (Bahasa Indonesia)
=============================================================

Input: data/output/mbg_preprocessed_final.csv
       (hasil dari 02_filtering_preprocessing.py + terapkan_filter_bot.py)

Output:
  - CSV lengkap dengan label sentimen per baris
  - Grafik distribusi sentimen (caption vs komentar)
  - Word cloud per kategori sentimen
  - Grafik hashtag terpopuler
  - Grafik tren sentimen dari waktu ke waktu
  - Tabel top-N komentar dengan like terbanyak + sentimennya
    (analisis tambahan: opini paling berpengaruh/menonjol)
"""

import pandas as pd
import json
import re
import os
import glob
from datetime import datetime
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

from wordcloud import WordCloud
from transformers import pipeline

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────

OUTPUT_DIR = "data/output"
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
INPUT_CSV = os.path.join(OUTPUT_DIR, "mbg_preprocessed_final.csv")

MODEL_NAME = "mdhugol/indonesia-bert-sentiment-classification"
LABEL_MAP = {
    "LABEL_0": "Negatif",
    "LABEL_1": "Netral",
    "LABEL_2": "Positif",
}

WARNA = {"Positif": "#4CAF50", "Netral": "#2196F3", "Negatif": "#F44336"}


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}\n"
            "Jalankan dulu 02_filtering_preprocessing.py lalu terapkan_filter_bot.py"
        )
    print(f"[📂] Membaca: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")

    if "dicurigai_bot" in df.columns:
        df["dicurigai_bot"] = df["dicurigai_bot"].astype(str).str.strip().str.lower() == "true"

    df["teks_final"] = df["teks_final"].fillna("").astype(str)

    print(f"[📊] Total baris dimuat: {len(df)}")
    print(df["tipe"].value_counts().to_string())
    return df


# ─────────────────────────────────────────────
# 2. ANALISIS SENTIMEN
# ─────────────────────────────────────────────

def init_model_sentimen():
    print("\n[🤖] Loading model IndoBERT... (pertama kali butuh beberapa menit)")
    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        truncation=True,
        max_length=512,
    )
    print("[✅] Model siap!\n")
    return classifier


def prediksi_sentimen_batch(classifier, df: pd.DataFrame, kolom_teks: str = "teks_final") -> pd.DataFrame:
    sentimen_list = []
    skor_list = []
    total = len(df)

    for i, teks in enumerate(df[kolom_teks], 1):
        teks = teks.strip()
        if len(teks) < 3:
            sentimen_list.append("Netral")
            skor_list.append(0.0)
        else:
            try:
                hasil = classifier(teks[:512])[0]
                sentimen_list.append(LABEL_MAP.get(hasil["label"], hasil["label"]))
                skor_list.append(round(hasil["score"], 4))
            except Exception:
                sentimen_list.append("Netral")
                skor_list.append(0.0)

        if i % 100 == 0 or i == total:
            print(f"   [{i:>5}/{total}] diproses...")

    df = df.copy()
    df["sentimen"] = sentimen_list
    df["skor_kepercayaan"] = skor_list
    return df


# ─────────────────────────────────────────────
# 3. VISUALISASI
# ─────────────────────────────────────────────

def plot_distribusi_sentimen(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Distribusi Sentimen — Data Instagram MBG", fontsize=14, fontweight='bold')

    for ax, tipe in zip(axes, ["caption", "komentar"]):
        df_tipe = df[df["tipe"] == tipe]
        if len(df_tipe) == 0:
            continue
        counts = df_tipe["sentimen"].value_counts()
        colors = [WARNA.get(s, "grey") for s in counts.index]
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val}\n({val/len(df_tipe)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=10)
        ax.set_title(f"Sentimen {tipe.capitalize()} (n={len(df_tipe)})")
        ax.set_ylabel("Jumlah")
        ax.set_ylim(0, counts.max() * 1.25)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "01_distribusi_sentimen.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[💾] Gambar disimpan: {path}")


def plot_tren_waktu(df: pd.DataFrame):
    df_caption = df[df["tipe"] == "caption"].copy()
    df_caption["tanggal"] = pd.to_datetime(df_caption["tanggal"], errors="coerce")
    df_caption = df_caption.dropna(subset=["tanggal"])
    if len(df_caption) == 0:
        print("[⚠️] Tidak ada data tanggal valid untuk tren waktu, dilewati.")
        return

    df_caption["minggu"] = df_caption["tanggal"].dt.to_period("W")
    tren = df_caption.groupby(["minggu", "sentimen"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 5))
    for kolom in tren.columns:
        ax.plot(tren.index.astype(str), tren[kolom],
                marker='o', linewidth=2, label=kolom,
                color=WARNA.get(kolom, "grey"))

    ax.set_title("Tren Sentimen Caption Per Minggu — #MBG", fontsize=13, fontweight='bold')
    ax.set_xlabel("Minggu")
    ax.set_ylabel("Jumlah Postingan")
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "02_tren_waktu.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[💾] Gambar disimpan: {path}")


def plot_wordcloud(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Word Cloud per Sentimen — Semua Teks (Caption + Komentar)", fontsize=14, fontweight='bold')

    sentimen_list = ["Positif", "Netral", "Negatif"]
    CMAP = {"Positif": "Greens", "Netral": "Blues", "Negatif": "Reds"}

    for ax, sentimen in zip(axes, sentimen_list):
        teks_gabung = " ".join(df[df["sentimen"] == sentimen]["teks_final"].astype(str))
        if teks_gabung.strip():
            wc = WordCloud(
                width=500, height=350, background_color="white",
                colormap=CMAP[sentimen], max_words=80, collocations=False,
            ).generate(teks_gabung)
            ax.imshow(wc, interpolation="bilinear")
        else:
            ax.text(0.5, 0.5, "Tidak ada data", ha='center', va='center')
        ax.set_title(f"Sentimen {sentimen}", fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "03_wordcloud.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[💾] Gambar disimpan: {path}")


def plot_hashtag_populer(df: pd.DataFrame):
    df_caption = df[df["tipe"] == "caption"]
    semua_hashtag = []
    for h in df_caption.get("hashtag_sumber", []):
        if isinstance(h, str) and h.strip():
            semua_hashtag.extend([t.strip() for t in h.split(",") if t.strip()])

    if not semua_hashtag:
        print("[⚠️] Tidak ada data hashtag untuk divisualisasikan, dilewati.")
        return

    counter = Counter(semua_hashtag)
    top20 = counter.most_common(20)
    labels = [h for h, _ in top20][::-1]
    counts = [c for _, c in top20][::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(labels, counts, color="#5C6BC0", edgecolor="white")
    for bar, val in zip(bars, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=9)
    ax.set_title("20 Sumber/Hashtag Terpopuler", fontsize=13, fontweight='bold')
    ax.set_xlabel("Frekuensi")
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "04_hashtag_populer.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[💾] Gambar disimpan: {path}")


def tampilkan_top_komentar_like(df: pd.DataFrame, n: int = 15):
    df_komentar = df[df["tipe"] == "komentar"].copy()
    if len(df_komentar) == 0:
        print("[⚠️] Tidak ada data komentar untuk top-likes, dilewati.")
        return

    top = df_komentar.sort_values("likes", ascending=False).head(n)

    print(f"\n{'='*70}")
    print(f"  TOP {n} KOMENTAR DENGAN LIKE TERBANYAK (Opini Paling Menonjol)")
    print(f"{'='*70}")
    for _, row in top.iterrows():
        teks = str(row.get("teks_asli", ""))[:100]
        print(f"  ❤️ {row['likes']:>4} | [{row['sentimen']:>8}] {teks}")

    path = os.path.join(OUTPUT_DIR, "top_komentar_like.csv")
    top[["pemilik", "teks_asli", "likes", "sentimen", "skor_kepercayaan"]].to_csv(
        path, index=False, encoding="utf-8-sig"
    )
    print(f"\n[💾] Tabel top komentar disimpan: {path}")


# ─────────────────────────────────────────────
# 4. SIMPAN HASIL & RINGKASAN
# ─────────────────────────────────────────────

def simpan_hasil(df: pd.DataFrame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_csv = os.path.join(OUTPUT_DIR, f"mbg_sentimen_{timestamp}.csv")
    df.to_csv(path_csv, index=False, encoding="utf-8-sig")
    print(f"\n[💾] Hasil sentimen lengkap disimpan: {path_csv}")

    print("\n" + "="*50)
    print("  RINGKASAN ANALISIS SENTIMEN")
    print("="*50)
    for tipe in ["caption", "komentar"]:
        sub = df[df["tipe"] == tipe]
        if len(sub) == 0:
            continue
        print(f"\n📌 {tipe.upper()} (n={len(sub)}):")
        for label, count in sub["sentimen"].value_counts().items():
            pct = count / len(sub) * 100
            print(f"   {label:10s}: {count:4d} ({pct:.1f}%)")

    return path_csv


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Analisis Sentimen — Data MBG Instagram")
    print("=" * 60)

    os.makedirs(FIGURE_DIR, exist_ok=True)

    df = load_data(INPUT_CSV)

    classifier = init_model_sentimen()

    print("[🔄] Memproses sentimen untuk seluruh baris...")
    df = prediksi_sentimen_batch(classifier, df)

    print("\n[📈] Membuat visualisasi...")
    plot_distribusi_sentimen(df)
    plot_tren_waktu(df)
    plot_wordcloud(df)
    plot_hashtag_populer(df)

    tampilkan_top_komentar_like(df, n=15)

    simpan_hasil(df)

    print("\n[🎉] Analisis selesai!")
