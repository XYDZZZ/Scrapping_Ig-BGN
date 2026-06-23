"""
=============================================================
  TAHAP TAMBAHAN 1 — Deteksi Emosi
  Data MBG (Makan Bergizi Gratis)
=============================================================

Mendeteksi emosi dominan pada setiap caption/komentar menggunakan
kamus leksikon emosi Bahasa Indonesia (pendekatan berbasis kata kunci,
terinspirasi dari NRC Emotion Lexicon yang diadaptasi ke Bahasa Indonesia).

8 kategori emosi: Marah, Takut, Sedih, Senang, Terkejut, Percaya,
Antisipasi, Jijik -- ditambah Netral untuk teks tanpa kata kunci emosi.

Input : data/output/mbg_sentimen_final_terkoreksi.csv
Output: data/output/mbg_emosi.csv + visualisasi
"""

import pandas as pd
import re
import os
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
from wordcloud import WordCloud

OUTPUT_DIR = "data/output"
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
INPUT_CSV = os.path.join(OUTPUT_DIR, "mbg_sentimen_final_terkoreksi.csv")

# ─────────────────────────────────────────────
# KAMUS EMOSI BAHASA INDONESIA
# ─────────────────────────────────────────────
KAMUS_EMOSI = {
    "marah": [
        "marah", "murka", "amarah", "berang", "geram", "emosi", "protes",
        "demo", "ancam", "serang", "lawan", "tolak", "boikot", "benci",
        "menyerang", "bubarin", "bubarkan", "stop", "korupsi", "maling",
        "kontol", "tipu", "bohong", "bodoh", "goblok", "anjing",
    ],
    "takut": [
        "takut", "khawatir", "cemas", "was-was", "panik", "ngeri",
        "resiko", "risiko", "bahaya", "ancaman", "krisis", "darurat",
        "waspada", "terancam", "rawan", "rentan", "beracun", "racun",
    ],
    "sedih": [
        "sedih", "menderita", "susah", "sulit", "berat", "nelangsa",
        "terpuruk", "merana", "miskin", "kemiskinan", "rugi", "bangkrut",
        "kecewa", "putus asa", "nestapa", "menyesal", "kasian", "kasihan",
        "honorer", "gaji kecil", "150 rb", "tipis",
    ],
    "senang": [
        "senang", "bahagia", "lega", "puas", "untung", "berkah",
        "sukses", "berhasil", "gembira", "bangga", "optimis", "harapan",
        "terima kasih", "makasih", "alhamdulillah", "mantap", "bagus",
        "enak", "lezat", "bergizi",
    ],
    "terkejut": [
        "terkejut", "kaget", "mendadak", "tiba-tiba", "dadakan",
        "lonjakan", "melonjak", "meledak", "drastis", "signifikan",
        "ketahuan", "viral",
    ],
    "percaya": [
        "percaya", "yakin", "optimis", "dukung", "setuju", "apresiasi",
        "andalan", "terpercaya", "solid", "kuat", "amanah", "transparan",
    ],
    "antisipasi": [
        "antisipasi", "persiapan", "rencana", "strategi", "kebijakan",
        "alternatif", "solusi", "upaya", "langkah", "mitigasi",
        "evaluasi", "tindak lanjut", "pantau", "awasi",
    ],
    "jijik": [
        "jijik", "muak", "menjijikkan", "busuk", "bau", "kotor",
        "basi", "kadaluarsa", "kecoa", "ulat", "rusak",
    ],
}

EMOSI_EMOJI = {
    "marah": "😡", "takut": "😨", "sedih": "😢", "senang": "😊",
    "terkejut": "😲", "percaya": "🤝", "antisipasi": "🔮", "jijik": "🤢",
    "netral": "😐",
}

WARNA_EMOSI = {
    "marah": "#E24B4A", "takut": "#BA7517", "sedih": "#378ADD",
    "senang": "#639922", "terkejut": "#7F77DD", "percaya": "#1D9E75",
    "antisipasi": "#D85A30", "jijik": "#5F5E5A", "netral": "#888780",
}


def hitung_skor_emosi(teks: str) -> dict:
    """Hitung skor setiap kategori emosi berdasarkan jumlah kata kunci yang cocok."""
    if not isinstance(teks, str) or not teks.strip():
        return {emosi: 0 for emosi in KAMUS_EMOSI}

    teks_lower = teks.lower()
    skor = {}
    for emosi, kata_kunci in KAMUS_EMOSI.items():
        skor[emosi] = sum(1 for kata in kata_kunci if kata in teks_lower)
    return skor


def tentukan_emosi_dominan(skor: dict) -> str:
    """Pilih emosi dengan skor tertinggi; jika semua 0, kembalikan 'netral'."""
    if max(skor.values()) == 0:
        return "netral"
    return max(skor, key=skor.get)


def deteksi_emosi_dataframe(df: pd.DataFrame, kolom_teks: str = "teks_asli") -> pd.DataFrame:
    df = df.copy()
    semua_skor = df[kolom_teks].apply(hitung_skor_emosi)

    for emosi in KAMUS_EMOSI:
        df[f"emosi_{emosi}"] = semua_skor.apply(lambda s: s[emosi])

    df["emosi_dominan"] = semua_skor.apply(tentukan_emosi_dominan)
    return df


# ─────────────────────────────────────────────
# VISUALISASI
# ─────────────────────────────────────────────

def plot_distribusi_emosi(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Analisis Emosi — Data Instagram MBG", fontsize=14, fontweight="bold")

    emosi_dist = df["emosi_dominan"].value_counts()
    colors = [WARNA_EMOSI.get(e, "#999") for e in emosi_dist.index]
    axes[0].barh(emosi_dist.index, emosi_dist.values, color=colors)
    axes[0].set_title("Distribusi Emosi Dominan", fontweight="bold")
    axes[0].set_xlabel("Jumlah baris")
    axes[0].invert_yaxis()

    emosi_cols = [f"emosi_{e}" for e in KAMUS_EMOSI]
    mean_emosi = df[emosi_cols].mean()
    mean_emosi.index = list(KAMUS_EMOSI.keys())
    mean_emosi_sorted = mean_emosi.sort_values()
    axes[1].barh(
        mean_emosi_sorted.index, mean_emosi_sorted.values,
        color=[WARNA_EMOSI.get(e, "#999") for e in mean_emosi_sorted.index]
    )
    axes[1].set_title("Rata-rata Intensitas Skor Emosi", fontweight="bold")
    axes[1].set_xlabel("Rata-rata jumlah kata kunci cocok")

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "05_distribusi_emosi.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[💾] Gambar disimpan: {path}")


def plot_wordcloud_emosi(df: pd.DataFrame):
    daftar_emosi = list(KAMUS_EMOSI.keys()) + ["netral"]
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()
    fig.suptitle("WordCloud per Emosi Dominan", fontsize=15, fontweight="bold")

    for i, emosi in enumerate(daftar_emosi):
        subset = df[df["emosi_dominan"] == emosi]
        if len(subset) == 0:
            axes[i].text(0.5, 0.5, "Data\ntidak cukup", ha="center", va="center")
            axes[i].axis("off")
            continue

        teks_gabung = " ".join(subset["teks_final"].fillna("").astype(str))
        if len(teks_gabung.strip()) < 10:
            axes[i].axis("off")
            continue

        wc = WordCloud(
            width=350, height=220, background_color="white",
            colormap="RdYlGn" if emosi in ["senang", "percaya"] else "YlOrRd",
            max_words=30,
        ).generate(teks_gabung)
        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].axis("off")
        emoji = EMOSI_EMOJI.get(emosi, "")
        axes[i].set_title(f"{emoji} {emosi.capitalize()} (n={len(subset)})", fontweight="bold", fontsize=11)

    for j in range(len(daftar_emosi), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "06_wordcloud_emosi.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[💾] Gambar disimpan: {path}")


def plot_heatmap_korelasi_emosi(df: pd.DataFrame):
    import seaborn as sns
    import numpy as np

    emosi_cols = [f"emosi_{e}" for e in KAMUS_EMOSI]
    corr = df[emosi_cols].corr()
    corr.columns = list(KAMUS_EMOSI.keys())
    corr.index = list(KAMUS_EMOSI.keys())

    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                mask=mask, linewidths=0.5, annot_kws={"size": 9})
    plt.title("Korelasi Antar Emosi — Data MBG", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "07_korelasi_emosi.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[💾] Gambar disimpan: {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Deteksi Emosi — Data MBG Instagram")
    print("=" * 60)

    os.makedirs(FIGURE_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    print(f"[📂] Total baris dimuat: {len(df)}")

    print("\n[🔄] Mendeteksi emosi untuk setiap baris...")
    df = deteksi_emosi_dataframe(df, kolom_teks="teks_asli")

    print("\n[📊] Distribusi emosi dominan:")
    print(df["emosi_dominan"].value_counts().to_string())

    print("\n[📈] Membuat visualisasi...")
    plot_distribusi_emosi(df)
    plot_wordcloud_emosi(df)
    plot_heatmap_korelasi_emosi(df)

    output_path = os.path.join(OUTPUT_DIR, "mbg_emosi.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[💾] Data dengan label emosi disimpan: {output_path}")

    print("\n[🎉] Deteksi emosi selesai!")
    print("     Lanjutkan ke 06_modeling_ml.py untuk tahap modeling ML.")
