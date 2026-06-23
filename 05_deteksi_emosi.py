"""
=============================================================
  Deteksi Emosi — Data MBG (Makan Bergizi Gratis)
=============================================================

Pendekatan: lexicon-based, mirip NRC Emotion Lexicon, disesuaikan
ke Bahasa Indonesia dan konteks isu sosial-politik MBG.

Emosi yang dideteksi: marah, takut, sedih, senang, terkejut,
percaya, antisipasi, jijik.

CATATAN METODOLOGI:
Deteksi emosi BERBEDA dari analisis sentimen (Positif/Netral/Negatif).
Sentimen mengukur ARAH (baik/buruk), sedangkan emosi mengukur JENIS
perasaan spesifik. Dua komentar bisa sama-sama "negatif" tapi berbeda
emosi -- misal kemarahan terhadap kebijakan vs kesedihan terhadap
pelaksanaan di lapangan. Ini memberi nuansa yang tidak tertangkap
oleh sentimen 3 kelas saja.

Pendekatan lexicon-based memiliki keterbatasan: hanya mendeteksi
kata kunci eksplisit, tidak memahami konteks/sarkasme/negasi secara
mendalam (sama seperti keterbatasan yang sudah didokumentasikan
pada tahap analisis sentimen).
"""

import pandas as pd
import re
import os
import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
from wordcloud import WordCloud

OUTPUT_DIR = "data/output"
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

# ─────────────────────────────────────────────
# KAMUS EMOSI BAHASA INDONESIA
# ─────────────────────────────────────────────
# Disusun manual, fokus ke konteks isu sosial-politik/kebijakan publik
# (bukan sekadar terjemahan literal dari NRC Lexicon Inggris)

KAMUS_EMOSI = {
    'marah': [
        'marah', 'murka', 'amarah', 'berang', 'geram', 'emosi', 'protes',
        'demo', 'ancam', 'serang', 'lawan', 'tolak', 'boikot', 'benci',
        'menyerang', 'bubar', 'bubarin', 'korupsi', 'maling', 'tipu',
        'bohong', 'kontol', 'goblok', 'tolol', 'anjing', 'bangsat',
    ],
    'takut': [
        'takut', 'khawatir', 'cemas', 'was-was', 'panik', 'ngeri',
        'resiko', 'risiko', 'bahaya', 'ancaman', 'krisis', 'darurat',
        'waspada', 'terancam', 'rawan', 'rentan', 'racun', 'beracun',
        'keracunan', 'sakit', 'meninggal',
    ],
    'sedih': [
        'sedih', 'menderita', 'susah', 'sulit', 'berat', 'nelangsa',
        'terpuruk', 'merana', 'miskin', 'kemiskinan', 'rugi', 'bangkrut',
        'kecewa', 'putus asa', 'nestapa', 'menyesal', 'kasian', 'kasihan',
        'prihatin', 'memprihatinkan',
    ],
    'senang': [
        'senang', 'bahagia', 'lega', 'puas', 'untung', 'berkah',
        'sukses', 'berhasil', 'gembira', 'bangga', 'optimis', 'harapan',
        'terbantu', 'manfaat', 'bermanfaat', 'terima kasih', 'makasih',
        'alhamdulillah', 'mantap', 'keren', 'bagus',
    ],
    'terkejut': [
        'terkejut', 'kaget', 'mendadak', 'tiba-tiba', 'dadakan',
        'lonjakan', 'melonjak', 'meledak', 'drastis', 'signifikan',
        'ternyata', 'ketahuan', 'astaga', 'masyaallah',
    ],
    'percaya': [
        'percaya', 'yakin', 'optimis', 'dukung', 'setuju', 'apresiasi',
        'andalan', 'terpercaya', 'solid', 'kuat', 'amanah', 'tegas',
    ],
    'antisipasi': [
        'antisipasi', 'persiapan', 'rencana', 'strategi', 'kebijakan',
        'alternatif', 'solusi', 'upaya', 'langkah', 'mitigasi',
        'evaluasi', 'pengawasan', 'tindak lanjut', 'audit',
    ],
    'jijik': [
        'jijik', 'muak', 'menjijikkan', 'kotor', 'busuk', 'basi',
        'jamuran', 'belatung', 'ulat', 'bau', 'rusak', 'kadaluarsa',
    ],
}

EMOSI_EMOJI = {
    'marah': '😡', 'takut': '😨', 'sedih': '😢', 'senang': '😊',
    'terkejut': '😲', 'percaya': '🤝', 'antisipasi': '📋', 'jijik': '🤢',
}

WARNA_EMOSI = {
    'marah': '#e74c3c', 'takut': '#e67e22', 'sedih': '#3498db',
    'senang': '#2ecc71', 'terkejut': '#9b59b6', 'percaya': '#1abc9c',
    'antisipasi': '#f39c12', 'jijik': '#7f8c8d',
}


def hitung_skor_emosi(teks: str) -> dict:
    """Hitung skor setiap kategori emosi berdasarkan jumlah kata kunci yang cocok."""
    if not isinstance(teks, str) or not teks.strip():
        return {emosi: 0 for emosi in KAMUS_EMOSI}

    teks_lower = teks.lower()
    skor = {}
    for emosi, kata_list in KAMUS_EMOSI.items():
        skor[emosi] = sum(1 for kata in kata_list if kata in teks_lower)
    return skor


def deteksi_emosi_dominan(skor: dict) -> str:
    """Tentukan emosi dengan skor tertinggi; 'netral' jika semua skor 0."""
    if max(skor.values()) == 0:
        return 'netral'
    return max(skor, key=skor.get)


def tambah_kolom_emosi(df: pd.DataFrame, kolom_teks: str = "teks_asli") -> pd.DataFrame:
    df = df.copy()
    skor_list = df[kolom_teks].apply(hitung_skor_emosi)

    for emosi in KAMUS_EMOSI:
        df[f"emosi_{emosi}"] = skor_list.apply(lambda s: s[emosi])

    df["emosi_dominan"] = skor_list.apply(deteksi_emosi_dominan)
    return df


def plot_distribusi_emosi(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Analisis Emosi — Komentar & Caption MBG', fontsize=14, fontweight='bold')

    emosi_dist = df['emosi_dominan'].value_counts()
    colors = [WARNA_EMOSI.get(e, '#bdc3c7') for e in emosi_dist.index]
    axes[0].barh(emosi_dist.index, emosi_dist.values, color=colors)
    axes[0].set_title('Distribusi Emosi Dominan', fontweight='bold')
    axes[0].set_xlabel('Jumlah')
    axes[0].invert_yaxis()

    emosi_cols = [f'emosi_{e}' for e in KAMUS_EMOSI.keys()]
    mean_emosi = df[emosi_cols].mean()
    mean_emosi.index = list(KAMUS_EMOSI.keys())
    mean_sorted = mean_emosi.sort_values(ascending=True)
    axes[1].barh(mean_sorted.index, mean_sorted.values,
                 color=[WARNA_EMOSI.get(e, 'gray') for e in mean_sorted.index])
    axes[1].set_title('Rata-rata Intensitas Skor Emosi', fontweight='bold')
    axes[1].set_xlabel('Rata-rata Skor Kata Kunci')

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "05_distribusi_emosi.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[💾] Gambar disimpan: {path}")


def plot_wordcloud_emosi(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    fig.suptitle('Word Cloud per Emosi Dominan', fontsize=16, fontweight='bold')

    for i, emosi in enumerate(KAMUS_EMOSI.keys()):
        subset = df[df['emosi_dominan'] == emosi]
        if len(subset) == 0:
            axes[i].text(0.5, 0.5, 'Data\ntidak cukup', ha='center', va='center')
            axes[i].axis('off')
            continue

        teks_gabung = ' '.join(subset['teks_final'].fillna('').astype(str))
        if len(teks_gabung.strip()) < 10:
            axes[i].axis('off')
            continue

        wc = WordCloud(
            width=400, height=200, background_color='white',
            colormap='YlOrRd' if emosi not in ['senang', 'percaya'] else 'YlGn',
            max_words=40,
        ).generate(teks_gabung)

        axes[i].imshow(wc, interpolation='bilinear')
        axes[i].axis('off')
        emoji = EMOSI_EMOJI.get(emosi, '')
        axes[i].set_title(f'{emoji} {emosi.upper()} (n={len(subset)})', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "06_wordcloud_emosi.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[💾] Gambar disimpan: {path}")


def plot_emosi_vs_sentimen(df: pd.DataFrame):
    """Cross-tab: hubungan antara sentimen (Positif/Netral/Negatif) dan emosi dominan."""
    cross = pd.crosstab(df['emosi_dominan'], df['sentimen'])

    fig, ax = plt.subplots(figsize=(10, 6))
    cross.plot(kind='barh', stacked=True, ax=ax,
               color={'Positif': '#2ecc71', 'Netral': '#3498db', 'Negatif': '#e74c3c'})
    ax.set_title('Hubungan Emosi Dominan dengan Sentimen', fontweight='bold')
    ax.set_xlabel('Jumlah')
    ax.set_ylabel('Emosi Dominan')
    ax.legend(title='Sentimen')
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "07_emosi_vs_sentimen.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[💾] Gambar disimpan: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Deteksi Emosi — Data MBG Instagram")
    print("=" * 60)

    os.makedirs(FIGURE_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "mbg_sentimen_final_terkoreksi.csv")))
    if not files:
        files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "mbg_sentimen_*.csv")))
    if not files:
        raise FileNotFoundError("Tidak ada file hasil sentimen ditemukan.")

    path = files[-1]
    print(f"[📂] Membaca: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")

    print("[🔄] Menghitung skor emosi untuk setiap baris...")
    df = tambah_kolom_emosi(df, kolom_teks="teks_asli")

    print("\n[📊] Distribusi emosi dominan:")
    print(df['emosi_dominan'].value_counts().to_string())

    print("\n[📈] Membuat visualisasi...")
    plot_distribusi_emosi(df)
    plot_wordcloud_emosi(df)
    plot_emosi_vs_sentimen(df)

    output_path = os.path.join(OUTPUT_DIR, "mbg_dengan_emosi.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[💾] Data dengan kolom emosi disimpan: {output_path}")
    print("\n[🎉] Deteksi emosi selesai!")
