"""
Audit sistematis: cari SEMUA kata yang sering muncul di komentar
berlabel Positif, tapi SECARA LEKSIKAL berkonotasi negatif.

Pendekatan: pakai daftar kata "berbahaya" (red flag words) yang
jauh lebih luas, lalu cek frekuensi kemunculannya di tiap kelas
sentimen sekaligus -- supaya kita lihat SEMUA masalah dalam 1x jalan,
bukan ditemukan satu-satu secara manual berulang kali.
"""
import pandas as pd
from collections import Counter

df = pd.read_csv('data/output/mbg_dengan_emosi.csv')
df_komentar = df[df['tipe'] == 'komentar'].copy()

# Daftar kata yang SECARA LEKSIKAL/UMUM dikenal negatif dalam Bahasa Indonesia
# terkait isu kebijakan publik -- disusun lebih luas dari kamus sarkasme sebelumnya
KATA_NEGATIF_LEKSIKAL = [
    'korupsi', 'koruptor', 'korup', 'racun', 'beracun', 'keracunan',
    'maling', 'pencuri', 'mencuri', 'tipu', 'menipu', 'bohong', 'dusta',
    'bubar', 'gagal', 'rusak', 'busuk', 'basi', 'kadaluarsa', 'jamuran',
    'tolong hentikan', 'hentikan', 'stop', 'tutup', 'cabut',
    'becus', 'goblok', 'tolol', 'bangsat', 'anjing', 'kontol', 'bajingan',
    'parah', 'kacau', 'berantakan', 'amburadul', 'memalukan',
    'bohongi', 'bohongin', 'menipu', 'penipuan', 'kebohongan',
    'tanggung jawab', 'tanggungjawab',
]

print("=" * 80)
print("AUDIT: Kata negatif leksikal vs distribusi sentimen (KOMENTAR saja)")
print("=" * 80)

hasil = []
for kata in KATA_NEGATIF_LEKSIKAL:
    mask = df_komentar['teks_asli'].str.lower().str.contains(kata, na=False, regex=False)
    subset = df_komentar[mask]
    if len(subset) == 0:
        continue
    counts = subset['sentimen'].value_counts()
    total = len(subset)
    n_positif = counts.get('Positif', 0)
    pct_positif = n_positif / total * 100 if total > 0 else 0
    hasil.append({
        'kata': kata,
        'total': total,
        'positif': n_positif,
        'negatif': counts.get('Negatif', 0),
        'netral': counts.get('Netral', 0),
        'pct_positif': pct_positif,
    })

df_hasil = pd.DataFrame(hasil).sort_values('pct_positif', ascending=False)
print(df_hasil.to_string(index=False))

print()
print("=" * 80)
print("KATA DENGAN MASALAH SERIUS (>40% masih Positif, total >= 3 kemunculan)")
print("=" * 80)
bermasalah = df_hasil[(df_hasil['pct_positif'] > 40) & (df_hasil['total'] >= 3)]
print(bermasalah.to_string(index=False))

bermasalah.to_csv('audit_kata_bermasalah.csv', index=False)
print("\nTersimpan: audit_kata_bermasalah.csv")
