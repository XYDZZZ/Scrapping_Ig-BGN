"""
=============================================================
  Gabungkan Data Apify — Postingan + Komentar
  Data MBG (Makan Bergizi Gratis)
=============================================================

Menggabungkan:
  - 700_post_-_dataset-ig-scrapper-apify.json (700 postingan, caption lengkap)
  - dataset_200_post-15_komenpost.json (1.910 komentar dari 130 postingan
    yang berhasil diproses sebelum kredit Apify habis)

Output: satu file JSON terstruktur, siap untuk tahap filtering
(01b_filtering_preprocessing.py)
"""

import json
import re
from collections import defaultdict

# ─────────────────────────────────────────────
# KONFIGURASI PATH
# ─────────────────────────────────────────────

PATH_POSTS = "data/mentah/700_post_apify.json"
PATH_COMMENTS = "data/mentah/200_post_komentar_apify.json"
OUTPUT_PATH = "data/output/mbg_data_gabungan.json"


def ekstrak_hashtag(teks: str) -> list:
    if not teks:
        return []
    return re.findall(r'#\w+', teks.lower())


def ekstrak_mention(teks: str) -> list:
    if not teks:
        return []
    return re.findall(r'@\w+', teks.lower())


def main():
    print("=" * 60)
    print("  Menggabungkan Data Apify — Postingan + Komentar")
    print("=" * 60)

    with open(PATH_POSTS, "r", encoding="utf-8") as f:
        posts = json.load(f)
    with open(PATH_COMMENTS, "r", encoding="utf-8") as f:
        comments = json.load(f)

    print(f"[*] Total postingan dimuat: {len(posts)}")
    print(f"[*] Total komentar dimuat: {len(comments)}")

    comments_by_post = defaultdict(list)
    for c in comments:
        comments_by_post[c["postUrl"]].append({
            "username": c.get("ownerUsername", ""),
            "teks": c.get("text", ""),
            "tanggal": (c.get("timestamp", "") or "").replace("T", " ").replace(".000Z", ""),
            "likes": c.get("likesCount", 0) or 0,
        })

    hasil = []
    for post in posts:
        caption = post.get("caption", "") or ""
        url = post.get("url", "")

        data_post = {
            "shortcode"       : post.get("shortCode", ""),
            "url"             : url,
            "tanggal"         : (post.get("timestamp", "") or "").replace("T", " ").replace(".000Z", ""),
            "caption"         : caption,
            "likes"           : post.get("likesCount", 0) or 0,
            "jumlah_komentar" : post.get("commentsCount", 0) or 0,
            "pemilik"         : post.get("ownerUsername", ""),
            "is_video"        : post.get("type", "") == "Video",
            "hashtags"        : ekstrak_hashtag(caption) or post.get("hashtags", []),
            "mentions"        : ekstrak_mention(caption) or post.get("mentions", []),
            "akun_sumber"     : post.get("ownerUsername", ""),
            "komentar"        : comments_by_post.get(url, []),
        }
        hasil.append(data_post)

    total_post = len(hasil)
    post_dengan_komentar = sum(1 for p in hasil if len(p["komentar"]) > 0)
    total_komentar_tergabung = sum(len(p["komentar"]) for p in hasil)

    print(f"\n[*] Total postingan final: {total_post}")
    print(f"[*] Postingan yang punya komentar terambil: {post_dengan_komentar}")
    print(f"[*] Total komentar tergabung: {total_komentar_tergabung}")

    import os
    os.makedirs("output", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(hasil, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Data gabungan tersimpan: {OUTPUT_PATH}")
    print("[OK] File ini siap dipakai untuk 01b_filtering_preprocessing.py")


if __name__ == "__main__":
    main()
