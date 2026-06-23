"""
=============================================================
  Setup Login via Cookie Browser — Alternatif Anti-Checkpoint
=============================================================
Cara ini meminjam cookie session dari browser yang SUDAH login,
bukan login ulang lewat API (yang sering kena checkpoint).

SYARAT SEBELUM JALANKAN FILE INI:
  1. Buka Chrome/Firefox di KOMPUTER INI
  2. Login ke instagram.com seperti biasa (sampai masuk feed, bukan checkpoint)
  3. JANGAN logout setelah itu
  4. Baru jalankan file ini

Install dulu library yang dibutuhkan:
    pip install browser_cookie3 instaloader
"""

import instaloader
import browser_cookie3
import sys

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────

# Pilih browser yang dipakai untuk login Instagram: "chrome", "firefox", "chromium", atau "brave"
BROWSER = "brave"


def ambil_cookie_instagram(browser: str) -> dict:
    """Ambil semua cookie domain instagram.com dari browser yang sudah login."""
    print(f"[🍪] Membaca cookie dari browser: {browser}")

    if browser == "chrome":
        cj = browser_cookie3.chrome(domain_name="instagram.com")
    elif browser == "firefox":
        cj = browser_cookie3.firefox(domain_name="instagram.com")
    elif browser == "chromium":
        cj = browser_cookie3.chromium(domain_name="instagram.com")
    elif browser == "brave":
        cj = browser_cookie3.brave(domain_name="instagram.com")
    else:
        raise ValueError(f"Browser '{browser}' tidak didukung.")

    cookie_dict = {c.name: c.value for c in cj}
    return cookie_dict


def main():
    print("=" * 60)
    print("  Setup Login Instagram via Cookie Browser")
    print("=" * 60)

    try:
        cookie_dict = ambil_cookie_instagram(BROWSER)
    except Exception as e:
        print(f"\n[❌] Gagal membaca cookie dari {BROWSER}: {e}")
        print("\n[💡] Kemungkinan penyebab:")
        print("    - Browser sedang terbuka (tutup dulu semua window, lalu coba lagi)")
        print("    - Browser yang dipakai login beda dari yang dikonfigurasi di BROWSER")
        print("    - Belum pernah login ke instagram.com di browser ini")
        sys.exit(1)

    if not cookie_dict or "sessionid" not in cookie_dict:
        print("\n[❌] Cookie Instagram tidak ditemukan atau tidak lengkap.")
        print("[💡] Pastikan:")
        print("    1. Sudah login PENUH ke instagram.com (sampai masuk feed)")
        print("    2. Bukan masih di halaman checkpoint/verifikasi")
        print("    3. Browser yang dicek sesuai (ubah variabel BROWSER jika perlu)")
        sys.exit(1)

    # Instagram menyimpan ID user di 'ds_user_id' (angka), bukan 'ds_user'.
    # Username asli tidak tersimpan di cookie, jadi kita ambil lewat API setelah autentikasi.
    user_id = cookie_dict.get("ds_user_id", "unknown")
    print(f"[✅] Ditemukan login aktif. User ID: {user_id}")

    # ── Pasang cookie ke instaloader ──────────────────
    L = instaloader.Instaloader(
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=True,
        save_metadata=False,
        quiet=True,
    )

    L.context._session.cookies.update(cookie_dict)
    L.context._session.headers.update({"X-CSRFToken": cookie_dict["csrftoken"]})

    # ── Dapatkan username asli dari user_id numerik, lewat instaloader sendiri ──
    try:
        profile = instaloader.Profile.from_id(L.context, int(user_id))
        username = profile.username
        print(f"[✅] Cookie valid! Berhasil identifikasi akun: @{username}")
    except Exception as e:
        print(f"\n[❌] Cookie tidak valid / gagal mengidentifikasi akun: {e}")
        print("[💡] Coba login ulang di browser, lalu jalankan file ini lagi.")
        sys.exit(1)

    L.context.username = username

    # ── Simpan sebagai session instaloader biasa ──────
    session_file = f"session-{username}"
    L.save_session_to_file(filename=session_file)
    print(f"\n[💾] Session disimpan ke '{session_file}'")
    print("[🎉] Setup selesai! Sekarang jalankan: python 01_scrape_instagram.py")


if __name__ == "__main__":
    main()
