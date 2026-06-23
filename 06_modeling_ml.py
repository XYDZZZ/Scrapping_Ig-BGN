"""
=============================================================
  TAHAP TAMBAHAN 2 — Modeling Machine Learning
  Data MBG (Makan Bergizi Gratis)
=============================================================

Melatih 3 model klasifikasi teks klasik untuk memprediksi sentimen,
menggunakan label hasil IndoBERT (yang sudah dikoreksi sarkasme)
sebagai ground truth:
  1. Logistic Regression
  2. Multinomial Naive Bayes
  3. Linear SVM

Fitur: TF-IDF (unigram + bigram)

Tujuan: membandingkan performa model klasik vs model deep learning
(IndoBERT), serta mengidentifikasi kata-kata yang paling berpengaruh
terhadap masing-masing kelas sentimen (interpretability).

Input : data/output/mbg_sentimen_final_terkoreksi.csv
Output: data/output/model_comparison.csv + visualisasi
"""

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import seaborn as sns

OUTPUT_DIR = "data/output"
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
INPUT_CSV = os.path.join(OUTPUT_DIR, "mbg_sentimen_final_terkoreksi.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ─────────────────────────────────────────────
# 1. LOAD & SIAPKAN DATA
# ─────────────────────────────────────────────

def siapkan_data():
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    print(f"[📂] Total baris dimuat: {len(df)}")

    df = df[df["teks_final"].fillna("").str.strip().str.len() >= 3].copy()
    print(f"[🧹] Setelah buang teks kosong/pendek: {len(df)}")

    X = df["teks_final"].fillna("")
    y = df["sentimen"]

    print(f"\n[📊] Distribusi label sebelum split:")
    print(y.value_counts().to_string())

    return X, y, df


# ─────────────────────────────────────────────
# 2. TRAINING & EVALUASI
# ─────────────────────────────────────────────

def latih_dan_evaluasi(X_train, X_test, y_train, y_test, vectorizer):
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model_list = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(random_state=RANDOM_STATE, class_weight="balanced", max_iter=5000),
    }

    hasil = {}
    for nama, model in model_list.items():
        print(f"\n[🤖] Melatih {nama}...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        akurasi = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"   Akurasi  : {akurasi:.4f}")
        print(f"   F1-score : {f1:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall   : {recall:.4f}")
        print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

        hasil[nama] = {
            "model": model,
            "akurasi": akurasi,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "y_pred": y_pred,
        }

    return hasil, vectorizer


# ─────────────────────────────────────────────
# 3. VISUALISASI
# ─────────────────────────────────────────────

def plot_perbandingan_model(hasil: dict):
    metrik = ["akurasi", "f1_score", "precision", "recall"]
    nama_model = list(hasil.keys())

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(nama_model))
    width = 0.2

    for i, m in enumerate(metrik):
        nilai = [hasil[model][m] for model in nama_model]
        ax.bar(x + i * width, nilai, width, label=m.capitalize())

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(nama_model)
    ax.set_ylabel("Skor")
    ax.set_ylim(0, 1.05)
    ax.set_title("Perbandingan Performa Model Klasik — Klasifikasi Sentimen MBG", fontweight="bold")
    ax.legend(loc="lower right")
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "08_perbandingan_model.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[💾] Gambar disimpan: {path}")


def plot_confusion_matrix_semua(hasil: dict, y_test, label_order=("Negatif", "Netral", "Positif")):
    fig, axes = plt.subplots(1, len(hasil), figsize=(6 * len(hasil), 5))
    if len(hasil) == 1:
        axes = [axes]

    for ax, (nama, info) in zip(axes, hasil.items()):
        cm = confusion_matrix(y_test, info["y_pred"], labels=label_order)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=label_order, yticklabels=label_order)
        ax.set_title(f"{nama}\n(akurasi={info['akurasi']:.3f})", fontweight="bold")
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "09_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[💾] Gambar disimpan: {path}")


def tampilkan_fitur_penting(vectorizer, model, nama_model: str, label_order, top_n: int = 15):
    if not hasattr(model, "coef_"):
        print(f"[ℹ️] {nama_model} tidak memiliki koefisien linear, fitur penting dilewati.")
        return None

    fitur_names = np.array(vectorizer.get_feature_names_out())
    hasil_fitur = {}

    for i, label in enumerate(model.classes_):
        koef = model.coef_[i] if model.coef_.shape[0] > 1 else model.coef_[0]
        top_idx = np.argsort(koef)[-top_n:][::-1]
        hasil_fitur[label] = list(zip(fitur_names[top_idx], koef[top_idx]))

    return hasil_fitur


def plot_fitur_penting(hasil_fitur: dict, nama_model: str):
    if not hasil_fitur:
        return

    n_label = len(hasil_fitur)
    fig, axes = plt.subplots(1, n_label, figsize=(6 * n_label, 6))
    if n_label == 1:
        axes = [axes]

    WARNA = {"Positif": "#4CAF50", "Netral": "#2196F3", "Negatif": "#F44336"}

    for ax, (label, fitur_list) in zip(axes, hasil_fitur.items()):
        kata = [f for f, _ in fitur_list][::-1]
        skor = [s for _, s in fitur_list][::-1]
        ax.barh(kata, skor, color=WARNA.get(label, "grey"))
        ax.set_title(f"Kata Paling Berpengaruh — {label}", fontweight="bold")
        ax.set_xlabel("Koefisien (kontribusi ke kelas)")

    fig.suptitle(f"Fitur Penting — {nama_model}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "10_fitur_penting.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[💾] Gambar disimpan: {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Modeling Machine Learning — Klasifikasi Sentimen MBG")
    print("=" * 60)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y, df = siapkan_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[📊] Data latih: {len(X_train)} | Data uji: {len(X_test)}")

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
    hasil, vectorizer = latih_dan_evaluasi(X_train, X_test, y_train, y_test, vectorizer)

    print("\n[📈] Membuat visualisasi perbandingan model...")
    plot_perbandingan_model(hasil)
    plot_confusion_matrix_semua(hasil, y_test)

    print("\n[🔍] Menganalisis fitur/kata paling berpengaruh...")
    model_terbaik_nama = max(hasil, key=lambda k: hasil[k]["f1_score"])
    print(f"[🏆] Model dengan F1-score terbaik: {model_terbaik_nama}")

    hasil_fitur = tampilkan_fitur_penting(
        vectorizer, hasil["Logistic Regression"]["model"],
        "Logistic Regression", label_order=y.unique()
    )
    if hasil_fitur:
        plot_fitur_penting(hasil_fitur, "Logistic Regression")

    joblib.dump(hasil[model_terbaik_nama]["model"], os.path.join(MODEL_DIR, "model_terbaik.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    print(f"\n[💾] Model terbaik ({model_terbaik_nama}) & vectorizer disimpan di {MODEL_DIR}/")

    df_perbandingan = pd.DataFrame({
        nama: {
            "Akurasi": info["akurasi"],
            "F1-Score": info["f1_score"],
            "Precision": info["precision"],
            "Recall": info["recall"],
        }
        for nama, info in hasil.items()
    }).T
    path_csv = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    df_perbandingan.to_csv(path_csv, encoding="utf-8-sig")
    print(f"[💾] Tabel perbandingan model disimpan: {path_csv}")
    print(f"\n{df_perbandingan.to_string()}")

    print("\n[🎉] Modeling selesai!")
    print("     Lanjutkan ke 07_evaluasi_lengkap.py untuk laporan evaluasi detail.")
