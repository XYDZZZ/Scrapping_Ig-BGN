"""
=============================================================
  Modeling Machine Learning — Klasifikasi Sentimen MBG
=============================================================

Melatih 3 model klasik untuk klasifikasi sentimen, menggunakan
label dari IndoBERT (yang sudah dikoreksi sarkasme) sebagai
ground truth:
  - Logistic Regression
  - Multinomial Naive Bayes
  - Linear SVM

Fitur: TF-IDF (unigram + bigram) dari teks yang sudah di-preprocessing
(teks_final: hasil cleaning, normalisasi, stopword removal, stemming).

CATATAN METODOLOGI:
Label yang dipakai untuk training BUKAN label manual (gold standard),
melainkan label dari IndoBERT yang sudah melalui koreksi sarkasme
berbasis kamus. Ini berarti model ML yang dilatih di sini mengukur
seberapa baik fitur TF-IDF + model klasik bisa MENIRU keputusan
IndoBERT (+koreksi), bukan mengukur akurasi absolut terhadap
kebenaran objektif. Akurasi tinggi berarti pola linguistik yang
dipelajari IndoBERT cukup konsisten untuk dipelajari ulang oleh
model yang lebih sederhana.

Fokus pada kelas Positif vs Negatif (kelas Netral dikeluarkan dari
training, mengikuti praktik umum karena kelas netral seringkali
tidak punya sinyal kata kunci yang jelas, hanya "ketidakhadiran"
sinyal positif/negatif).
"""

import pandas as pd
import numpy as np
import glob
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

OUTPUT_DIR = "data/output"
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")


def load_data() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "mbg_dengan_emosi.csv")))
    if not files:
        files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "mbg_sentimen_final_terkoreksi.csv")))
    if not files:
        files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "mbg_sentimen_*.csv")))
    if not files:
        raise FileNotFoundError("Tidak ada file hasil sentimen ditemukan.")

    path = files[-1]
    print(f"[📂] Membaca: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["teks_final"] = df["teks_final"].fillna("").astype(str)
    return df


def siapkan_data_modeling(df: pd.DataFrame):
    # Fokus Positif vs Negatif, keluarkan Netral
    df_model = df[df["teks_final"].str.len() > 5].copy()
    df_model = df_model[df_model["sentimen"].isin(["Positif", "Negatif"])].copy()

    print(f"\n[📊] Data untuk modeling: {len(df_model)} baris")
    print(df_model["sentimen"].value_counts().to_string())

    if len(df_model) < 30:
        raise ValueError(
            "Data terlalu sedikit untuk modeling yang representatif "
            f"({len(df_model)} baris, minimal disarankan 30+)."
        )

    le = LabelEncoder()
    y = le.fit_transform(df_model["sentimen"])
    X = df_model["teks_final"]

    return X, y, le, df_model


def main():
    print("=" * 60)
    print("  Modeling Machine Learning — Sentimen MBG")
    print("=" * 60)

    os.makedirs(FIGURE_DIR, exist_ok=True)

    df = load_data()
    X, y, le, df_model = siapkan_data_modeling(df)

    # ── TF-IDF ──────────────────────────────────
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_tfidf = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n[✅] TF-IDF selesai!")
    print(f"   Fitur: {X_tfidf.shape[1]}")
    print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # ── Training 3 model ──────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Linear SVM": LinearSVC(C=1.0, max_iter=2000, random_state=42, class_weight="balanced"),
    }

    hasil_model = {}
    print("\n" + "=" * 60)
    print("  EVALUASI MODEL KLASIFIKASI SENTIMEN")
    print("=" * 60)

    for nama, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        hasil_model[nama] = {"model": model, "accuracy": acc, "report": report, "cm": cm, "y_pred": y_pred}

        print(f"\n📌 {nama}")
        print(f"   Akurasi: {acc:.4f} ({acc*100:.2f}%)")
        print(f"   F1-score (macro): {report['macro avg']['f1-score']:.4f}")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

    best_model_name = max(hasil_model, key=lambda x: hasil_model[x]["accuracy"])
    print(f"\n🏆 Model Terbaik: {best_model_name} ({hasil_model[best_model_name]['accuracy']*100:.2f}%)")

    # ── Visualisasi: Confusion Matrix ────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Confusion Matrix — 3 Model Klasifikasi Sentimen", fontsize=14, fontweight="bold")
    for i, (nama, hasil) in enumerate(hasil_model.items()):
        sns.heatmap(hasil["cm"], annot=True, fmt="d", ax=axes[i], cmap="Blues",
                    xticklabels=le.classes_, yticklabels=le.classes_)
        axes[i].set_title(f"{nama}\nAkurasi: {hasil['accuracy']:.2%}", fontweight="bold")
        axes[i].set_xlabel("Prediksi")
        axes[i].set_ylabel("Aktual")
    plt.tight_layout()
    path_cm = os.path.join(FIGURE_DIR, "08_confusion_matrix.png")
    plt.savefig(path_cm, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[💾] Gambar disimpan: {path_cm}")

    # ── Visualisasi: Perbandingan akurasi & F1 ───
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(hasil_model.keys())
    accs = [hasil_model[n]["accuracy"] for n in names]
    f1s = [hasil_model[n]["report"]["macro avg"]["f1-score"] for n in names]

    x = np.arange(len(names))
    ax.bar(x - 0.2, accs, 0.35, label="Accuracy", color="#3498db")
    ax.bar(x + 0.2, f1s, 0.35, label="F1-score", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    ax.set_ylim(0, 1)
    ax.set_title("Perbandingan Performa 3 Model", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path_perf = os.path.join(FIGURE_DIR, "09_perbandingan_model.png")
    plt.savefig(path_perf, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[💾] Gambar disimpan: {path_perf}")

    # ── Fitur Penting (Logistic Regression) ──────
    lr_model = hasil_model["Logistic Regression"]["model"]
    feature_names = tfidf.get_feature_names_out()

    if hasattr(lr_model, "coef_") and len(le.classes_) == 2:
        coef = lr_model.coef_[0]
        top_pos = np.argsort(coef)[-20:][::-1]
        top_neg = np.argsort(coef)[:20]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].barh([feature_names[i] for i in top_pos], coef[top_pos], color="#2ecc71")
        axes[0].set_title(f"Top 20 Kata → {le.classes_[1]}", fontweight="bold")
        axes[0].invert_yaxis()

        axes[1].barh([feature_names[i] for i in top_neg], np.abs(coef[top_neg]), color="#e74c3c")
        axes[1].set_title(f"Top 20 Kata → {le.classes_[0]}", fontweight="bold")
        axes[1].invert_yaxis()

        plt.tight_layout()
        path_fitur = os.path.join(FIGURE_DIR, "10_fitur_penting.png")
        plt.savefig(path_fitur, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[💾] Gambar disimpan: {path_fitur}")

        print(f"\n[📋] Top 10 kata → kelas '{le.classes_[1]}':")
        for i in top_pos[:10]:
            print(f"   {feature_names[i]:20s} (koef: {coef[i]:+.3f})")
        print(f"\n[📋] Top 10 kata → kelas '{le.classes_[0]}':")
        for i in top_neg[:10]:
            print(f"   {feature_names[i]:20s} (koef: {coef[i]:+.3f})")

    print("\n[🎉] Modeling selesai!")
    print(f"\n[📋] RINGKASAN UNTUK LAPORAN:")
    for nama, hasil in hasil_model.items():
        print(f"   {nama:<22}: Akurasi {hasil['accuracy']*100:.1f}% | F1-macro {hasil['report']['macro avg']['f1-score']:.3f}")
    print(f"   🏆 Model terbaik: {best_model_name}")


if __name__ == "__main__":
    main()
