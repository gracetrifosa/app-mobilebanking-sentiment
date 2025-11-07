import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pathlib import Path

# ------------------ SETUP PAGE ------------------
st.set_page_config(page_title="Analisis Sentimen BCA Mobile dan BRImo", layout="wide")
st.title("📊 Analisis Sentimen BCA Mobile dan BRImo")

# ------------------ CUSTOM THEME COLOR ------------------
st.markdown("""
<style>
:root { --primary-color: #A05A5A; }

/* chip multiselect */
[data-baseweb="tag"] {
  background-color:#A05A5A !important;
  border-color:#A05A5A !important;
  color:#fff !important;
}

/* tombol */
.stButton > button {
  background-color:#A05A5A !important;
  border-color:#A05A5A !important;
  color:#fff !important;
}

/* checkbox & radio */
input[type="checkbox"], input[type="radio"] {
  accent-color:#A05A5A !important;
}

/* slider */
[data-testid="stSlider"] [role="slider"] {
  background-color:#A05A5A !important; 
  border:1px solid #A05A5A !important;
}
[data-testid="stSlider"] .st-bq { background:#A05A5A !important; }
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_and_prepare():
    p = Path("data/dataset_sentimen.csv")
    if not p.exists():
        st.error("❌ File tidak ditemukan: data/dataset_sentimen.csv")
        st.stop()
    df = pd.read_csv(p)

    # mapping kolom sesuai dataset
    df = df.rename(columns={
        "app_name": "aplikasi",
        "label": "sentimen",
        "stemming": "clean_text",
        "review": "review"
    })

    # normalisasi nilai sentimen
    df["sentimen"] = df["sentimen"].astype(str).str.lower().str.strip()
    df["sentimen"] = df["sentimen"].replace({
        "pos": "positif", "positive": "positif",
        "neg": "negatif", "negative": "negatif",
        "neutral": "netral"
    })

    # fallback kalau label kosong tapi ada skor_sentimen
    if ("positif" not in df["sentimen"].values) and ("skor_sentimen" in df.columns):
        df["sentimen"] = df["skor_sentimen"].map({1: "positif", -1: "negatif"})

    # hanya ambil dua kelas aja
    df = df[df["sentimen"].isin(["positif", "negatif"])]

    # pastikan ada clean_text
    if "clean_text" not in df or df["clean_text"].isnull().all():
        df["clean_text"] = df["review"]

    return df

df = load_and_prepare()
st.success(f"✅ Data termuat: {len(df):,} baris (dua kelas: positif & negatif).")

# ------------------ SIDEBAR FILTER ------------------
with st.sidebar:
    st.header("🔍 Filter")
    apps_all = sorted(df["aplikasi"].unique().tolist())
    default_apps = [a for a in ["BCA Mobile", "BRImo"] if a in apps_all] or apps_all[:2]
    selected_apps = st.multiselect("Pilih Aplikasi", apps_all, default=default_apps)
    selected_sents = st.multiselect(
        "Pilih Sentimen",
        ["positif", "negatif"],
        default=["positif", "negatif"]
    )

fdf = df[df["aplikasi"].isin(selected_apps) & df["sentimen"].isin(selected_sents)]

# ------------------ DISTRIBUSI SENTIMEN ------------------
st.subheader("📈 Distribusi Sentimen per Aplikasi")

COLS_PER_ROW = 2
for i in range(0, len(selected_apps), COLS_PER_ROW):
    row_apps = selected_apps[i:i + COLS_PER_ROW]
    cols = st.columns(len(row_apps))
    for col, app in zip(cols, row_apps):
        app_df = fdf[fdf["aplikasi"] == app]
        counts = (
            app_df["sentimen"]
            .value_counts()
            .reindex(["positif", "negatif"])
            .fillna(0)
        )

        with col:
            fig, ax = plt.subplots(figsize=(5.5, 4), dpi=120)

            bars = ax.bar(
                counts.index,
                counts.values,
                color="#A05A5A",
                width=0.55,
                edgecolor="black",
                linewidth=0.6,
            )

            # batas atas +10%
            max_val = counts.values.max() if len(counts) > 0 else 0
            ax.set_ylim(0, max_val * 1.1)

            # label angka di atas batang
            for b in bars:
                h = b.get_height()
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    h + (max_val * 0.02),
                    f"{int(h)}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="black",
                )

            # garis tipis di atas angka
            ax.axhline(y=max_val * 1.05, color="#A05A5A", linestyle="--", linewidth=0.8, alpha=0.7)

            # style chart
            ax.set_title(
                f"Distribusi Sentimen - {app}",
                fontsize=12,
                fontweight="bold",
                pad=12,
            )
            ax.set_ylabel("Jumlah Ulasan", fontsize=10)
            ax.set_xlabel("Sentimen", fontsize=10)
            ax.tick_params(axis="x", labelsize=10)
            ax.tick_params(axis="y", labelsize=9)
            ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

            st.pyplot(fig, clear_figure=True)

# ------------------ WORD CLOUD ------------------
st.subheader("☁️ Word Cloud per Aplikasi & Sentimen")
for app in selected_apps:
    st.markdown(f"### 📱 {app}")
    app_df = fdf[fdf["aplikasi"] == app]
    c1, c2 = st.columns(2)

    with c1:
        st.write("**Word Cloud – positif**")
        text_pos = " ".join(app_df[app_df["sentimen"] == "positif"]["clean_text"].astype(str))
        if text_pos.strip():
            wc = WordCloud(width=900, height=400, background_color="white").generate(text_pos)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("Tidak ada data positif.")

    with c2:
        st.write("**Word Cloud – negatif**")
        text_neg = " ".join(app_df[app_df["sentimen"] == "negatif"]["clean_text"].astype(str))
        if text_neg.strip():
            wc = WordCloud(width=900, height=400, background_color="white").generate(text_neg)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("Tidak ada data negatif.")
