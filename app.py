# app.py
# ============================
# MOBILE BANKING SENTIMENT APP
# ============================

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from wordcloud import WordCloud
import joblib

st.set_page_config(page_title="Mobile Banking Sentiment", layout="wide")

# ============================
# CUSTOM CSS (NAVY SIDEBAR + CARD)
# ============================

st.markdown("""
<style>

/* ==== SIDEBAR CONTAINER (NAVY SOLID) ==== */
[data-testid="stSidebar"] {
    background-color: #1E3A8A !important;
    padding: 20px 0 40px 0 !important;
}

/* ==== SIDEBAR TITLE ==== */
.sidebar-title {
    font-size: 24px;
    font-weight: 800;
    color: #E5E7EB;
    padding: 0 24px;
    margin-bottom: 18px;
}

/* ==== MENU BUTTON DI SIDEBAR (GANTI RADIO) ==== */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background-color: transparent;
    border: none;
    color: #E5E7EB;

    display: flex !important;
    justify-content: flex-start !important;
    align-items: center;
    text-align: left !important;

    padding: 12px 16px;
    border-radius: 0;
    border-bottom: 1px solid rgba(255,255,255,0.12);
    font-size: 16px;
    cursor: pointer;
    box-shadow: none !important;
}

/* HOVER */
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #102459;
    color: #FFFFFF;
}

/* ACTIVE / FOCUS (TANPA OREN) */
[data-testid="stSidebar"] .stButton > button:focus,
[data-testid="stSidebar"] .stButton > button:active,
[data-testid="stSidebar"] .stButton > button:focus-visible {
    background-color: #0B1A3F !important;
    color: #FFFFFF !important;
    border-bottom: 1px solid rgba(148,163,184,0.6) !important;
    box-shadow: none !important;
    outline: none !important;
}

/* ==== MAIN ==== */
.main {background-color: #F8FAFC;}

.title {
    color: white;
    padding: 20px;
    background-color:#1E3A8A;
    text-align:center;
    border-radius:10px;
    margin-bottom: 20px;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

/* CARD Sentiment Analysis */
.sa-card {
    background-color: #FFFFFF;
    padding: 20px 24px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(15,23,42,0.12);
    margin-bottom: 8px;
}

/* PERBESAR SEDIKIT FONT PLACEHOLDER TEXTAREA */
textarea::placeholder {
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ============================
# LOAD MODEL NAIVE BAYES
# ============================

@st.cache_resource
def load_nb_model():
    vec = joblib.load("nb_vectorizer.pkl")
    model = joblib.load("nb_model.pkl")
    return vec, model

vec_nb, nb_model = load_nb_model()


def analyze_sentiment_text(text: str):
    """Prediksi sentimen kalimat (positive/negative) pakai model Naive Bayes."""
    if text is None or str(text).strip() == "":
        return 0.0, "neutral"

    t = str(text).lower()
    X_vec = vec_nb.transform([t])

    proba = nb_model.predict_proba(X_vec)[0]   # [P(negatif), P(positif)]
    pred_label = nb_model.predict(X_vec)[0]    # 'positif' / 'negatif'

    if pred_label == "positif":
        label = "positive"
    elif pred_label == "negatif":
        label = "negative"
    else:
        label = pred_label

    score = float(proba[1] - proba[0])
    return score, label


# ============================
# LOAD DATA
# ============================

@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")

    df = df.rename(columns={
        "app_name": "platform",
        "review": "review",
        "label": "sentiment",
        "at": "date",
        "score": "score",
        "stemming": "stemming"
    })

    df["platform"] = df["platform"].replace({
        "BCA Mobile": "BCA Mobile",
        "BRImo": "BRImo",
        "brimo": "BRImo",
        "BRIMO": "BRImo"
    })

    df["sentiment"] = df["sentiment"].replace({
        "positif": "positive",
        "negatif": "negative",
        "Positif": "positive",
        "Negatif": "negative",
        "POSITIVE": "positive",
        "NEGATIVE": "negative"
    })

    df = df[df["sentiment"].isin(["positive", "negative"])]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


df = load_data()

TOTAL_DOC = len(df)
POS_DOC = int((df["sentiment"] == "positive").sum())
NEG_DOC = int((df["sentiment"] == "negative").sum())

# akurasi global dari hasil uji di colab
GLOBAL_ACCURACY = 0.905  # 90,5%

# ============================
# SIDEBAR MENU
# ============================

st.sidebar.markdown("<div class='sidebar-title'>Menu</div>", unsafe_allow_html=True)

options = ["Home", "Dashboard", "Sentiment Analysis", "Data", "About"]

if "page" not in st.session_state:
    st.session_state["page"] = "Home"

for opt in options:
    clicked = st.sidebar.button(opt, key=f"nav_{opt}")
    if clicked:
        st.session_state["page"] = opt

menu = st.session_state["page"]

# ============================
# HOME PAGE
# ============================

if menu == "Home":

    st.markdown("""
    <style>
    .hero {
        width: 100%;
        background-color: #1E3A8A;
        padding: 80px 30px;
        border-radius: 18px;
        text-align: center;
        color: white;
        margin-bottom: 40px;
    }

    .hero-title {
        font-size: 48px;
        font-weight: 900;
    }

    .hero-sub {
        font-size: 18px;
        opacity: 0.95;
        margin-top: 8px;
    }

    .hero-btn {
        margin-top: 22px;
        display: inline-block;
        background-color: #FACC15;  
        color: black;
        padding: 12px 28px;
        border-radius: 8px;
        font-weight: 700;
        text-decoration: none;
    }

    .section-title { 
        text-align: center;
        font-size: 30px;
        font-weight: 900;
        color: #1E3A8A;
        margin-bottom: 8px;
    }

    .section-sub { 
        text-align: center;
        font-size: 16px;
        margin-bottom: 30px;
        color: #475569;
    }

    .card3 {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.1);
        min-height: 200px;
        text-align: center;
    }

    .card3 h3 {
        color: #1E3A8A;
        font-size: 20px;
        margin-bottom: 10px;
        font-weight: 800;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
        <div class="hero-title">Analisis Sentimen BCA Mobile & BRImo</div>
        <div class="hero-sub">Menggali persepsi pengguna berdasarkan ulasan Google PlayStore</div>
        <a class="hero-btn" href="#metodologi">Pelajari Lebih Lanjut</a>
    </div>
    """, unsafe_allow_html=True)

    total = len(df)
    bca = len(df[df.platform == "BCA Mobile"])
    bri = len(df[df.platform == "BRImo"])

    colL, colA, colB, colC, colR = st.columns([3, 2, 2, 2, 1])
    colA.metric("Total Review", total)
    colB.metric("BCA Mobile", bca)
    colC.metric("BRImo", bri)

    st.markdown("""
    <div style="margin-top: 40px;" id="metodologi"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-title">Metodologi dan Tujuan Proyek</div>
    <div class="section-sub">Analisis sentimen digunakan untuk memahami persepsi masyarakat terhadap aplikasi mobile banking.</div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="card3">
            <h3>Analisis Sentimen</h3>
            <p>Mengklasifikasikan komentar pengguna menjadi positif dan negatif.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card3">
            <h3>Pemanfaatan Data</h3>
            <p>Mengolah data ulasan untuk melihat proporsi sentimen dan kualitas layanan.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card3">
            <h3>Rekomendasi</h3>
            <p>Memberikan rekomendasi berdasarkan hasil analisis data.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-title" style="margin-top:40px;">Apa itu BCA Mobile & BRImo?</div>
    """, unsafe_allow_html=True)

    html_info = """
    <style>
    .info-box {
        background: white;
        padding: 45px;
        border-radius: 20px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.1);
        text-align: center;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .info-text {
        text-align: center;
        font-size: 14px;
        margin-bottom: 24px;
        color: #475569;
        line-height: 1.7;
    }

    .info-subtitle {
        font-size: 20px;
        color: #1E3A8A;
        font-weight: 800;
        margin-top: 10px;
        margin-bottom: 18px;
    }

    .info-list {
        font-size: 14px;
        line-height: 1.9;
        display: inline-block;
        text-align: left;
        font-weight: 400;
        color: #475569;
    }
    </style>

    <div class="info-box">

        <div class="info-text">
            BCA Mobile dan BRImo adalah aplikasi mobile banking untuk transaksi digital seperti
            pembayaran, transfer, cek saldo, dan pengelolaan keuangan.
        </div>

        <div class="info-subtitle">Bagaimana Cara Kerjanya?</div>

        <ol class="info-list">
            <li>Pengumpulan Data – Mengambil komentar dari Google Play Store.</li>
            <li>Analisis Sentimen – Mengklasifikasikan komentar menjadi positif dan negatif.</li>
            <li>Rekomendasi – Menyusun rekomendasi berdasarkan hasil analisis.</li>
        </ol>
        
    </div>
    """

    components.html(html_info, height=550)


# ============================
# DASHBOARD PAGE
# ============================

elif menu == "Dashboard":

    st.markdown("<h1 class='title'>Sentiment Dashboard</h1>", unsafe_allow_html=True)

    # ---------- FILTER ATAS ----------
    fcol1, fcol2 = st.columns(2)

    with fcol1:
        pilih_app = st.selectbox("Pilih Aplikasi", ["All", "BCA Mobile", "BRImo"])

    with fcol2:
        pilih_sentimen = st.selectbox("Pilih Sentimen", ["All", "positive", "negative"])

    # apply filter ke data
    data_dash = df.copy()
    if pilih_app != "All":
        data_dash = data_dash[data_dash.platform == pilih_app]
    if pilih_sentimen != "All":
        data_dash = data_dash[data_dash.sentiment == pilih_sentimen]

    # ===================== ROW 1 : PIE + BAR =====================

    row1_col1, row1_col2 = st.columns(2)

    # ---- PIE CHART DISTRIBUSI SENTIMEN ----
    with row1_col1:
        st.markdown("### Sentiment Distribution")

        if len(data_dash) > 0:
            pie = px.pie(
                data_dash,
                names="sentiment",
                color="sentiment",
                color_discrete_map={"positive": "#1E3A8A", "negative": "#3B82F6"}
            )
            pie.update_layout(
                legend=dict(x=0.75, y=0.5, font=dict(size=12)),
                margin=dict(l=0, r=0, t=30, b=0),
                height=250   # <<< di sini kamu atur tingginya (misal 300–400)
            )
            st.plotly_chart(pie, use_container_width=True)
        else:
            st.info("Tidak ada data untuk ditampilkan.")

    # ---- BAR CHART PER APLIKASI ----
    with row1_col2:
        st.markdown("### Bar Chart")

        if len(data_dash) > 0:
            counts_app = (
                data_dash.groupby(["platform", "sentiment"])
                .size()
                .reset_index(name="jumlah")
            )

            fig_bar = px.bar(
                counts_app,
                x="platform",
                y="jumlah",
                color="sentiment",
                barmode="group",
                color_discrete_map={"positive": "#1E3A8A", "negative": "#3B82F6"}
            )
            fig_bar.update_layout(
                margin=dict(l=0, r=0, t=30, b=0),
                height=350
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Tidak ada data untuk ditampilkan.")

        # ===================== ROW 2 : LENGTH + WORDCLOUD =====================

    row2_col1, row2_col2 = st.columns(2)

     # ===================== ROW 2 : LENGTH + WORDCLOUD =====================

    row2_col1, row2_col2 = st.columns(2)

    # ---- COMMENT LENGTH DISTRIBUTION ----
    with row2_col1:
        st.markdown("### Comment Length Distribution")

        if len(data_dash) > 0:
            data_dash["length"] = data_dash["review"].astype(str).apply(len)

            fig_len = px.histogram(
                data_dash,
                x="length",
                nbins=50,
                color="sentiment",
                color_discrete_map={"positive": "#1E3A8A", "negative": "#3B82F6"}
            )
            fig_len.update_layout(
                legend=dict(x=0.92, y=0.95, font=dict(size=10)),
                margin=dict(l=0, r=0, t=30, b=0),
                height=320                # <<< tinggi disamain
            )

            st.plotly_chart(fig_len, use_container_width=True)
        else:
            st.info("Tidak ada data untuk ditampilkan.")

    # ---- WORD CLOUD ----
    with row2_col2:
        st.markdown("### Word Cloud – Most Frequent Words")

        if len(data_dash) > 0:
            all_text = " ".join(data_dash["review"].astype(str).tolist())
            if all_text.strip() != "":
                wc = WordCloud(
                    width=800,             # agak lebar
                    height=350,            # <<< sama: 350px
                    background_color="white",
                    colormap="Blues",
                    max_words=50
                ).generate(all_text)

                # ukuran figure disesuaikan dengan tinggi 350px
                fig_wc, ax_wc = plt.subplots(figsize=(6, 3.5))
                ax_wc.imshow(wc, interpolation="bilinear")
                ax_wc.axis("off")

                st.pyplot(fig_wc, use_container_width=True)
            else:
                st.info("Teks kosong, tidak dapat membentuk word cloud.")
        else:
            st.info("Tidak ada data untuk ditampilkan.")


# ============================
# SENTIMENT ANALYSIS PAGE
# ============================

elif menu == "Sentiment Analysis":

    st.markdown("<h1 class='title'>Sentiment Analysis</h1>", unsafe_allow_html=True)

    col_pred_left, col_pred_right = st.columns([2.3, 1.7])

    # ----- KIRI: FORM INPUT -----
    with col_pred_left:
        st.markdown("#### Masukkan Kalimat Ulasan")
        user_text = st.text_area(
            "Response",
            placeholder='contoh: "pelayanannya cepat dan aplikasinya mudah digunakan"',
            height=130
        )

        if st.button("Analisis"):
            score, label_pred = analyze_sentiment_text(user_text)
            st.session_state["last_pred_label"] = label_pred
            st.session_state["last_pred_score"] = score

        pred_label = st.session_state.get("last_pred_label", None)
        pred_score = st.session_state.get("last_pred_score", None)

    # ----- KANAN: PROSES + HASIL ANALISIS -----
    with col_pred_right:
        st.markdown(f"""
        <div style="background-color:#1E3A8A;
                    padding:12px 18px;
                    border-radius:12px;
                    max-width: 500px;
                    margin: 0 0 16px auto;
                    text-align:center;
                    color:#E5E7EB;">
            <h4 style="margin-top:0;margin-bottom:10px;color:#FFFFFF;">Proses</h4>
            <p style="margin:0;font-size:14px;">
                <b>Total Label</b><br>
                positive : 1<br>
                negative : 1
            </p>
            <p style="margin-top:12px;font-size:14px;">
                <b>Document by Label</b><br>
                Total Document : {TOTAL_DOC}<br>
                positive : {POS_DOC}<br>
                negative : {NEG_DOC}
            </p>
        </div>
        """, unsafe_allow_html=True)

        if pred_label is not None:
            if pred_label == "positive":
                warna_label = "#16A34A"
                teks_label = "positive"
            elif pred_label == "negative":
                warna_label = "#DC2626"
                teks_label = "negative"
            else:
                warna_label = "#6B7280"
                teks_label = "neutral"

            st.markdown(f"""
            <div style="background-color:white;
                        padding:18px 20px;
                        border-radius:12px;
                        box-shadow:0 4px 10px rgba(15,23,42,0.12);">
                <h4 style="margin-top:0;margin-bottom:10px;color:#0F172A;">Hasil Analisis</h4>
                <div style="background-color:{warna_label};
                            color:white;
                            padding:10px 14px;
                            border-radius:8px;
                            margin-bottom:10px;
                            text-align:center;
                            font-weight:600;">
                    Hasil Sentimen : {teks_label}
                </div>
                <div style="background-color:#1E3A8A;
                            color:white;
                            padding:10px 14px;
                            border-radius:8px;
                            text-align:center;
                            font-weight:600;">
                    Akurasi Model (data uji) : {GLOBAL_ACCURACY*100:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color:white;
                        padding:18px 20px;
                        border-radius:12px;
                        box-shadow:0 4px 10px rgba(15,23,42,0.12);">
                <h4 style="margin-top:0;margin-bottom:10px;color:#0F172A;">Hasil Analisis</h4>
                <p style="font-size:14px;color:#475569;">
                    Belum ada analisis. Silakan masukkan kalimat ulasan pada kolom di sebelah kiri
                    lalu tekan tombol <b>Analisis</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)


# ============================
# DATA PAGE
# ============================

elif menu == "Data":

    st.markdown("<h1 class='title'>Sentiment Data</h1>", unsafe_allow_html=True)
    st.markdown("### View Data")

    table = df[["platform", "review", "sentiment", "stemming"]]
    table = table.rename(columns={
        "platform": "Aplikasi",
        "review": "Review",
        "sentiment": "Sentimen",
        "stemming": "Cleaned Text",
    })

    st.dataframe(table, use_container_width=True, height=600)


# ============================
# ABOUT PAGE
# ============================

elif menu == "About":

    st.markdown("<h1 class='title'>Tentang Aplikasi</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class='card' style='font-size:16px; line-height:1.7;'>
       <h3> Deskripsi Aplikasi</h3>
       <p>
       Aplikasi ini dirancang untuk menganalisis sentimen publik terhadap layanan mobile banking 
       <b>BCA Mobile</b> dan <b>BRImo</b>, berdasarkan ulasan pengguna dan ditampilkan dalam bentuk visualisasi grafik yang mudah dipahami.
       </p>

       <h3> Fitur Utama</h3>
       <ul>
           <li>Analisis komentar positif & negatif</li>
           <li>Perbandingan sentimen antar aplikasi</li>
           <li>Visualisasi grafik interaktif</li>
           <li>Tampilan data komentar yang telah diproses</li>
           <li>Fitur analisis kalimat untuk uji sentimen secara langsung</li>
       </ul>

       <h3>‍Pengembang</h3>
       <p>
       Aplikasi ini dikembangkan oleh Grace Trifosa Sagala (NIM 825220125) sebagai bagian dari Tugas Akhir di Universitas Tarumanagara. 
       Tujuannya adalah memberikan pemahaman yang lebih mendalam mengenai opini pengguna terhadap layanan mobile banking 
       melalui analisis sentimen yang terstruktur dan mudah dipahami.
       </p>

       <h3> Kontak</h3>
       Jika Anda memiliki pertanyaan, saran, atau masukan, silakan hubungi melalui email: 
       grace.825220125@stu.untar.ac.id
    </div>
    """, unsafe_allow_html=True)
