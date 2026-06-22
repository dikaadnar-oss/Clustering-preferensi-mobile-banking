import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(page_title="K-Means Mobile Banking", layout="wide")

st.title("📊 Analisis Preferensi Pengguna Mobile Banking")
st.caption("K-Means Clustering dengan PCA — Dashboard Interaktif")

# =====================================================
# UPLOAD FILE
# =====================================================

uploaded_file = st.file_uploader("Upload File Excel", type=["xlsx"])

if uploaded_file is not None:

    # =================================================
    # READ & VALIDASI AWAL
    # =================================================

    df_raw = pd.read_excel(uploaded_file)

    FREK_COL = 'Seberapa sering Anda menggunakan mobile banking?  '
    TUJ_COL  = 'Apa tujuan utama Anda menggunakan mobile banking?  '

    # --- Cek kolom wajib ada ---
    REQUIRED_COLS = [
       'Apakah Anda menggunakan layanan mobile banking?  ',
       'Kemudahan: Aplikasi mobile banking mudah digunakan ',
       'Kemudahan: Tampilan aplikasi mobile banking mudah dipahami ',
       'Kemudahan: Menu dalam aplikasi mudah diakses ',
       'Kecepatan: Proses transaksi berjalan cepat ',
       'Kecepatan: Aplikasi jarang mengalami gangguan ',
       'Kecepatan: Proses login dan akses aplikasi cepat ',
       'Keamanan: Data pribadi saya aman saat menggunakan mobile banking ',
       'Keamanan: Sistem keamanan dapat dipercaya ',
       'Keamanan: Saya merasa nyaman bertransaksi ',
       'Fitur: Fitur mobile banking lengkap ',
       'Fitur: Layanan sesuai kebutuhan saya ',
       'Fitur: Fitur membantu aktivitas transaksi ',
        FREK_COL,
        TUJ_COL,
    ]

    missing_cols = [c for c in REQUIRED_COLS if c not in df_raw.columns]
    if missing_cols:
        st.error("❌ **Kolom berikut tidak ditemukan di file yang diupload:**")
        for mc in missing_cols:
            st.write(f"  • `{mc}`")
        st.info("💡 Pastikan file yang diupload adalah file survei Mobile Banking yang benar.")
        st.stop()

    # --- Konversi kolom Likert ke numerik (paksa, non-angka → NaN) ---
    LIKERT_COLS = REQUIRED_COLS[1:13]
    for col in LIKERT_COLS:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    # --- Mapping Frekuensi & Tujuan ---
    frekuensi_mapping = {
        'Jarang': 1, '1 kali seminggu': 2,
        '2\u20133 kali seminggu': 3, 'Setiap hari': 4
    }
    tujuan_mapping = {
        'Transfer uang': 1, 'Pembayaran (tagihan, dll)': 2,
        'Top up (e-wallet, pulsa)': 3, 'Semua benar': 4,
        'Transaksi keuangan': 1, 'Semua bisa': 4,
        'semua nya': 4, 'Transfer uang dan top up': 1,
        'Transfer,bayar tagihan,kris': 2
    }

    df_raw['Frekuensi'] = df_raw[FREK_COL].astype(str).str.strip().map(frekuensi_mapping)
    df_raw['Tujuan']    = df_raw[TUJ_COL].astype(str).str.strip().map(tujuan_mapping)

    # --- Hitung rata-rata Kemudahan (untuk slider filter) ---
    df_raw['_avg_K'] = df_raw[LIKERT_COLS[:3]].mean(axis=1)

    # --- Informasi baris bermasalah (NaN) ---
    nan_frek = df_raw['Frekuensi'].isna().sum()
    nan_tuj  = df_raw['Tujuan'].isna().sum()
    nan_lik  = df_raw[LIKERT_COLS].isna().any(axis=1).sum()
    n_dup    = df_raw.duplicated().sum()

    if nan_frek > 0 or nan_tuj > 0 or nan_lik > 0 or n_dup > 0:
        with st.expander(f"⚠️ Ditemukan data bermasalah — klik untuk detail", expanded=True):
            col_i1, col_i2, col_i3, col_i4 = st.columns(4)
            col_i1.metric("Frekuensi tidak dikenal", nan_frek)
            col_i2.metric("Tujuan tidak dikenal",    nan_tuj)
            col_i3.metric("Nilai Likert kosong/teks", nan_lik)
            col_i4.metric("Baris duplikat",           n_dup)
            st.caption("Baris bermasalah otomatis dihapus sebelum analisis.")

    # =================================================
    # SIDEBAR — FILTER & PENGATURAN
    # =================================================

    st.sidebar.header("🎛️ Pengaturan & Filter")

    st.sidebar.subheader("🔍 Filter Data")

    frek_options = sorted([f for f in df_raw[FREK_COL].dropna().unique().tolist() if str(f) != 'nan'])
    tuj_options  = sorted([t for t in df_raw[TUJ_COL].dropna().unique().tolist() if str(t) != 'nan'])

    selected_frek = st.sidebar.multiselect(
        "Frekuensi Penggunaan", options=frek_options, default=frek_options
    )
    selected_tuj = st.sidebar.multiselect(
        "Tujuan Penggunaan", options=tuj_options, default=tuj_options
    )

    min_k, max_k = st.sidebar.slider(
        "Rentang Rata-rata Skor Kemudahan (K1–K3)", 1, 5, (1, 5)
    )

    st.sidebar.subheader("⚙️ Pengaturan Cluster")
    n_cluster = st.sidebar.slider("Jumlah Cluster", 2, 10, 2)

    # Apply filter
    mask = (
        df_raw[FREK_COL].isin(selected_frek) &
        df_raw[TUJ_COL].isin(selected_tuj) &
        (df_raw['_avg_K'] >= min_k) &
        (df_raw['_avg_K'] <= max_k)
    )
    df = df_raw[mask].reset_index(drop=True)

    st.sidebar.markdown("---")
    st.sidebar.metric("📋 Total Responden (raw)", len(df_raw))
    st.sidebar.metric("✅ Setelah Filter", len(df))

    # =================================================
    # CLEANING: drop duplikat + drop NaN pada kolom analisis
    # =================================================

    CLEAN_COLS = LIKERT_COLS + ['Frekuensi', 'Tujuan']
    n_before = len(df)
    df = df[df['Apakah Anda menggunakan layanan mobile banking?  '] == 'Ya']
    df = df.drop_duplicates(subset=['Nama']).reset_index(drop=True)
    df = df[df['Nama'].notna()]
    df = df[df['Nama'].str.match(r'^[A-Za-z\s\.]+$', na=False)]
    df = df[df['Nama'].str.len() >= 3]
    df = df.reset_index(drop=True)
    n_after = len(df)

    if n_before != n_after:
        st.sidebar.warning(f"🧹 {n_before - n_after} baris dihapus saat cleaning.")
    st.sidebar.metric("🧹 Setelah Cleaning", n_after)

    if n_after < 5:
        st.error("❌ Data bersih terlalu sedikit untuk dianalisis. Coba kurangi filter.")
        st.stop()

    # =================================================
    # PREPARE data_eda & data_cluster (by column name)
    # =================================================

    RENAME_MAP = {
        'Kemudahan: Aplikasi mobile banking mudah digunakan ': 'K1',
        'Kemudahan: Tampilan aplikasi mobile banking mudah dipahami ': 'K2',
        'Kemudahan: Menu dalam aplikasi mudah diakses ': 'K3',
        'Kecepatan: Proses transaksi berjalan cepat ': 'C1',
        'Kecepatan: Aplikasi jarang mengalami gangguan ': 'C2',
        'Kecepatan: Proses login dan akses aplikasi cepat ': 'C3',
        'Keamanan: Data pribadi saya aman saat menggunakan mobile banking ': 'S1',
        'Keamanan: Sistem keamanan dapat dipercaya ': 'S2',
        'Keamanan: Saya merasa nyaman bertransaksi ': 'S3',
        'Fitur: Fitur mobile banking lengkap ': 'F1',
        'Fitur: Layanan sesuai kebutuhan saya ': 'F2',
        'Fitur: Fitur membantu aktivitas transaksi ': 'F3',
    }

    data_eda = df[LIKERT_COLS].copy().rename(columns=RENAME_MAP)
    data_eda['Frekuensi'] = df['Frekuensi'].values
    data_eda['Tujuan']    = df['Tujuan'].values

    data_cluster = df[LIKERT_COLS].copy().rename(columns=RENAME_MAP)
    data_cluster['Frekuensi'] = df['Frekuensi'].values
    data_cluster['Tujuan']    = df['Tujuan'].values

    # =================================================
    # NORMALISASI & CLUSTERING (shared)
    # =================================================

    try:
        if n_cluster > len(df):
            st.error(f"❌ Jumlah cluster ({n_cluster}) lebih besar dari jumlah data ({len(df)}). Kurangi jumlah cluster.")
            st.stop()

        scaler       = StandardScaler()
        data_scaled  = scaler.fit_transform(data_cluster)

        kmeans_final = KMeans(n_clusters=n_cluster, init='k-means++', random_state=42)
        cluster      = kmeans_final.fit_predict(data_scaled)

        df['Cluster'] = cluster

        data_cluster_labeled            = data_cluster.copy()
        data_cluster_labeled['Cluster'] = cluster

        cluster_mean = data_cluster_labeled.groupby('Cluster').mean(numeric_only=True)

        pca        = PCA(n_components=2)
        pca_result = pca.fit_transform(data_scaled)

    except Exception as e:
        st.error(f"❌ Gagal menjalankan clustering: {e}")
        st.stop()

    visualisasi = pd.DataFrame({
        'PCA1': pca_result[:, 0],
        'PCA2': pca_result[:, 1],
        'Cluster': cluster
    })

    # =================================================
    # TABS
    # =================================================

    tab1, tab2, tab3 = st.tabs(["📊 EDA", "🔬 Clustering", "🔍 Drill-Down Cluster"])

    # =============================================
    # TAB 1 — EDA
    # =============================================

    with tab1:
        st.header("Exploratory Data Analysis (EDA)")
        st.caption(f"Menampilkan **{len(df)}** responden setelah filter")

        with st.expander("📄 Lihat Dataset Awal", expanded=False):
            st.dataframe(df_raw.drop(columns=['_avg_K']))

        st.subheader("Statistik Deskriptif")
        st.dataframe(data_eda.describe())

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Rata-rata Jawaban Responden")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            data_eda.mean().plot(kind='bar', ax=ax1, color='steelblue')
            ax1.set_title('Rata-rata Jawaban Responden')
            ax1.set_xlabel('Variabel')
            ax1.set_ylabel('Mean')
            ax1.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig1)

        with col2:
            st.subheader("Heatmap Korelasi Variabel")
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            sns.heatmap(data_eda.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax3)
            ax3.set_title('Heatmap Korelasi Variabel')
            plt.tight_layout()
            st.pyplot(fig3)

        st.subheader("Distribusi Data (Histogram)")
        fig2, axes2 = plt.subplots(3, 5, figsize=(16, 10))
        data_eda.hist(ax=axes2.flatten()[:len(data_eda.columns)])
        plt.tight_layout()
        st.pyplot(fig2)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Frekuensi Penggunaan")
            fig4, ax4 = plt.subplots(figsize=(7, 4))
            df[FREK_COL].value_counts().plot(kind='bar', ax=ax4, color='coral')
            ax4.set_xlabel('Kategori')
            ax4.set_ylabel('Jumlah Responden')
            ax4.tick_params(axis='x', rotation=30)
            plt.tight_layout()
            st.pyplot(fig4)

        with col4:
            st.subheader("Tujuan Penggunaan")
            fig5, ax5 = plt.subplots(figsize=(7, 4))
            df[TUJ_COL].value_counts().plot(kind='bar', ax=ax5, color='mediumseagreen')
            ax5.set_xlabel('Kategori')
            ax5.set_ylabel('Jumlah Responden')
            ax5.tick_params(axis='x', rotation=30)
            plt.tight_layout()
            st.pyplot(fig5)

    # =============================================
    # TAB 2 — CLUSTERING
    # =============================================

    with tab2:
        st.header("K-Means Clustering")

        with st.expander("📐 Data Normalisasi (preview 10 baris)", expanded=False):
            st.dataframe(pd.DataFrame(data_scaled, columns=data_cluster.columns).head(10))

        col_el, col_si = st.columns(2)

        with col_el:
            st.subheader("Metode Elbow")
            wcss = []
            for i in range(1, 11):
                km = KMeans(n_clusters=i, init='k-means++', random_state=42)
                km.fit(data_scaled)
                wcss.append(km.inertia_)
            fig6, ax6 = plt.subplots(figsize=(7, 4))
            ax6.plot(range(1, 11), wcss, marker='o', color='royalblue')
            ax6.set_title('Metode Elbow')
            ax6.set_xlabel('Jumlah Cluster')
            ax6.set_ylabel('WCSS')
            ax6.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig6)

        with col_si:
            st.subheader("Silhouette Score")
            sil_scores = []
            for i in range(2, 11):
                km_s  = KMeans(n_clusters=i, init='k-means++', random_state=42)
                lbl   = km_s.fit_predict(data_scaled)
                sil_scores.append(silhouette_score(data_scaled, lbl))
            best_k = np.argmax(sil_scores) + 2
            fig7, ax7 = plt.subplots(figsize=(7, 4))
            ax7.plot(range(2, 11), sil_scores, marker='o', color='darkorange')
            ax7.axvline(x=best_k, color='red', linestyle='--', alpha=0.6, label=f'Best k={best_k}')
            ax7.set_title('Silhouette Score')
            ax7.set_xlabel('Jumlah Cluster')
            ax7.set_ylabel('Score')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig7)
            st.info(f"💡 Silhouette terbaik: **k = {best_k}** (score = {max(sil_scores):.4f})")

        st.subheader(f"Hasil Clustering (k = {n_cluster})")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.write("**Distribusi Cluster:**")
            st.dataframe(df['Cluster'].value_counts().sort_index().rename("Jumlah Responden"))
        with col_d2:
            sc_val = silhouette_score(data_scaled, cluster)
            st.metric("Silhouette Score (k saat ini)", f"{sc_val:.4f}")

        st.subheader("Karakteristik Tiap Cluster (Rata-rata)")
        st.dataframe(cluster_mean.style.background_gradient(cmap='Blues'))

        fig8, ax8 = plt.subplots(figsize=(12, 5))
        cluster_mean.T.plot(kind='bar', ax=ax8)
        ax8.set_title('Karakteristik Tiap Cluster')
        ax8.set_xlabel('Variabel')
        ax8.set_ylabel('Rata-rata')
        ax8.legend(title='Cluster')
        ax8.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig8)

        st.subheader("Visualisasi PCA 2D")
        centroids_pca = visualisasi.groupby('Cluster')[['PCA1','PCA2']].mean()
        fig9, ax9 = plt.subplots(figsize=(10, 6))
        sc9 = ax9.scatter(
            visualisasi['PCA1'], visualisasi['PCA2'],
            c=visualisasi['Cluster'], cmap='viridis', alpha=0.6
        )
        ax9.scatter(
            centroids_pca['PCA1'], centroids_pca['PCA2'],
            marker='X', s=200, c='red', edgecolors='black', label='Centroids'
        )
        plt.colorbar(sc9, ax=ax9, label='Cluster')
        ax9.set_title('Visualisasi K-Means dengan PCA dan Centroid')
        ax9.set_xlabel('PCA 1')
        ax9.set_ylabel('PCA 2')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        st.pyplot(fig9)

        st.subheader("Download Hasil")
        st.download_button(
            label="📥 Download CSV (semua cluster)",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='hasil_clustering.csv',
            mime='text/csv'
        )

    # =============================================
    # TAB 3 — DRILL-DOWN
    # =============================================

    with tab3:
        st.header("🔍 Drill-Down per Cluster")

        DIM_COLS = {
            'Kemudahan': ['K1','K2','K3'],
            'Kecepatan': ['C1','C2','C3'],
            'Keamanan':  ['S1','S2','S3'],
            'Fitur':     ['F1','F2','F3'],
        }

        selected_cluster = st.selectbox(
            "Pilih Cluster:",
            options=sorted(df['Cluster'].unique()),
            format_func=lambda x: f"Cluster {x}"
        )

        df_cl      = data_cluster_labeled[data_cluster_labeled['Cluster'] == selected_cluster].drop(columns='Cluster')
        df_cl_info = df[df['Cluster'] == selected_cluster].reset_index(drop=True)
        n_members  = len(df_cl)
        pct        = n_members / len(df) * 100

        st.markdown(f"### Cluster {selected_cluster} — **{n_members} Responden** ({pct:.1f}% dari total)")

        # Metrics per dimensi
        cols_m = st.columns(5)
        for i, (dim, cols) in enumerate(DIM_COLS.items()):
            val     = df_cl[cols].mean().mean()
            overall = data_cluster[cols].mean().mean()
            cols_m[i].metric(f"{dim}", f"{val:.2f}", f"{val - overall:+.2f} vs all")
        cols_m[4].metric("Silhouette Score", f"{silhouette_score(data_scaled, cluster):.4f}")

        st.markdown("---")

        col_a, col_b = st.columns(2)

        # Radar Chart
        with col_a:
            st.subheader("Radar Chart — Cluster vs Semua")
            categories  = list(DIM_COLS.keys())
            N           = len(categories)
            cl_vals     = [df_cl[DIM_COLS[d]].mean().mean() for d in categories]
            all_vals    = [data_cluster[DIM_COLS[d]].mean().mean() for d in categories]
            angles      = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
            cl_vals_r   = cl_vals  + [cl_vals[0]]
            all_vals_r  = all_vals + [all_vals[0]]
            angles_r    = angles   + [angles[0]]

            fig_r, ax_r = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
            ax_r.plot(angles_r, cl_vals_r, 'o-', lw=2, label=f'Cluster {selected_cluster}', color='steelblue')
            ax_r.fill(angles_r, cl_vals_r, alpha=0.2, color='steelblue')
            ax_r.plot(angles_r, all_vals_r, 'o--', lw=2, label='Semua', color='gray')
            ax_r.fill(angles_r, all_vals_r, alpha=0.1, color='gray')
            ax_r.set_xticks(angles)
            ax_r.set_xticklabels(categories, size=11)
            ax_r.set_ylim(1, 5)
            ax_r.set_title(f'Profil Cluster {selected_cluster}', pad=15)
            ax_r.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            plt.tight_layout()
            st.pyplot(fig_r)

        # Boxplot
        with col_b:
            st.subheader("Boxplot Variabel per Dimensi")
            sel_dim  = st.selectbox("Pilih Dimensi:", options=list(DIM_COLS.keys()), key="dim_box")
            dim_vars = DIM_COLS[sel_dim]

            plot_data, plot_labels = [], []
            for var in dim_vars:
                plot_data.append(data_cluster_labeled[data_cluster_labeled['Cluster'] == selected_cluster][var].values)
                plot_data.append(data_cluster_labeled[data_cluster_labeled['Cluster'] != selected_cluster][var].values)
                plot_labels.extend([f'{var}\n(Cluster {selected_cluster})', f'{var}\n(Lainnya)'])

            fig_b, ax_b = plt.subplots(figsize=(8, 5))
            bp = ax_b.boxplot(plot_data, tick_labels=plot_labels, patch_artist=True)
            colors = []
            for _ in dim_vars:
                colors.extend(['steelblue', 'lightgray'])
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            ax_b.set_title(f'{sel_dim} — Cluster {selected_cluster} vs Lainnya')
            ax_b.set_ylabel('Skor')
            ax_b.tick_params(axis='x', rotation=15)
            ax_b.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_b)

        # Frekuensi & Tujuan dalam cluster
        st.markdown("---")
        col_c, col_d = st.columns(2)

        with col_c:
            st.subheader(f"Frekuensi Penggunaan — Cluster {selected_cluster}")
            fig_fc, ax_fc = plt.subplots(figsize=(6, 4))
            df_cl_info[FREK_COL].value_counts().plot(kind='bar', ax=ax_fc, color='coral')
            ax_fc.set_xlabel('Kategori')
            ax_fc.set_ylabel('Jumlah')
            ax_fc.tick_params(axis='x', rotation=30)
            plt.tight_layout()
            st.pyplot(fig_fc)

        with col_d:
            st.subheader(f"Tujuan Penggunaan — Cluster {selected_cluster}")
            fig_tc, ax_tc = plt.subplots(figsize=(6, 4))
            df_cl_info[TUJ_COL].value_counts().plot(kind='bar', ax=ax_tc, color='mediumseagreen')
            ax_tc.set_xlabel('Kategori')
            ax_tc.set_ylabel('Jumlah')
            ax_tc.tick_params(axis='x', rotation=30)
            plt.tight_layout()
            st.pyplot(fig_tc)

        # Tabel data mentah cluster
        st.markdown("---")
        st.subheader(f"📋 Data Responden — Cluster {selected_cluster}")
        show_cols = ['Nama','Usia','Jenis Kelamin','Pekerjaan', FREK_COL, TUJ_COL, 'Cluster']
        show_cols = [c for c in show_cols if c in df_cl_info.columns]
        st.dataframe(df_cl_info[show_cols])

        st.download_button(
            label=f"📥 Download Data Cluster {selected_cluster}",
            data=df_cl_info.to_csv(index=False).encode('utf-8'),
            file_name=f'cluster_{selected_cluster}.csv',
            mime='text/csv'
        )

else:
    st.info("Silakan upload file Excel terlebih dahulu.")
