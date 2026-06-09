import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="Spotify Analytics", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("data/dataset.csv", index_col=0)
    df = df.dropna(subset=["artists", "album_name", "track_name"])
    df = df.drop_duplicates(subset=["track_id"])
    return df

df = load_data()

audio_features = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

st.title("Spotify Tracks — Análisis Exploratorio")

st.sidebar.header("Filtros")
all_genres = sorted(df["track_genre"].unique())
selected_genres = st.sidebar.multiselect("Géneros", all_genres, default=all_genres[:6])
min_pop, max_pop = st.sidebar.slider("Rango de popularidad", 0, 100, (0, 100))

if not selected_genres:
    st.warning("Selecciona al menos un género.")
    st.stop()

df_f = df[df["track_genre"].isin(selected_genres)]
df_f = df_f[(df_f["popularity"] >= min_pop) & (df_f["popularity"] <= max_pop)]

st.markdown(f"**{len(df_f):,} canciones** con los filtros actuales.")

tab1, tab2, tab3, tab4 = st.tabs(["EDA", "PCA", "Espectral", "Distancia de edición"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribución de popularidad")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(df_f["popularity"], bins=40, edgecolor="none")
        ax.set_xlabel("Popularidad")
        ax.set_ylabel("Canciones")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Popularidad promedio por género")
        pop_genre = df_f.groupby("track_genre")["popularity"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(pop_genre.index, pop_genre.values)
        ax.set_xlabel("Popularidad promedio")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Correlación entre variables de audio")
    corr = df_f[audio_features + ["popularity"]].corr()
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab2:
    st.subheader("PCA sobre features de audio")

    scaler = StandardScaler()
    X = scaler.fit_transform(df_f[audio_features].values)

    pca = PCA(n_components=min(9, len(audio_features)))
    X_pca = pca.fit_transform(X)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Varianza explicada por componente**")
        exp_var = pca.explained_variance_ratio_
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(range(1, len(exp_var)+1), exp_var)
        ax.plot(range(1, len(exp_var)+1), np.cumsum(exp_var), marker="o", color="red", label="Acumulado")
        ax.set_xlabel("Componente")
        ax.set_ylabel("Varianza explicada")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Variables importantes por componente (umbral 0.3)**")
        loadings = pd.DataFrame(
            pca.components_.T,
            index=audio_features,
            columns=[f"PC{i+1}" for i in range(len(exp_var))]
        )
        threshold = 0.3
        for col in loadings.columns:
            important = loadings[col][loadings[col].abs() >= threshold].index.tolist()
            if important:
                st.markdown(f"**{col}**: {', '.join(important)}")

    st.markdown("**PC1 vs PC2 por género**")
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_genres)))
    fig, ax = plt.subplots(figsize=(9, 5))
    for g, c in zip(selected_genres, colors):
        m = df_f["track_genre"].values == g
        ax.scatter(X_pca[m, 0], X_pca[m, 1], label=g, alpha=0.35, s=8, color=c)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=4, fontsize=8, bbox_to_anchor=(1, 1))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab3:
    st.subheader("Análisis espectral — popularidad promedio por género")

    serie = df.groupby("track_genre")["popularity"].mean().sort_values().reset_index()
    y = serie["popularity"].values
    T = len(y)
    t = np.arange(T)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(t, y)
        ax.set_ylabel("Popularidad promedio")
        ax.set_title("Serie de popularidad por género")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        Y = np.fft.fft(y)
        freqs = np.fft.fftfreq(T)
        amplitude = np.abs(Y) / T
        half = T // 2
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(freqs[:half], amplitude[:half])
        ax.set_xlabel("Frecuencia")
        ax.set_ylabel("Amplitud")
        ax.set_title("Espectro de amplitud")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    n_comp = st.slider("Componentes para reconstruir", 1, 10, 5)
    idx = np.argsort(amplitude[:half])[::-1][1:n_comp+1]
    trend = np.polyval(np.polyfit(t, y, 1), t)
    y_rec = trend.copy()
    for i in idx:
        y_rec += amplitude[i] * np.cos(2 * np.pi * freqs[i] * t)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, y, label="Original", alpha=0.7)
    ax.plot(t, y_rec, label="Reconstruida", linestyle="--")
    ax.plot(t, trend, label="Tendencia", linestyle=":")
    ax.set_ylabel("Popularidad promedio")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab4:
    st.subheader("Distancia de edición entre géneros")

    def edit_distance(s1, s2):
        m, n = len(s1), len(s2)
        L = np.zeros((m+1, n+1), dtype=int)
        for i in range(m+1):
            L[i, 0] = i
        for j in range(n+1):
            L[0, j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                f = 0 if s1[i-1] == s2[j-1] else 1
                L[i, j] = min(L[i-1, j]+1, L[i, j-1]+1, L[i-1, j-1]+f)
        return L, L[m, n]

    col1, col2 = st.columns(2)
    g1 = col1.selectbox("Género 1", all_genres, index=0)
    g2 = col2.selectbox("Género 2", all_genres, index=1)

    L, dist = edit_distance(g1, g2)
    st.markdown(f"**Distancia de edición entre '{g1}' y '{g2}': {dist}**")

    row_idx = ["-"] + [f"{c}{i}" if list(g1).count(c) > 1 else c for i, c in enumerate(g1)]
    col_idx = ["-"] + [f"{c}{i}" if list(g2).count(c) > 1 else c for i, c in enumerate(g2)]
    df_matrix = pd.DataFrame(L, index=row_idx, columns=col_idx)
    st.dataframe(df_matrix.astype(str))

    st.markdown("---")
    st.subheader("Matriz de distancias entre géneros seleccionados")
    genres_to_compare = st.multiselect("Géneros a comparar", all_genres, default=all_genres[:8])
    if len(genres_to_compare) >= 2:
        dist_matrix = pd.DataFrame(index=genres_to_compare, columns=genres_to_compare)
        for ga in genres_to_compare:
            for gb in genres_to_compare:
                _, d = edit_distance(ga, gb)
                dist_matrix.loc[ga, gb] = d
        st.dataframe(dist_matrix.astype(int))