import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Spotify Analytics", layout="wide")

@st.cache_data
def load_spotify():
    df = pd.read_csv("data/dataset.csv", index_col=0)
    df = df.dropna(subset=["artists", "album_name", "track_name"])
    df = df.drop_duplicates(subset=["track_id"])
    return df

@st.cache_data
def load_billboard():
    df = pd.read_csv("data/charts.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df = df[(df["year"] >= 1990) & (df["year"] <= 2020)]
    df["period"] = df["date"].dt.to_period("M")
    return df

df = load_spotify()
bb = load_billboard()

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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["EDA", "PCA", "Espectral", "Similitud entre géneros", "Clasificación"])

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
        n = len(pop_genre)
        fig_h = max(3, n * 0.25)
        fig, ax = plt.subplots(figsize=(5, fig_h))
        ax.barh(pop_genre.index, pop_genre.values)
        ax.set_xlabel("Popularidad promedio")
        ax.tick_params(axis="y", labelsize=max(5, min(9, int(120 / n))))
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
        for col in loadings.columns:
            important = loadings[col][loadings[col].abs() >= 0.3].index.tolist()
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
    st.subheader("Análisis espectral — Billboard Hot 100 (1990–2020)")
    st.markdown("Serie mensual del promedio de semanas que las canciones permanecen en el chart. Refleja cómo cambia la longevidad de los hits a lo largo del tiempo.")

    variable_bb = st.selectbox(
        "Variable a analizar",
        ["weeks-on-board", "rank"],
        format_func=lambda x: "Semanas en el chart" if x == "weeks-on-board" else "Posición promedio"
    )

    serie = bb.groupby("period")[variable_bb].mean().reset_index()
    serie.columns = ["period", "valor"]
    y = serie["valor"].values
    T = len(y)
    t = np.arange(T)
    fechas = [str(p) for p in serie["period"]]

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(t, y)
        tick_step = max(1, T // 8)
        ax.set_xticks(t[::tick_step])
        ax.set_xticklabels(fechas[::tick_step], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(variable_bb)
        ax.set_title("Serie temporal mensual")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        Y = np.fft.fft(y)
        freqs = np.fft.fftfreq(T)
        amplitude = np.abs(Y) / T
        half = T // 2
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(freqs[1:half], amplitude[1:half])
        ax.set_xlabel("Frecuencia (ciclos/mes)")
        ax.set_ylabel("Amplitud")
        ax.set_title("Espectro de amplitud")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    top5_idx = np.argsort(amplitude[1:half])[::-1][:5] + 1
    top5_phases = np.angle(Y[top5_idx])
    st.markdown("**Componentes dominantes detectadas:**")
    rows = []
    for i, phi in zip(top5_idx, top5_phases):
        periodo = 1 / freqs[i]
        rows.append({"Frecuencia": round(freqs[i], 5),
                     "Periodo (meses)": round(periodo, 1),
                     "Amplitud": round(amplitude[i], 4)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    n_comp = st.slider("Componentes para reconstruir", 1, 10, 5)
    idx = np.argsort(amplitude[1:half])[::-1][:n_comp] + 1
    idx_phases = np.angle(Y[idx])
    trend = np.polyval(np.polyfit(t, y, 1), t)
    y_rec = trend.copy()
    for i, phi in zip(idx, idx_phases):
        y_rec += 2 * amplitude[i] * np.cos(2 * np.pi * freqs[i] * t + phi)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, y, label="Original", alpha=0.7)
    ax.plot(t, y_rec, label="Reconstruida", linestyle="--")
    ax.plot(t, trend, label="Tendencia", linestyle=":")
    tick_step = max(1, T // 8)
    ax.set_xticks(t[::tick_step])
    ax.set_xticklabels(fechas[::tick_step], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(variable_bb)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab4:
    st.subheader("Similitud entre géneros basada en features de audio")
    st.markdown("Centroide de cada género en el espacio PCA. La distancia euclidiana entre centroides indica qué tan similares son sus características de audio.")

    scaler_all = StandardScaler()
    X_all = scaler_all.fit_transform(df[audio_features].values)
    pca_all = PCA(n_components=2)
    X_pca_all = pca_all.fit_transform(X_all)

    genre_labels = df["track_genre"].values
    genres_all = sorted(df["track_genre"].unique())

    centroids = {g: X_pca_all[genre_labels == g].mean(axis=0) for g in genres_all}
    centroid_df = pd.DataFrame(centroids).T
    dist_matrix = pd.DataFrame(
        euclidean_distances(centroid_df.values),
        index=genres_all, columns=genres_all
    )

    ref_genre = st.selectbox("Género de referencia", genres_all, index=0)
    distances = dist_matrix[ref_genre].drop(ref_genre).sort_values()
    top_similar = distances.head(10)
    top_different = distances.tail(10).sort_values(ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**10 géneros más similares a {ref_genre}**")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(top_similar.index[::-1], top_similar.values[::-1])
        ax.set_xlabel("Distancia euclidiana")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown(f"**10 géneros más distintos a {ref_genre}**")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(top_different.index[::-1], top_different.values[::-1])
        ax.set_xlabel("Distancia euclidiana")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("**Mapa de centroides en espacio PC1 vs PC2**")
    highlight = st.multiselect("Resaltar géneros", genres_all, default=[ref_genre])

    fig, ax = plt.subplots(figsize=(10, 6))
    for g in genres_all:
        x, yc = centroids[g]
        if g in highlight:
            ax.scatter(x, yc, color="red", s=80, zorder=5)
            ax.annotate(g, (x, yc), fontsize=7, color="red",
                        xytext=(4, 4), textcoords="offset points")
        else:
            ax.scatter(x, yc, color="steelblue", s=20, alpha=0.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Centroides de géneros")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab5:
    st.subheader("Clasificación de género — Naive Bayes Multinomial")
    st.markdown("Predicción del género a partir de las features de audio, escaladas a [0, 1] (MultinomialNB requiere valores no negativos).")

    if len(selected_genres) < 2:
        st.warning("Selecciona al menos 2 géneros para clasificar.")
    else:
        X_clf = df_f[audio_features].to_numpy(dtype=float)
        y_clf = df_f["track_genre"].astype(str).to_numpy()

        scaler_clf = MinMaxScaler()
        X_clf_scaled = scaler_clf.fit_transform(X_clf)

        X_train, X_test, y_train, y_test = train_test_split(
            X_clf_scaled, y_clf, test_size=0.2, random_state=42, stratify=y_clf
        )

        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{acc:.3f}")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Reporte de clasificación**")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

        with col2:
            st.markdown("**Matriz de confusión**")
            labels = sorted(set(y_clf))
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel("Predicho")
            ax.set_ylabel("Real")
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("**Log-probabilidad de cada feature por género**")
        feat_prob = pd.DataFrame(model.feature_log_prob_, index=model.classes_, columns=audio_features)
        st.dataframe(feat_prob.round(3), use_container_width=True)