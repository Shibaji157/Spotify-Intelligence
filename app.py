from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.visualization import (
    genre_distribution_chart,
    correlation_heatmap,
    cluster_scatter,
    genre_cluster_chart,
    playlist_cluster_chart,
    feature_by_genre,
    cluster_radar,
    popularity_distribution,
    top_artists_chart,
    cluster_size_chart,
)

from src.explainability import generate_all_personas

from src.recommender import (
    build_track_catalog,
    create_track_labels,
    recommend_tracks,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Spotify Intelligence",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "spotify_clustered.csv"
)

METADATA_PATH = (
    BASE_DIR
    / "models"
    / "model_metadata.json"
)

EVALUATION_PATH = (
    BASE_DIR
    / "models"
    / "cluster_evaluation.csv"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1500px;
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.10);
        padding: 18px;
        border-radius: 15px;
    }

    .hero {
        padding: 34px 38px;
        border-radius: 22px;
        background: linear-gradient(
            135deg,
            rgba(29,185,84,0.22),
            rgba(10,10,15,0.88)
        );
        border: 1px solid rgba(255,255,255,0.10);
        margin-bottom: 28px;
    }

    .hero h1 {
        margin: 0;
        font-size: 46px;
        font-weight: 800;
    }

    .hero p {
        margin-top: 12px;
        font-size: 18px;
        opacity: 0.82;
        max-width: 900px;
    }

    .persona-card {
        padding: 22px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.035);
        margin-bottom: 15px;
    }

    .insight-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(29,185,84,0.25);
        background: rgba(29,185,84,0.06);
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .small-note {
        opacity: 0.75;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():

        st.error(
            "spotify_clustered.csv was not found. "
            "Run preprocessing.py and clustering.py first."
        )

        st.stop()

    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metadata():

    if not METADATA_PATH.exists():
        return {}

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


@st.cache_data
def load_evaluation():

    if not EVALUATION_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        EVALUATION_PATH
    )


@st.cache_data
def get_catalog(data):

    catalog = build_track_catalog(data)

    catalog = create_track_labels(
        catalog
    )

    return catalog.sort_values(
        "search_label"
    )


# ============================================================
# LOAD PROJECT
# ============================================================

df = load_data()

metadata = load_metadata()

evaluation = load_evaluation()

personas = generate_all_personas()


# ============================================================
# MODEL METRICS
# ============================================================

best_k = metadata.get(
    "best_k",
    int(df["cluster"].nunique())
)

silhouette = metadata.get(
    "silhouette_score",
    0.1583
)

davies_bouldin = metadata.get(
    "davies_bouldin_score",
    1.5932
)

calinski_harabasz = metadata.get(
    "calinski_harabasz_score",
    4762.90
)

pca_variance = metadata.get(
    "pca_explained_variance",
    0.4067
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🎧 Spotify Intelligence"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "EDA & Correlations",
        "Genre Intelligence",
        "AI Segmentation",
        "Recommendation Engine",
        "Model Transparency",
    ],
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
**Spotify Intelligence**

Explainable Music Segmentation  
& Recommendation Engine

**Machine Learning:** K-Means + PCA  
**Recommendation:** Cosine Similarity  
**Explainability:** Feature-level similarity  
**Dataset:** 32K+ playlist-song records
"""
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Independent machine-learning project "
    "for music intelligence analysis."
)


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    st.markdown(
        """
        <div class="hero">

        <h1>🎧 Spotify Intelligence</h1>

        <p>
        An explainable machine-learning platform for
        discovering music audio personas, analyzing
        genre and playlist patterns, and generating
        content-based song recommendations.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    unique_tracks = (
        df["track_id"].nunique()
    )

    unique_artists = (
        df["track_artist"].nunique()
    )

    genres = (
        df["playlist_genre"].nunique()
    )

    clusters = (
        df["cluster"].nunique()
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Dataset Records",
        f"{len(df):,}"
    )

    c2.metric(
        "Unique Tracks",
        f"{unique_tracks:,}"
    )

    c3.metric(
        "Artists",
        f"{unique_artists:,}"
    )

    c4.metric(
        "Genres",
        genres
    )

    c5.metric(
        "AI Segments",
        clusters
    )

    st.markdown(
        "### Business & Dataset Intelligence"
    )

    left, right = st.columns(2)

    with left:

        st.plotly_chart(
            genre_distribution_chart(df),
            width="stretch",
        )

    with right:

        st.plotly_chart(
            popularity_distribution(df),
            width="stretch",
        )

    left, right = st.columns(2)

    with left:

        st.plotly_chart(
            top_artists_chart(df),
            width="stretch",
        )

    with right:

        st.plotly_chart(
            cluster_size_chart(
                df,
                personas
            ),
            width="stretch",
        )

    st.markdown(
        """
        <div class="insight-card">

        <b>How does Spotify Intelligence work?</b>
        <br><br>

        The platform analyzes measurable audio
        characteristics instead of assuming that playlist
        genre labels perfectly describe a song.

        Machine learning discovers independent audio
        segments, compares those segments with genre and
        playlist metadata, and uses audio-feature similarity
        to generate explainable recommendations.

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# EDA & CORRELATIONS
# ============================================================

elif page == "EDA & Correlations":

    st.title(
        "Exploratory Data Analysis"
    )

    st.write(
        "Investigate relationships between Spotify audio "
        "characteristics and compare feature distributions "
        "across playlist-defined genres."
    )

    st.markdown(
        "### Audio Feature Correlation"
    )

    st.plotly_chart(
        correlation_heatmap(df),
        width="stretch",
    )

    st.info(
        "Correlation describes statistical association "
        "between audio features. It does not establish "
        "causation."
    )

    st.markdown(
        "### Feature Distribution Explorer"
    )

    feature = st.selectbox(
        "Choose an audio feature",
        [
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
        ],
    )

    st.plotly_chart(
        feature_by_genre(
            df,
            feature
        ),
        width="stretch",
    )

    st.markdown(
        "### Feature Statistics by Genre"
    )

    stats = (
        df.groupby(
            "playlist_genre"
        )[feature]
        .agg(
            [
                "mean",
                "median",
                "std",
                "min",
                "max",
            ]
        )
        .round(3)
        .reset_index()
    )

    st.dataframe(
        stats,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# GENRE INTELLIGENCE
# ============================================================

elif page == "Genre Intelligence":

    st.title(
        "Genre & Playlist Intelligence"
    )

    st.write(
        "Compare playlist-defined genres and playlist "
        "names with the audio segments independently "
        "discovered by the machine-learning model."
    )

    st.markdown(
        "### Genre × Audio Segment Analysis"
    )

    st.plotly_chart(
        genre_cluster_chart(df),
        width="stretch",
    )

    st.markdown(
        "### Playlist-Level Segmentation"
    )

    st.write(
        "Analyze how learned audio segments are "
        "distributed across the most represented "
        "playlist names."
    )

    top_n = st.slider(
        "Number of playlists to analyze",
        min_value=5,
        max_value=25,
        value=15,
        step=5,
    )

    st.plotly_chart(
        playlist_cluster_chart(
            df,
            top_n=top_n
        ),
        width="stretch",
    )

    st.markdown(
        "### Genre Deep Dive"
    )

    genre = st.selectbox(
        "Select genre",
        sorted(
            df[
                "playlist_genre"
            ]
            .dropna()
            .unique()
        ),
    )

    genre_df = df[
        df["playlist_genre"] == genre
    ]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Records",
        f"{len(genre_df):,}"
    )

    c2.metric(
        "Unique Tracks",
        f"{genre_df['track_id'].nunique():,}"
    )

    c3.metric(
        "Artists",
        f"{genre_df['track_artist'].nunique():,}"
    )

    c4.metric(
        "Average Popularity",
        f"{genre_df['track_popularity'].mean():.1f}"
    )

    st.markdown(
        f"### Popular Tracks in {genre.upper()}"
    )

    top_tracks = (
        genre_df[
            [
                "track_name",
                "track_artist",
                "playlist_subgenre",
                "track_popularity",
                "cluster",
            ]
        ]
        .drop_duplicates(
            subset=[
                "track_name",
                "track_artist",
            ]
        )
        .sort_values(
            "track_popularity",
            ascending=False,
        )
        .head(30)
    )

    st.dataframe(
        top_tracks,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# AI SEGMENTATION
# ============================================================

elif page == "AI Segmentation":

    st.title(
        "AI Audio Segmentation"
    )

    st.write(
        "K-Means discovers naturally occurring audio "
        "patterns using nine standardized Spotify "
        "audio characteristics."
    )

    st.markdown(
        "### PCA Cluster Explorer"
    )

    st.plotly_chart(
        cluster_scatter(df),
        width="stretch",
    )

    st.caption(
        f"The first two principal components explain "
        f"approximately {pca_variance * 100:.2f}% "
        f"of the standardized feature variance. "
        f"This chart is a 2D visualization of the "
        f"higher-dimensional clustering."
    )

    st.markdown(
        "### Audio Persona Explorer"
    )

    cluster_id = st.selectbox(
        "Select an audio segment",
        sorted(
            df["cluster"].unique()
        ),
    )

    cluster_id = int(
        cluster_id
    )

    persona = personas[
        cluster_id
    ]

    cluster_df = df[
        df["cluster"] == cluster_id
    ]

    st.markdown(
        f"""
        <div class="persona-card">

        <h3>
        Cluster {cluster_id} —
        {persona["persona"]}
        </h3>

        <p>
        {persona["description"]}
        </p>

        <b>Dominant characteristics:</b>
        {", ".join(persona["strongest_features"])}

        <br><br>

        <b>Records in segment:</b>
        {len(cluster_df):,}

        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:

        st.plotly_chart(
            cluster_radar(
                df,
                cluster_id
            ),
            width="stretch",
        )

    with right:

        genre_counts = (
            cluster_df[
                "playlist_genre"
            ]
            .value_counts()
            .reset_index()
        )

        genre_counts.columns = [
            "Genre",
            "Songs",
        ]

        fig = px.pie(
            genre_counts,
            names="Genre",
            values="Songs",
            title="Playlist Genre Composition",
            hole=0.42,
        )

        fig.update_layout(
            template="plotly_dark",
            title_x=0.02,
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

    st.markdown(
        "### Representative Tracks"
    )

    representative_tracks = (
        cluster_df[
            [
                "track_name",
                "track_artist",
                "playlist_genre",
                "playlist_subgenre",
                "track_popularity",
            ]
        ]
        .drop_duplicates(
            subset=[
                "track_name",
                "track_artist",
            ]
        )
        .sort_values(
            "track_popularity",
            ascending=False,
        )
        .head(20)
    )

    st.dataframe(
        representative_tracks,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

elif page == "Recommendation Engine":

    st.title(
        "Explainable Recommendation Engine"
    )

    st.write(
        "Select a song and discover acoustically similar "
        "tracks using standardized audio characteristics "
        "and cosine similarity."
    )

    catalog = get_catalog(
        df
    )

    selected_label = st.selectbox(
        "Search or select a song",
        catalog[
            "search_label"
        ].tolist(),
    )

    selected_row = catalog[
        catalog["search_label"]
        == selected_label
    ].iloc[0]

    selected_cluster = int(
        selected_row["cluster"]
    )

    selected_persona = personas[
        selected_cluster
    ]

    st.markdown(
        f"""
        <div class="persona-card">

        <b>Selected Track</b>
        <br>

        {selected_row["track_name"]}
        — {selected_row["track_artist"]}

        <br><br>

        <b>Audio Persona:</b>
        {selected_persona["persona"]}

        <br>

        <b>Playlist Genre:</b>
        {selected_row["playlist_genre"]}

        <br>

        <b>Popularity:</b>
        {int(selected_row["track_popularity"])}

        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        same_cluster = st.checkbox(
            "Same audio segment only"
        )

    with col2:

        same_genre = st.checkbox(
            "Same playlist genre only"
        )

    with col3:

        number = st.slider(
            "Number of recommendations",
            min_value=3,
            max_value=10,
            value=5,
        )

    generate = st.button(
        "Generate Recommendations",
        type="primary",
        width="stretch",
    )

    if generate:

        try:

            with st.spinner(
                "Analyzing multidimensional "
                "audio similarity..."
            ):

                result = recommend_tracks(
                    track_id=(
                        selected_row[
                            "track_id"
                        ]
                    ),
                    n_recommendations=number,
                    same_cluster=same_cluster,
                    same_genre=same_genre,
                )

            st.success(
                "Recommendations generated successfully."
            )

            st.markdown(
                "### Recommended Tracks"
            )

            download_rows = []

            for index, recommendation in enumerate(
                result["recommendations"],
                start=1,
            ):

                recommendation_persona = (
                    personas[
                        recommendation[
                            "cluster"
                        ]
                    ]["persona"]
                )

                title = (
                    f"{index}. "
                    f"{recommendation['track_name']} "
                    f"— {recommendation['artist']} "
                    f"• "
                    f"{recommendation['similarity_score']}%"
                )

                with st.expander(
                    title,
                    expanded=(index == 1),
                ):

                    a, b, c, d = (
                        st.columns(4)
                    )

                    a.metric(
                        "Similarity",
                        f"{recommendation['similarity_score']}%",
                    )

                    b.metric(
                        "Popularity",
                        recommendation[
                            "popularity"
                        ],
                    )

                    c.metric(
                        "Cluster",
                        recommendation[
                            "cluster"
                        ],
                    )

                    d.metric(
                        "Same Segment",
                        (
                            "Yes"
                            if recommendation[
                                "shared_cluster"
                            ]
                            else "No"
                        ),
                    )

                    st.write(
                        f"**Genre:** "
                        f"{recommendation['genre']}"
                    )

                    st.write(
                        f"**Subgenre:** "
                        f"{recommendation['subgenre']}"
                    )

                    st.write(
                        f"**Audio Persona:** "
                        f"{recommendation_persona}"
                    )

                    st.markdown(
                        "**Why was this track recommended?**"
                    )

                    for feature, score in (
                        recommendation[
                            "top_matching_features"
                        ]
                    ):

                        progress_value = min(
                            max(
                                int(score),
                                0
                            ),
                            100,
                        )

                        st.progress(
                            progress_value,
                            text=(
                                f"{feature.replace('_', ' ').title()}: "
                                f"{score}% feature similarity"
                            ),
                        )

                    st.caption(
                        "Feature-level values describe "
                        "audio similarity and should not "
                        "be interpreted as causal drivers "
                        "of listener preference."
                    )

                download_rows.append(
                    {
                        "Rank":
                            index,

                        "Track":
                            recommendation[
                                "track_name"
                            ],

                        "Artist":
                            recommendation[
                                "artist"
                            ],

                        "Genre":
                            recommendation[
                                "genre"
                            ],

                        "Subgenre":
                            recommendation[
                                "subgenre"
                            ],

                        "Similarity (%)":
                            recommendation[
                                "similarity_score"
                            ],

                        "Popularity":
                            recommendation[
                                "popularity"
                            ],

                        "Audio Segment":
                            recommendation[
                                "cluster"
                            ],

                        "Audio Persona":
                            recommendation_persona,
                    }
                )

            download_df = pd.DataFrame(
                download_rows
            )

            csv = (
                download_df
                .to_csv(
                    index=False
                )
                .encode("utf-8")
            )

            st.download_button(
                label=(
                    "Download Recommendations as CSV"
                ),
                data=csv,
                file_name=(
                    "spotify_recommendations.csv"
                ),
                mime="text/csv",
                width="stretch",
            )

        except Exception as error:

            st.error(
                f"Recommendation failed: {error}"
            )


# ============================================================
# MODEL TRANSPARENCY
# ============================================================

elif page == "Model Transparency":

    st.title(
        "Model Transparency & Validation"
    )

    st.info(
        "This application uses unsupervised learning. "
        "The learned clusters represent audio-feature "
        "patterns and are not definitive genre labels."
    )

    st.markdown(
        "### Machine-Learning Architecture"
    )

    st.code(
        """
Raw Spotify Playlist Dataset
            ↓
Data Quality Validation
            ↓
Cleaning & Preprocessing
            ↓
9 Numerical Audio Features
            ↓
StandardScaler
            ↓
K-Means Optimization (K = 2 ... 10)
            ↓
Multi-Metric Model Selection
            ↓
Final K-Means Model
            ↓
PCA Visualization
            ↓
Explainable Audio Personas
            ↓
Track-Level Deduplication
            ↓
Cosine Similarity
            ↓
Feature-Level Explanations
            ↓
Song Recommendations
        """
    )

    st.markdown(
        "### Final Model Validation"
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Selected K",
        best_k,
    )

    m2.metric(
        "Silhouette",
        f"{silhouette:.4f}",
    )

    m3.metric(
        "Davies-Bouldin",
        f"{davies_bouldin:.4f}",
    )

    m4.metric(
        "PCA Variance",
        f"{pca_variance * 100:.2f}%",
    )

    st.caption(
        f"Calinski-Harabasz Score: "
        f"{calinski_harabasz:.2f}"
    )

    if not evaluation.empty:

        st.markdown(
            "### Cluster Optimization Analysis"
        )

        left, right = st.columns(2)

        with left:

            silhouette_fig = px.line(
                evaluation,
                x="k",
                y="silhouette",
                markers=True,
                title="Silhouette Score by K",
            )

            silhouette_fig.update_layout(
                template="plotly_dark",
                title_x=0.02,
            )

            st.plotly_chart(
                silhouette_fig,
                width="stretch",
            )

        with right:

            db_fig = px.line(
                evaluation,
                x="k",
                y="davies_bouldin",
                markers=True,
                title="Davies-Bouldin Index by K",
            )

            db_fig.update_layout(
                template="plotly_dark",
                title_x=0.02,
            )

            st.plotly_chart(
                db_fig,
                width="stretch",
            )

        inertia_fig = px.line(
            evaluation,
            x="k",
            y="inertia",
            markers=True,
            title="Elbow Analysis — K-Means Inertia",
        )

        inertia_fig.update_layout(
            template="plotly_dark",
            title_x=0.02,
        )

        st.plotly_chart(
            inertia_fig,
            width="stretch",
        )

    st.markdown(
        "### Discovered Audio Personas"
    )

    for cluster, information in (
        personas.items()
    ):

        st.markdown(
            f"""
            <div class="persona-card">

            <b>
            Cluster {cluster} —
            {information["persona"]}
            </b>

            <br><br>

            {information["description"]}

            <br><br>

            <span class="small-note">

            Strongest characteristics:
            {", ".join(
                information[
                    "strongest_features"
                ]
            )}

            </span>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "### Model Interpretation"
    )

    st.write(
        """
The relatively modest silhouette score indicates that the
discovered music segments overlap rather than forming
perfectly separated groups. This is reasonable because
musical properties such as energy, danceability, valence,
tempo and acousticness exist on continuous spectra.

The final number of clusters is selected using multiple
validation measures rather than assuming that the number
of clusters must equal the number of playlist genres.

Playlist genre labels are used to interpret the discovered
clusters and are not supervised targets during K-Means
training.
        """
    )

    st.markdown(
        "### Recommendation Methodology"
    )

    st.write(
        """
The recommendation engine compares tracks in standardized
audio-feature space using cosine similarity.

Duplicate song-artist identities are removed before
candidate ranking. This prevents a selected track from
being recommended back to the user simply because the same
song appears in multiple playlists.

Feature-level similarity scores provide an interpretable
description of why two tracks are acoustically similar.
They do not represent causal explanations of listener
preference.
        """
    )

    st.markdown(
        "### Limitations"
    )

    st.write(
        """
• Playlist genres are metadata and should not be treated as
perfect ground-truth song genres.

• The recommendation system is content-based and does not
use individual listening history.

• PCA is used for visualization. The original standardized
feature space is used for clustering and recommendation.

• Audio similarity does not guarantee that every listener
will perceive two songs as equally similar.

• The dataset is historical and is not a live music catalog.
        """
    )