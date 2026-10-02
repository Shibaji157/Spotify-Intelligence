import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# AUDIO FEATURES
# ============================================================

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


# ============================================================
# COMMON LAYOUT
# ============================================================

def apply_layout(fig, title=None):

    fig.update_layout(
        template="plotly_dark",
        title=title,
        title_x=0.02,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
        legend_title_text="",
    )

    return fig


# ============================================================
# 1. GENRE DISTRIBUTION
# ============================================================

def genre_distribution_chart(df):

    genre_counts = (
        df["playlist_genre"]
        .value_counts()
        .reset_index()
    )

    genre_counts.columns = [
        "Genre",
        "Records",
    ]

    fig = px.bar(
        genre_counts,
        x="Genre",
        y="Records",
        text="Records",
        title="Playlist Genre Distribution",
    )

    fig.update_traces(
        textposition="outside"
    )

    return apply_layout(fig)


# ============================================================
# 2. CORRELATION HEATMAP
# ============================================================

def correlation_heatmap(df):

    available_features = [
        feature
        for feature in AUDIO_FEATURES
        if feature in df.columns
    ]

    correlation = (
        df[available_features]
        .corr()
        .round(2)
    )

    fig = px.imshow(
        correlation,
        text_auto=True,
        aspect="auto",
        title="Audio Feature Correlation Matrix",
    )

    return apply_layout(fig)


# ============================================================
# 3. PCA CLUSTER SCATTER
# ============================================================

def cluster_scatter(df):

    required = [
        "pca_1",
        "pca_2",
        "cluster",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing PCA/cluster columns: "
            + ", ".join(missing)
        )

    plot_df = df.copy()

    plot_df["cluster_label"] = (
        "Cluster "
        + plot_df["cluster"]
        .astype(str)
    )

    hover_columns = []

    for column in [
        "track_name",
        "track_artist",
        "playlist_genre",
        "playlist_subgenre",
        "track_popularity",
    ]:
        if column in plot_df.columns:
            hover_columns.append(column)

    # Sampling keeps browser rendering responsive.
    if len(plot_df) > 12000:

        plot_df = plot_df.sample(
            12000,
            random_state=42,
        )

    fig = px.scatter(
        plot_df,
        x="pca_1",
        y="pca_2",
        color="cluster_label",
        hover_data=hover_columns,
        opacity=0.60,
        title="PCA Projection of Learned Audio Segments",
    )

    fig.update_traces(
        marker=dict(
            size=6
        )
    )

    fig.update_layout(
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
    )

    return apply_layout(fig)


# ============================================================
# 4. GENRE × CLUSTER
# ============================================================

def genre_cluster_chart(df):

    distribution = pd.crosstab(
        df["playlist_genre"],
        df["cluster"],
        normalize="index",
    ) * 100

    distribution = (
        distribution
        .round(2)
        .reset_index()
    )

    melted = distribution.melt(
        id_vars="playlist_genre",
        var_name="Cluster",
        value_name="Percentage",
    )

    melted["Cluster"] = (
        "Cluster "
        + melted["Cluster"]
        .astype(str)
    )

    fig = px.bar(
        melted,
        x="playlist_genre",
        y="Percentage",
        color="Cluster",
        barmode="stack",
        title="Audio Segment Distribution Within Each Genre",
    )

    fig.update_layout(
        xaxis_title="Playlist Genre",
        yaxis_title="Percentage of Records (%)",
    )

    return apply_layout(fig)


# ============================================================
# 5. PLAYLIST × CLUSTER
# ============================================================

def playlist_cluster_chart(
    df,
    top_n=15,
):

    if "playlist_name" not in df.columns:

        raise ValueError(
            "playlist_name column was not found."
        )

    top_playlists = (
        df["playlist_name"]
        .value_counts()
        .head(top_n)
        .index
    )

    subset = df[
        df["playlist_name"].isin(
            top_playlists
        )
    ].copy()

    distribution = pd.crosstab(
        subset["playlist_name"],
        subset["cluster"],
        normalize="index",
    ) * 100

    distribution = (
        distribution
        .round(2)
        .reset_index()
    )

    melted = distribution.melt(
        id_vars="playlist_name",
        var_name="Cluster",
        value_name="Percentage",
    )

    melted["Cluster"] = (
        "Cluster "
        + melted["Cluster"]
        .astype(str)
    )

    playlist_order = (
        subset["playlist_name"]
        .value_counts()
        .index
        .tolist()
    )

    fig = px.bar(
        melted,
        y="playlist_name",
        x="Percentage",
        color="Cluster",
        orientation="h",
        barmode="stack",
        category_orders={
            "playlist_name":
                playlist_order[::-1]
        },
        title=(
            f"Audio Segment Distribution — "
            f"Top {top_n} Playlists"
        ),
    )

    fig.update_layout(
        xaxis_title="Percentage of Playlist Records (%)",
        yaxis_title="Playlist Name",
        height=max(
            500,
            top_n * 35
        ),
    )

    return apply_layout(fig)


# ============================================================
# 6. FEATURE DISTRIBUTION BY GENRE
# ============================================================

def feature_by_genre(
    df,
    feature,
):

    if feature not in df.columns:

        raise ValueError(
            f"{feature} is not available "
            "in the dataset."
        )

    fig = px.box(
        df,
        x="playlist_genre",
        y=feature,
        color="playlist_genre",
        points=False,
        title=(
            f"{feature.replace('_', ' ').title()} "
            "Distribution by Genre"
        ),
    )

    fig.update_layout(
        xaxis_title="Playlist Genre",
        yaxis_title=(
            feature
            .replace("_", " ")
            .title()
        ),
        showlegend=False,
    )

    return apply_layout(fig)


# ============================================================
# 7. CLUSTER RADAR
# ============================================================

def cluster_radar(
    df,
    cluster_id,
):

    radar_features = [
        "danceability",
        "energy",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
    ]

    radar_features = [
        feature
        for feature in radar_features
        if feature in df.columns
    ]

    cluster_data = df[
        df["cluster"] == cluster_id
    ]

    if cluster_data.empty:

        raise ValueError(
            f"Cluster {cluster_id} "
            "contains no records."
        )

    cluster_means = (
        cluster_data[
            radar_features
        ]
        .mean()
    )

    global_min = (
        df[radar_features]
        .min()
    )

    global_max = (
        df[radar_features]
        .max()
    )

    denominator = (
        global_max - global_min
    ).replace(
        0,
        1
    )

    normalized = (
        cluster_means
        - global_min
    ) / denominator

    labels = [
        feature
        .replace("_", " ")
        .title()
        for feature in radar_features
    ]

    values = normalized.tolist()

    # Close radar polygon.
    labels = labels + [
        labels[0]
    ]

    values = values + [
        values[0]
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            name=f"Cluster {cluster_id}",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
            )
        ),
        showlegend=False,
        title=(
            f"Audio Profile — "
            f"Cluster {cluster_id}"
        ),
    )

    return apply_layout(fig)


# ============================================================
# 8. POPULARITY DISTRIBUTION
# ============================================================

def popularity_distribution(df):

    if "track_popularity" not in df.columns:

        raise ValueError(
            "track_popularity column "
            "was not found."
        )

    fig = px.histogram(
        df,
        x="track_popularity",
        nbins=30,
        title="Track Popularity Distribution",
    )

    fig.update_layout(
        xaxis_title="Track Popularity",
        yaxis_title="Number of Records",
    )

    return apply_layout(fig)


# ============================================================
# 9. TOP ARTISTS
# ============================================================

def top_artists_chart(
    df,
    top_n=10,
):

    artist_counts = (
        df["track_artist"]
        .dropna()
        .value_counts()
        .head(top_n)
        .sort_values()
        .reset_index()
    )

    artist_counts.columns = [
        "Artist",
        "Records",
    ]

    fig = px.bar(
        artist_counts,
        x="Records",
        y="Artist",
        orientation="h",
        text="Records",
        title=f"Top {top_n} Artists by Dataset Presence",
    )

    fig.update_layout(
        xaxis_title="Number of Records",
        yaxis_title="Artist",
    )

    return apply_layout(fig)


# ============================================================
# 10. CLUSTER SIZE
# ============================================================

def cluster_size_chart(
    df,
    personas=None,
):

    counts = (
        df["cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    counts.columns = [
        "Cluster",
        "Records",
    ]

    if personas is not None:

        counts["Segment"] = (
            counts["Cluster"]
            .apply(
                lambda cluster: (
                    f"Cluster {cluster} — "
                    f"{personas.get(
                        int(cluster),
                        {}
                    ).get(
                        'persona',
                        'Audio Segment'
                    )}"
                )
            )
        )

    else:

        counts["Segment"] = (
            "Cluster "
            + counts["Cluster"]
            .astype(str)
        )

    fig = px.bar(
        counts,
        x="Segment",
        y="Records",
        text="Records",
        title="Learned Audio Segment Sizes",
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="Audio Segment",
        yaxis_title="Number of Records",
    )

    return apply_layout(fig)