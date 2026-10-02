from pathlib import Path

import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "spotify_cleaned.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

CLUSTERED_DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "spotify_clustered.csv"
)


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
# LOAD DATA
# ============================================================

def load_processed_data():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            "Processed dataset not found.\n"
            "Run: python src/preprocessing.py"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    print("=" * 70)
    print(
        "SPOTIFY INTELLIGENCE — "
        "ML SEGMENTATION ENGINE"
    )
    print("=" * 70)

    print(
        f"Rows       : {len(df):,}"
    )

    print(
        f"Features   : {len(AUDIO_FEATURES)}"
    )

    # --------------------------------------------------------
    # Validate required features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in AUDIO_FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing required audio features: "
            + ", ".join(missing_features)
        )

    # --------------------------------------------------------
    # Ensure feature values are numeric
    # --------------------------------------------------------

    for feature in AUDIO_FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    missing_feature_rows = (
        df[AUDIO_FEATURES]
        .isna()
        .any(axis=1)
        .sum()
    )

    if missing_feature_rows > 0:

        print(
            f"Removing {missing_feature_rows:,} rows "
            f"with invalid audio features."
        )

        df = (
            df.dropna(
                subset=AUDIO_FEATURES
            )
            .reset_index(drop=True)
        )

    return df


# ============================================================
# SCALE FEATURES
# ============================================================

def scale_features(df):

    X = (
        df[AUDIO_FEATURES]
        .copy()
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    print(
        "\nFeature scaling completed."
    )

    print(
        f"Scaled matrix shape: "
        f"{X_scaled.shape}"
    )

    return (
        X_scaled,
        scaler
    )


# ============================================================
# EVALUATE DIFFERENT K VALUES
# ============================================================

def evaluate_clusters(
    X_scaled,
    k_values=range(2, 11)
):

    results = []

    print(
        "\n" + "=" * 70
    )

    print(
        "CLUSTER OPTIMIZATION"
    )

    print(
        "=" * 70
    )

    print(
        f"{'K':<5}"
        f"{'Inertia':<16}"
        f"{'Silhouette':<16}"
        f"{'Davies-Bouldin':<18}"
        f"{'Calinski-Harabasz'}"
    )

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            random_state=42
        )

        labels = model.fit_predict(
            X_scaled
        )

        inertia = (
            model.inertia_
        )

        silhouette = silhouette_score(
            X_scaled,
            labels,
            sample_size=min(
                10000,
                len(X_scaled)
            ),
            random_state=42
        )

        db_score = (
            davies_bouldin_score(
                X_scaled,
                labels
            )
        )

        ch_score = (
            calinski_harabasz_score(
                X_scaled,
                labels
            )
        )

        results.append(
            {
                "k": k,
                "inertia": inertia,
                "silhouette": silhouette,
                "davies_bouldin": db_score,
                "calinski_harabasz": ch_score,
            }
        )

        print(
            f"{k:<5}"
            f"{inertia:<16.2f}"
            f"{silhouette:<16.4f}"
            f"{db_score:<18.4f}"
            f"{ch_score:.2f}"
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# SELECT BEST K
# ============================================================

def select_best_k(results):

    """
    Select K using a transparent
    multi-metric ranking.

    Higher Silhouette = better
    Lower Davies-Bouldin = better
    Higher Calinski-Harabasz = better
    """

    ranked = results.copy()

    ranked[
        "silhouette_rank"
    ] = (
        ranked["silhouette"]
        .rank(
            ascending=False
        )
    )

    ranked[
        "db_rank"
    ] = (
        ranked["davies_bouldin"]
        .rank(
            ascending=True
        )
    )

    ranked[
        "ch_rank"
    ] = (
        ranked["calinski_harabasz"]
        .rank(
            ascending=False
        )
    )

    ranked[
        "combined_rank"
    ] = (
        ranked["silhouette_rank"]
        + ranked["db_rank"]
        + ranked["ch_rank"]
    )

    best_row = (
        ranked.sort_values(
            [
                "combined_rank",
                "silhouette"
            ],
            ascending=[
                True,
                False
            ]
        )
        .iloc[0]
    )

    best_k = int(
        best_row["k"]
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "MODEL SELECTION"
    )

    print(
        "=" * 70
    )

    print(
        "Recommended number "
        f"of clusters: K = {best_k}"
    )

    print(
        "Silhouette Score      : "
        f"{best_row['silhouette']:.4f}"
    )

    print(
        "Davies-Bouldin Index  : "
        f"{best_row['davies_bouldin']:.4f}"
    )

    print(
        "Calinski-Harabasz     : "
        f"{best_row['calinski_harabasz']:.2f}"
    )

    return (
        best_k,
        ranked
    )


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

def train_final_model(
    X_scaled,
    best_k
):

    print(
        "\nTraining final "
        "K-Means model..."
    )

    model = KMeans(
        n_clusters=best_k,
        init="k-means++",
        n_init=50,
        max_iter=500,
        random_state=42
    )

    labels = model.fit_predict(
        X_scaled
    )

    print(
        "Final model training complete."
    )

    return (
        model,
        labels
    )


# ============================================================
# PCA
# ============================================================

def apply_pca(
    X_scaled
):

    pca = PCA(
        n_components=2,
        random_state=42
    )

    components = pca.fit_transform(
        X_scaled
    )

    variance = (
        pca.explained_variance_ratio_
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "PCA DIMENSIONALITY REDUCTION"
    )

    print(
        "=" * 70
    )

    print(
        "PC1 explained variance: "
        f"{variance[0] * 100:.2f}%"
    )

    print(
        "PC2 explained variance: "
        f"{variance[1] * 100:.2f}%"
    )

    print(
        "Total explained variance: "
        f"{variance.sum() * 100:.2f}%"
    )

    return (
        components,
        pca
    )


# ============================================================
# CLUSTER PROFILE
# ============================================================

def create_cluster_profiles(
    df
):

    profiles = (
        df.groupby(
            "cluster"
        )[AUDIO_FEATURES]
        .mean()
        .round(3)
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CLUSTER AUDIO PROFILES"
    )

    print(
        "=" * 70
    )

    print(
        profiles
    )

    return profiles


# ============================================================
# GENRE DISTRIBUTION INSIDE CLUSTERS
# ============================================================

def genre_cluster_analysis(
    df
):

    genre_distribution = (
        pd.crosstab(
            df["cluster"],
            df["playlist_genre"],
            normalize="index"
        )
        * 100
    )

    genre_distribution = (
        genre_distribution
        .round(2)
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "GENRE DISTRIBUTION "
        "INSIDE EACH CLUSTER (%)"
    )

    print(
        "=" * 70
    )

    print(
        genre_distribution
    )

    return (
        genre_distribution
    )


# ============================================================
# SAVE MODEL ARTIFACTS
# ============================================================

def save_artifacts(
    df,
    scaler,
    model,
    pca,
    evaluation_results,
    ranked_results,
    profiles,
    genre_distribution,
    best_k
):

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CLUSTERED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save trained ML objects
    # --------------------------------------------------------

    joblib.dump(
        scaler,
        MODEL_DIR
        / "scaler.joblib"
    )

    joblib.dump(
        model,
        MODEL_DIR
        / "kmeans.joblib"
    )

    joblib.dump(
        pca,
        MODEL_DIR
        / "pca.joblib"
    )

    # --------------------------------------------------------
    # Save analytical outputs
    # --------------------------------------------------------

    evaluation_results.to_csv(
        MODEL_DIR
        / "cluster_evaluation.csv",
        index=False
    )

    ranked_results.to_csv(
        MODEL_DIR
        / "cluster_ranking.csv",
        index=False
    )

    profiles.to_csv(
        MODEL_DIR
        / "cluster_profiles.csv"
    )

    genre_distribution.to_csv(
        MODEL_DIR
        / "genre_cluster_distribution.csv"
    )

    # --------------------------------------------------------
    # Save final clustered dataset
    # --------------------------------------------------------

    df.to_csv(
        CLUSTERED_DATA_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Get metrics for selected model
    # --------------------------------------------------------

    best_result = (
        evaluation_results[
            evaluation_results["k"]
            == best_k
        ]
        .iloc[0]
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "project_name":
            "Spotify Intelligence",

        "model_version":
            "1.0",

        "best_k":
            int(best_k),

        "audio_features":
            AUDIO_FEATURES,

        "training_rows":
            int(len(df)),

        "algorithm":
            "KMeans",

        "scaling":
            "StandardScaler",

        "dimensionality_reduction":
            "PCA",

        "random_state":
            42,

        "silhouette_score":
            float(
                best_result[
                    "silhouette"
                ]
            ),

        "davies_bouldin_score":
            float(
                best_result[
                    "davies_bouldin"
                ]
            ),

        "calinski_harabasz_score":
            float(
                best_result[
                    "calinski_harabasz"
                ]
            ),

        "pca_pc1_explained_variance":
            float(
                pca.explained_variance_ratio_[0]
            ),

        "pca_pc2_explained_variance":
            float(
                pca.explained_variance_ratio_[1]
            ),

        "pca_explained_variance":
            float(
                pca.explained_variance_ratio_
                .sum()
            ),
    }

    # --------------------------------------------------------
    # Save metadata JSON
    # --------------------------------------------------------

    with open(
        MODEL_DIR
        / "model_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Confirmation
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "MODEL ARTIFACTS SAVED"
    )

    print(
        "=" * 70
    )

    print(
        "✓ scaler.joblib"
    )

    print(
        "✓ kmeans.joblib"
    )

    print(
        "✓ pca.joblib"
    )

    print(
        "✓ cluster_evaluation.csv"
    )

    print(
        "✓ cluster_ranking.csv"
    )

    print(
        "✓ cluster_profiles.csv"
    )

    print(
        "✓ genre_cluster_distribution.csv"
    )

    print(
        "✓ model_metadata.json"
    )

    print(
        "✓ spotify_clustered.csv"
    )


# ============================================================
# COMPLETE TRAINING PIPELINE
# ============================================================

def run_training():

    # --------------------------------------------------------
    # 1. Load processed dataset
    # --------------------------------------------------------

    df = (
        load_processed_data()
    )

    # --------------------------------------------------------
    # 2. Standardize audio features
    # --------------------------------------------------------

    X_scaled, scaler = (
        scale_features(
            df
        )
    )

    # --------------------------------------------------------
    # 3. Evaluate possible cluster counts
    # --------------------------------------------------------

    evaluation_results = (
        evaluate_clusters(
            X_scaled
        )
    )

    # --------------------------------------------------------
    # 4. Select best K
    # --------------------------------------------------------

    best_k, ranked_results = (
        select_best_k(
            evaluation_results
        )
    )

    # --------------------------------------------------------
    # 5. Train final K-Means model
    # --------------------------------------------------------

    model, labels = (
        train_final_model(
            X_scaled,
            best_k
        )
    )

    # --------------------------------------------------------
    # 6. Add cluster assignments
    # --------------------------------------------------------

    df["cluster"] = (
        labels
    )

    # --------------------------------------------------------
    # 7. PCA dimensionality reduction
    # --------------------------------------------------------

    components, pca = (
        apply_pca(
            X_scaled
        )
    )

    df["pca_1"] = (
        components[:, 0]
    )

    df["pca_2"] = (
        components[:, 1]
    )

    # --------------------------------------------------------
    # 8. Build cluster profiles
    # --------------------------------------------------------

    profiles = (
        create_cluster_profiles(
            df
        )
    )

    # --------------------------------------------------------
    # 9. Compare clusters with playlist genres
    # --------------------------------------------------------

    genre_distribution = (
        genre_cluster_analysis(
            df
        )
    )

    # --------------------------------------------------------
    # 10. Print cluster sizes
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "CLUSTER SIZES"
    )

    print(
        "=" * 70
    )

    cluster_sizes = (
        df["cluster"]
        .value_counts()
        .sort_index()
    )

    print(
        cluster_sizes
    )

    # --------------------------------------------------------
    # 11. Save all model artifacts
    # --------------------------------------------------------

    save_artifacts(
        df=df,
        scaler=scaler,
        model=model,
        pca=pca,
        evaluation_results=(
            evaluation_results
        ),
        ranked_results=(
            ranked_results
        ),
        profiles=profiles,
        genre_distribution=(
            genre_distribution
        ),
        best_k=best_k
    )

    # --------------------------------------------------------
    # 12. Finish
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "SPOTIFY SEGMENTATION "
        "PIPELINE COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_training()