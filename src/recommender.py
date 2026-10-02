from pathlib import Path
import re
import unicodedata

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "processed" / "spotify_clustered.csv"
)

MODEL_DIR = BASE_DIR / "models"

SCALER_PATH = MODEL_DIR / "scaler.joblib"


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
# LOAD RESOURCES
# ============================================================

def load_resources():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "spotify_clustered.csv not found. "
            "Run clustering.py first."
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            "scaler.joblib not found. "
            "Run clustering.py first."
        )

    df = pd.read_csv(DATA_PATH)
    scaler = joblib.load(SCALER_PATH)

    return df, scaler


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(value):
    """
    Normalize text so duplicate song/artist combinations can be detected.
    """

    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    value = unicodedata.normalize(
        "NFKD",
        value
    )

    value = "".join(
        character
        for character in value
        if not unicodedata.combining(character)
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value


# ============================================================
# UNIQUE TRACK CATALOG
# ============================================================

def build_track_catalog(df):
    """
    Build a song-level catalog.

    Spotify playlist datasets may contain:
    - repeated track IDs,
    - the same song in multiple playlists,
    - multiple IDs/records representing the same song-artist identity.

    Recommendations therefore deduplicate using normalized
    track name + artist.
    """

    catalog = df.copy()

    catalog["normalized_track_name"] = (
        catalog["track_name"]
        .apply(normalize_text)
    )

    catalog["normalized_artist"] = (
        catalog["track_artist"]
        .apply(normalize_text)
    )

    catalog["song_identity"] = (
        catalog["normalized_track_name"]
        + "|||"
        + catalog["normalized_artist"]
    )

    # Prefer the most popular representation
    catalog = catalog.sort_values(
        "track_popularity",
        ascending=False
    )

    # First remove duplicate Spotify IDs
    catalog = catalog.drop_duplicates(
        subset=["track_id"],
        keep="first"
    )

    # Then remove duplicate song + artist identities
    catalog = catalog.drop_duplicates(
        subset=["song_identity"],
        keep="first"
    )

    catalog = catalog.reset_index(drop=True)

    return catalog


# ============================================================
# SEARCH LABEL
# ============================================================

def create_track_labels(catalog):

    catalog = catalog.copy()

    catalog["search_label"] = (
        catalog["track_name"].astype(str)
        + " — "
        + catalog["track_artist"].astype(str)
    )

    return catalog


# ============================================================
# PREPARE FEATURE FRAME
# ============================================================

def feature_frame(row):
    """
    Preserve feature names expected by StandardScaler.
    """

    return pd.DataFrame(
        [[float(row[feature]) for feature in AUDIO_FEATURES]],
        columns=AUDIO_FEATURES
    )


# ============================================================
# FEATURE-LEVEL EXPLANATION
# ============================================================

def feature_similarity_percentages(
    selected_row,
    recommended_row,
    scaler
):

    selected_features = feature_frame(
        selected_row
    )

    recommended_features = feature_frame(
        recommended_row
    )

    selected_scaled = scaler.transform(
        selected_features
    )[0]

    recommended_scaled = scaler.transform(
        recommended_features
    )[0]

    distance = np.abs(
        selected_scaled - recommended_scaled
    )

    similarity = np.exp(-distance) * 100

    return {
        feature: round(float(score), 1)
        for feature, score in zip(
            AUDIO_FEATURES,
            similarity
        )
    }


# ============================================================
# EXPLAIN RECOMMENDATION
# ============================================================

def explain_recommendation(
    selected_row,
    recommended_row,
    scaler
):

    feature_scores = (
        feature_similarity_percentages(
            selected_row,
            recommended_row,
            scaler
        )
    )

    sorted_features = sorted(
        feature_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return {
        "shared_cluster": (
            int(selected_row["cluster"])
            == int(recommended_row["cluster"])
        ),

        "top_matching_features":
            sorted_features[:4],

        "all_feature_scores":
            feature_scores,
    }


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def recommend_tracks(
    track_id,
    n_recommendations=10,
    same_cluster=False,
    same_genre=False,
):

    df, scaler = load_resources()

    catalog = build_track_catalog(df)
    catalog = create_track_labels(catalog)

    selected_matches = catalog[
        catalog["track_id"] == track_id
    ]

    if selected_matches.empty:
        raise ValueError(
            f"Track ID '{track_id}' was not found "
            "in the unique track catalog."
        )

    selected_row = selected_matches.iloc[0]

    selected_identity = (
        selected_row["song_identity"]
    )

    # --------------------------------------------------------
    # Exclude selected song identity completely
    # --------------------------------------------------------

    candidates = catalog[
        catalog["song_identity"]
        != selected_identity
    ].copy()

    # --------------------------------------------------------
    # Optional filters
    # --------------------------------------------------------

    if same_cluster:

        candidates = candidates[
            candidates["cluster"]
            == selected_row["cluster"]
        ]

    if same_genre:

        candidates = candidates[
            candidates["playlist_genre"]
            == selected_row["playlist_genre"]
        ]

    if candidates.empty:
        raise ValueError(
            "No recommendation candidates remain "
            "after applying filters."
        )

    # --------------------------------------------------------
    # Prepare feature matrices WITH feature names
    # --------------------------------------------------------

    selected_features = pd.DataFrame(
        [
            selected_row[
                AUDIO_FEATURES
            ].astype(float).values
        ],
        columns=AUDIO_FEATURES
    )

    candidate_features = (
        candidates[AUDIO_FEATURES]
        .astype(float)
    )

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    selected_scaled = scaler.transform(
        selected_features
    )

    candidates_scaled = scaler.transform(
        candidate_features
    )

    # --------------------------------------------------------
    # Cosine similarity
    # --------------------------------------------------------

    similarities = cosine_similarity(
        selected_scaled,
        candidates_scaled
    )[0]

    candidates["similarity_score"] = (
        similarities * 100
    )

    candidates = candidates.sort_values(
        "similarity_score",
        ascending=False
    )

    # Safety deduplication
    candidates = candidates.drop_duplicates(
        subset=["song_identity"],
        keep="first"
    )

    candidates = candidates.head(
        n_recommendations
    )

    # --------------------------------------------------------
    # Build response
    # --------------------------------------------------------

    recommendations = []

    for _, row in candidates.iterrows():

        explanation = explain_recommendation(
            selected_row,
            row,
            scaler
        )

        recommendations.append({

            "track_id":
                row["track_id"],

            "track_name":
                row["track_name"],

            "artist":
                row["track_artist"],

            "album":
                row["track_album_name"],

            "genre":
                row["playlist_genre"],

            "subgenre":
                row["playlist_subgenre"],

            "popularity":
                int(row["track_popularity"]),

            "cluster":
                int(row["cluster"]),

            "similarity_score":
                round(
                    float(
                        row["similarity_score"]
                    ),
                    2
                ),

            "shared_cluster":
                explanation[
                    "shared_cluster"
                ],

            "top_matching_features":
                explanation[
                    "top_matching_features"
                ],

            "all_feature_scores":
                explanation[
                    "all_feature_scores"
                ],
        })

    return {

        "selected_track": {

            "track_id":
                selected_row["track_id"],

            "track_name":
                selected_row["track_name"],

            "artist":
                selected_row["track_artist"],

            "album":
                selected_row["track_album_name"],

            "genre":
                selected_row["playlist_genre"],

            "subgenre":
                selected_row[
                    "playlist_subgenre"
                ],

            "popularity":
                int(
                    selected_row[
                        "track_popularity"
                    ]
                ),

            "cluster":
                int(
                    selected_row["cluster"]
                ),
        },

        "recommendations":
            recommendations,
    }


# ============================================================
# DEMO
# ============================================================

def demo():

    df, _ = load_resources()

    catalog = build_track_catalog(df)

    sample = (
        catalog.sort_values(
            "track_popularity",
            ascending=False
        )
        .iloc[0]
    )

    print("=" * 70)
    print(
        "SPOTIFY INTELLIGENCE — "
        "RECOMMENDATION ENGINE"
    )
    print("=" * 70)

    print(
        f"\nSelected: "
        f"{sample['track_name']} — "
        f"{sample['track_artist']}"
    )

    result = recommend_tracks(
        track_id=sample["track_id"],
        n_recommendations=5
    )

    print("\nTop Recommendations")
    print("-" * 70)

    for number, recommendation in enumerate(
        result["recommendations"],
        start=1
    ):

        print(
            f"\n{number}. "
            f"{recommendation['track_name']} — "
            f"{recommendation['artist']}"
        )

        print(
            f"   Similarity : "
            f"{recommendation['similarity_score']}%"
        )

        print(
            f"   Genre      : "
            f"{recommendation['genre']}"
        )

        print(
            f"   Cluster    : "
            f"{recommendation['cluster']}"
        )

        print(
            f"   Same audio cluster: "
            f"{recommendation['shared_cluster']}"
        )

        matches = ", ".join(
            f"{feature} ({score}%)"
            for feature, score
            in recommendation[
                "top_matching_features"
            ]
        )

        print(
            f"   Why similar: "
            f"{matches}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    demo()