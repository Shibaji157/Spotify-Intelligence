from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "spotify_songs.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "spotify_cleaned.csv"


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

def load_data(path=RAW_DATA_PATH):
    """
    Load the raw Spotify dataset.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Place spotify_songs.csv inside data/raw/"
        )

    df = pd.read_csv(path)

    print("=" * 60)
    print("SPOTIFY INTELLIGENCE — DATA LOADER")
    print("=" * 60)
    print(f"Rows loaded    : {df.shape[0]:,}")
    print(f"Columns loaded : {df.shape[1]}")

    return df


# ============================================================
# DATA QUALITY REPORT
# ============================================================

def data_quality_report(df):
    """
    Generate a basic data-quality summary.
    """

    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)

    print(f"Dataset shape       : {df.shape}")
    print(f"Duplicate rows      : {df.duplicated().sum():,}")
    print(f"Unique tracks       : {df['track_id'].nunique():,}")
    print(f"Unique artists      : {df['track_artist'].nunique():,}")
    print(f"Playlist genres     : {df['playlist_genre'].nunique():,}")
    print(f"Playlist subgenres  : {df['playlist_subgenre'].nunique():,}")
    print(f"Unique playlists    : {df['playlist_name'].nunique():,}")

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    print("\nMissing values:")

    if len(missing) == 0:
        print("No missing values detected.")
    else:
        print(missing)

    return missing


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df):
    """
    Clean and prepare Spotify data for downstream analysis.
    """

    cleaned = df.copy()

    # --------------------------------------------------------
    # 1. Remove exact duplicate rows
    # --------------------------------------------------------

    cleaned = cleaned.drop_duplicates()


    # --------------------------------------------------------
    # 2. Remove records without critical track information
    # --------------------------------------------------------

    cleaned = cleaned.dropna(
        subset=[
            "track_id",
            "track_name",
            "track_artist"
        ]
    )


    # --------------------------------------------------------
    # 3. Handle optional album information
    # --------------------------------------------------------

    cleaned["track_album_name"] = (
        cleaned["track_album_name"]
        .fillna("Unknown Album")
    )


    # --------------------------------------------------------
    # 4. Convert release date
    # --------------------------------------------------------

    cleaned["track_album_release_date"] = pd.to_datetime(
        cleaned["track_album_release_date"],
        errors="coerce"
    )


    # --------------------------------------------------------
    # 5. Feature engineering
    # --------------------------------------------------------

    cleaned["release_year"] = (
        cleaned["track_album_release_date"].dt.year
    )

    cleaned["duration_min"] = (
        cleaned["duration_ms"] / 60000
    )


    # --------------------------------------------------------
    # 6. Ensure numerical audio features
    # --------------------------------------------------------

    for feature in AUDIO_FEATURES:
        cleaned[feature] = pd.to_numeric(
            cleaned[feature],
            errors="coerce"
        )


    # --------------------------------------------------------
    # 7. Remove records unusable by clustering
    # --------------------------------------------------------

    cleaned = cleaned.dropna(
        subset=AUDIO_FEATURES
    )


    # --------------------------------------------------------
    # 8. Remove impossible numerical values
    # --------------------------------------------------------

    bounded_features = [
        "danceability",
        "energy",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
    ]

    for feature in bounded_features:
        cleaned = cleaned[
            cleaned[feature].between(0, 1)
        ]

    cleaned = cleaned[
        cleaned["tempo"] > 0
    ]

    cleaned = cleaned[
        cleaned["duration_ms"] > 0
    ]


    # --------------------------------------------------------
    # 9. Clean text
    # --------------------------------------------------------

    text_columns = [
        "track_name",
        "track_artist",
        "track_album_name",
        "playlist_name",
        "playlist_genre",
        "playlist_subgenre",
    ]

    for column in text_columns:
        cleaned[column] = (
            cleaned[column]
            .astype(str)
            .str.strip()
        )


    # --------------------------------------------------------
    # 10. Reset index
    # --------------------------------------------------------

    cleaned = cleaned.reset_index(drop=True)

    return cleaned


# ============================================================
# SAVE DATA
# ============================================================

def save_processed_data(df, path=PROCESSED_DATA_PATH):
    """
    Save cleaned data.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        path,
        index=False
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print(f"Clean rows : {len(df):,}")
    print(f"Saved to   : {path}")


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def run_preprocessing():
    """
    Execute the complete preprocessing pipeline.
    """

    df = load_data()

    data_quality_report(df)

    cleaned = clean_data(df)

    save_processed_data(cleaned)

    print("\nFinal dataset summary:")
    print(cleaned[AUDIO_FEATURES].describe().round(3))

    return cleaned


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_preprocessing()