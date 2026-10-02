from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

PROFILE_PATH = (
    BASE_DIR / "models" / "cluster_profiles.csv"
)

DATA_PATH = (
    BASE_DIR / "data" / "processed" / "spotify_clustered.csv"
)


FEATURE_LABELS = {
    "danceability": "Danceability",
    "energy": "Energy",
    "loudness": "Loudness",
    "speechiness": "Speechiness",
    "acousticness": "Acousticness",
    "instrumentalness": "Instrumentalness",
    "liveness": "Liveness",
    "valence": "Valence",
    "tempo": "Tempo",
}


def load_profiles():

    profiles = pd.read_csv(
        PROFILE_PATH,
        index_col="cluster"
    )

    return profiles


def normalized_profiles(profiles):
    """
    Normalize cluster centroids relative to other clusters.
    This is used for interpretation, not model training.
    """

    normalized = (
        profiles - profiles.min()
    ) / (
        profiles.max()
        - profiles.min()
    ).replace(0, 1)

    return normalized


def generate_persona(cluster_id):

    profiles = load_profiles()

    normalized = normalized_profiles(
        profiles
    )

    row = normalized.loc[cluster_id]

    top = (
        row.sort_values(
            ascending=False
        )
        .head(3)
        .index
        .tolist()
    )

    # Descriptive rules based on dominant centroid characteristics

    if (
        row["instrumentalness"] > 0.75
        and row["energy"] > 0.55
    ):
        name = "Electronic Instrumental Pulse"

    elif (
        row["speechiness"] > 0.75
    ):
        name = "Rhythmic Vocal Flow"

    elif (
        row["acousticness"] > 0.65
        and row["energy"] < 0.40
    ):
        name = "Acoustic Mellow"

    elif (
        row["valence"] > 0.70
        and row["danceability"] > 0.60
    ):
        name = "Feel-Good Dance"

    elif (
        row["liveness"] > 0.70
    ):
        name = "Live Energy"

    elif (
        row["energy"] > 0.65
    ):
        name = "High-Energy Drive"

    else:
        name = "Balanced Groove"

    strongest_features = [
        FEATURE_LABELS.get(
            feature,
            feature.title()
        )
        for feature in top
    ]

    description = (
        f"This segment is primarily characterized by "
        f"{strongest_features[0]}, "
        f"{strongest_features[1]}, and "
        f"{strongest_features[2]} relative to the "
        f"other discovered audio segments."
    )

    return {
        "cluster": int(cluster_id),
        "persona": name,
        "strongest_features": strongest_features,
        "description": description,
    }


def generate_all_personas():

    profiles = load_profiles()

    personas = {}

    for cluster_id in profiles.index:

        personas[int(cluster_id)] = (
            generate_persona(cluster_id)
        )

    return personas


def print_personas():

    personas = generate_all_personas()

    print("=" * 70)
    print("SPOTIFY INTELLIGENCE — EXPLAINABLE AUDIO PERSONAS")
    print("=" * 70)

    for cluster_id, information in personas.items():

        print(
            f"\nCluster {cluster_id}: "
            f"{information['persona']}"
        )

        print(
            "Key characteristics: "
            + ", ".join(
                information["strongest_features"]
            )
        )

        print(
            information["description"]
        )


if __name__ == "__main__":
    print_personas()