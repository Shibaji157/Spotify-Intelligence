# 🎧 Spotify Intelligence
## Explainable Music Segmentation & Recommendation Engine

Spotify Intelligence is an end-to-end machine-learning application that analyzes Spotify playlist data, discovers latent audio segments, explains their characteristics, and generates content-based song recommendations.

The project transforms a conventional clustering task into an interactive music-intelligence platform built with Python, Scikit-learn, Plotly, and Streamlit.

---

## 🚀 Live Application

The public Streamlit deployment link will be added here after deployment.

---

## 🎯 Project Objective

The objective is to analyze Spotify songs using their measurable audio characteristics and discover meaningful groups of acoustically similar tracks.

The system combines:

- Data preprocessing
- Exploratory data analysis
- Correlation analysis
- Playlist and genre intelligence
- K-Means clustering
- Multi-metric cluster validation
- PCA visualization
- Explainable audio personas
- Content-based recommendation
- Interactive Streamlit analytics

---

## 📊 Dataset

The processed dataset contains:

| Metric | Value |
|---|---:|
| Records | 32,827 |
| Unique Tracks | 28,351 |
| Artists | 10,691 |
| Playlist Genres | 6 |
| Learned Audio Segments | 6 |

The playlist genres represented in the dataset are:

- EDM
- Latin
- Pop
- R&B
- Rap
- Rock

---

## 🎵 Audio Features

The machine-learning pipeline uses nine Spotify audio characteristics:

1. Danceability
2. Energy
3. Loudness
4. Speechiness
5. Acousticness
6. Instrumentalness
7. Liveness
8. Valence
9. Tempo

The features are standardized using `StandardScaler` before clustering and similarity analysis.

---

## 🧠 Machine-Learning Pipeline

```text
Spotify Playlist Dataset
          ↓
Data Validation
          ↓
Cleaning & Preprocessing
          ↓
Audio Feature Engineering
          ↓
StandardScaler
          ↓
K-Means Evaluation (K = 2–10)
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
Explainable Recommendations
          ↓
Interactive Streamlit Application
```

---

## 🔬 Cluster Optimization

Candidate models from K=2 through K=10 were evaluated using:

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score
- K-Means Inertia

The project uses a combined ranking methodology instead of selecting the final model using only one validation measure.

### Final Model

| Metric | Result |
|---|---:|
| Selected Clusters | 6 |
| Silhouette Score | 0.1583 |
| Davies-Bouldin Index | 1.5932 |
| Calinski-Harabasz Score | 4762.90 |
| PCA 2D Explained Variance | 40.67% |

The clusters are interpreted as overlapping audio-feature segments rather than definitive genre labels.

---

## 🎭 Explainable Audio Personas

The six learned segments are translated into interpretable audio personas:

| Cluster | Audio Persona |
|---:|---|
| 0 | Electronic Instrumental Pulse |
| 1 | High-Energy Drive |
| 2 | Feel-Good Dance |
| 3 | Live Energy |
| 4 | Acoustic Mellow |
| 5 | Rhythmic Vocal Flow |

These labels are derived from the relative audio characteristics of each learned cluster and are used for interpretation rather than as ground-truth genres.

---

## 🎯 Recommendation Engine

The recommendation system is content-based.

For a selected track, the engine:

1. Standardizes its audio characteristics.
2. Compares it with candidate tracks in the same feature space.
3. Calculates cosine similarity.
4. Removes duplicate song-artist identities.
5. Excludes the selected song itself.
6. Ranks the most acoustically similar tracks.
7. Generates feature-level similarity explanations.

Optional filters allow recommendations to be restricted to:

- The same learned audio segment
- The same playlist genre

---

## 💡 Explainability

Instead of returning only a similarity score, Spotify Intelligence explains which audio characteristics are most similar between the selected track and each recommendation.

Example:

```text
Danceability      97.3%
Tempo             96.7%
Speechiness       95.6%
Instrumentalness 100.0%
```

These values describe feature similarity and are not causal explanations of listener preference.

---

## 📈 Interactive Dashboard

The Streamlit application contains six analytical modules.

### 1. Executive Overview
High-level dataset KPIs, genre distribution, popularity distribution, artist analysis, and segment sizes.

### 2. EDA & Correlations
Audio-feature correlation matrix, feature distributions, and descriptive statistics.

### 3. Genre Intelligence
Analysis of relationships between playlist-defined genres, playlist names, and learned audio segments.

### 4. AI Segmentation
Interactive PCA visualization, cluster profiles, radar charts, genre composition, and representative tracks.

### 5. Recommendation Engine
Interactive song selection, recommendation filters, similarity ranking, explanations, and downloadable results.

### 6. Model Transparency
Model architecture, validation metrics, K optimization, audio personas, methodology, interpretation, and limitations.

---

## 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Plotly
- Streamlit
- Joblib
- Git & GitHub

---

## 📁 Project Structure

```text
Spotify_Intelligence/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── spotify_cleaned.csv
│       └── spotify_clustered.csv
│
├── models/
│   ├── scaler.joblib
│   ├── kmeans.joblib
│   ├── pca.joblib
│   ├── cluster_evaluation.csv
│   ├── cluster_ranking.csv
│   ├── cluster_profiles.csv
│   ├── genre_cluster_distribution.csv
│   └── model_metadata.json
│
└── src/
    ├── preprocessing.py
    ├── clustering.py
    ├── recommender.py
    ├── explainability.py
    └── visualization.py
```

---

## ▶️ Run Locally

Create and activate a Python virtual environment, install the dependencies, and launch Streamlit:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The application will normally become available at:

```text
http://localhost:8501
```

---

## ⚠️ Model Limitations

- Playlist genre metadata is not treated as perfect ground-truth song genre.
- K-Means assumes centroid-based cluster structure.
- Musical characteristics naturally overlap across genres.
- PCA is used for visualization and does not represent all variance in the original feature space.
- The recommender is content-based and does not use personal listening history.
- Audio similarity does not guarantee identical subjective listener preferences.

---

## 🔮 Future Improvements

Potential extensions include:

- Spotify API integration
- Hybrid collaborative + content-based recommendation
- User listening-history personalization
- Artist and track embeddings
- Approximate nearest-neighbor retrieval
- Playlist generation
- Model monitoring
- Real-time catalog updates

---

## 👨‍💻 Project

**Spotify Intelligence — Spotify Songs' Genre Segmentation**

Machine Learning • Unsupervised Learning • Recommendation Systems • Data Analytics • Explainable AI • Streamlit