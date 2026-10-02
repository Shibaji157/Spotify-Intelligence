# 🎧 Spotify Intelligence
### Explainable Music Segmentation & Recommendation Engine

<p align="center">
  <b>End-to-End Machine Learning • Unsupervised Learning • Recommendation System • Explainable AI • Interactive Analytics</b>
</p>

<p align="center">
  <a href="https://spotify-intelligence.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
  </a>
  <a href="https://github.com/Shibaji157/Spotify-Intelligence">
    <img src="https://img.shields.io/badge/Source%20Code-GitHub-181717?style=for-the-badge&logo=github&logoColor=white">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=flat-square&logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat-square&logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=flat-square&logo=plotly&logoColor=white">
  <img src="https://img.shields.io/badge/Status-Live-success?style=flat-square">
</p>

---

## 🌐 Live Application

### 🚀 [Launch Spotify Intelligence](https://spotify-intelligence.streamlit.app/)

The application is publicly deployed using **Streamlit Community Cloud** and provides an interactive environment for exploring Spotify audio characteristics, learned music segments, genre relationships, and explainable song recommendations.

### 💻 [View GitHub Repository](https://github.com/Shibaji157/Spotify-Intelligence)

---

## 📌 Project Overview

**Spotify Intelligence** is an end-to-end machine-learning project designed to analyze Spotify playlist-song data and discover meaningful patterns in music using audio characteristics.

Instead of treating playlist genres as perfect ground-truth labels, the project uses **unsupervised machine learning** to discover latent audio segments from the songs themselves.

The final system combines:

- 🧹 Data Cleaning & Preprocessing
- 📊 Exploratory Data Analysis
- 🔗 Audio Feature Correlation Analysis
- 🎼 Genre & Playlist Intelligence
- 🤖 K-Means Clustering
- 📉 Principal Component Analysis (PCA)
- 🎭 Explainable Audio Personas
- 🎯 Content-Based Song Recommendation
- 🔍 Feature-Level Recommendation Explanations
- 📈 Interactive Plotly Visualizations
- 🌐 Streamlit Web Application
- ☁️ Public Cloud Deployment

The result is not simply a clustering notebook—it is a complete interactive **Music Intelligence & Recommendation Platform**.

---

# 🎯 Problem Statement

Music streaming platforms contain thousands or millions of tracks with overlapping musical characteristics.

Genre labels alone may not completely describe how songs relate to one another because songs from different genres can share similar characteristics such as:

- Danceability
- Energy
- Tempo
- Valence
- Acousticness
- Instrumentalness

The objective of this project is therefore to:

> **Discover meaningful audio-based song segments and use those learned relationships to support explainable music recommendations.**

---

# 💡 Solution

Spotify Intelligence implements a complete machine-learning pipeline:

```text
Spotify Playlist Dataset
          │
          ▼
Data Quality Validation
          │
          ▼
Cleaning & Preprocessing
          │
          ▼
Exploratory Data Analysis
          │
          ▼
Audio Feature Selection
          │
          ▼
Feature Standardization
          │
          ▼
K-Means Evaluation (K = 2–10)
          │
          ▼
Multi-Metric Cluster Selection
          │
          ▼
Final K-Means Segmentation
          │
          ├──────────────► Cluster Profiling
          │
          ├──────────────► Genre Intelligence
          │
          └──────────────► Audio Personas
          │
          ▼
PCA Dimensionality Reduction
          │
          ▼
Standardized Audio Feature Space
          │
          ▼
Cosine Similarity
          │
          ▼
Explainable Song Recommendations
          │
          ▼
Interactive Streamlit Application
```

---

# 📊 Dataset Intelligence

The processed dataset used by the deployed system contains:

| Dataset Metric | Value |
|---|---:|
| 🎵 Playlist-Song Records | **32,827** |
| 🎧 Unique Tracks | **28,351** |
| 🎤 Unique Artists | **10,691** |
| 🎼 Playlist Genres | **6** |
| 🤖 Learned Audio Segments | **6** |

### Playlist Genres

The dataset includes six major playlist-defined genre categories:

```text
EDM • Latin • Pop • R&B • Rap • Rock
```

An important distinction in this project is that **playlist genre metadata is used for analysis and interpretation rather than being treated as perfect ground-truth classification labels**.

---

# 🎚️ Audio Features

Nine Spotify-style audio characteristics are used by the machine-learning pipeline.

| Feature | Description |
|---|---|
| 💃 `danceability` | Suitability of a track for dancing |
| ⚡ `energy` | Perceived intensity and activity |
| 🔊 `loudness` | Overall loudness level |
| 🗣️ `speechiness` | Presence of spoken-word characteristics |
| 🎸 `acousticness` | Confidence that the track is acoustic |
| 🎹 `instrumentalness` | Likelihood of limited vocal content |
| 🎤 `liveness` | Presence of live-performance characteristics |
| 😊 `valence` | Musical positivity |
| 🥁 `tempo` | Estimated tempo in BPM |

Before model training, these features are standardized using **Scikit-learn's `StandardScaler`**.

This prevents features with larger numerical scales—such as tempo—from dominating the clustering and similarity calculations.

---

# 🧹 Data Preprocessing

The preprocessing pipeline performs several data-quality operations before machine learning.

### Key Operations

- Exact duplicate handling
- Critical missing-value handling
- Album metadata handling
- Release-date processing
- Release-year extraction
- Track-duration conversion
- Numeric feature validation
- Audio-feature range validation
- Invalid tempo/duration removal
- Text normalization
- Clean dataset generation

Processed data is stored separately from the original raw dataset to preserve reproducibility.

---

# 🔍 Exploratory Data Analysis

The application includes an interactive EDA environment for understanding relationships between Spotify audio characteristics.

### Analysis Includes

- Audio-feature correlation matrix
- Feature distribution analysis
- Genre-wise feature comparison
- Descriptive statistics
- Popularity analysis
- Artist-level exploration
- Genre distribution
- Cluster distribution

One visible relationship in the dataset is a relatively strong positive association between **energy and loudness**, while **energy and acousticness** show an inverse relationship.

These relationships help explain why certain tracks naturally occupy similar regions of the audio-feature space.

---

# 🧠 Machine Learning

## K-Means Clustering

The primary unsupervised-learning algorithm is:

```text
K-Means Clustering
```

Rather than manually selecting an arbitrary number of clusters, candidate solutions from:

```text
K = 2 → 10
```

were evaluated.

### Model Evaluation Metrics

The project evaluates candidate clustering solutions using multiple complementary metrics:

- **Silhouette Score**
- **Davies-Bouldin Index**
- **Calinski-Harabasz Score**
- **K-Means Inertia**

A combined ranking methodology is used rather than relying on a single metric.

---

# 🏆 Final Clustering Model

The multi-metric evaluation process selected:

```text
K = 6
```

for the final segmentation system.

| Metric | Final Result |
|---|---:|
| Number of Clusters | **6** |
| Silhouette Score | **0.1583** |
| Davies-Bouldin Index | **1.5932** |
| Calinski-Harabasz Score | **4762.90** |
| PCA 2D Explained Variance | **40.67%** |

### Important Interpretation

The moderate Silhouette Score is not interpreted as evidence that songs belong to perfectly isolated musical categories.

Music exists in a **continuous and overlapping audio-feature space**.

For example, tracks belonging to different playlist genres may still have similar:

```text
Energy + Tempo + Danceability + Valence
```

Therefore, the learned clusters should be interpreted as **audio-behavior segments**, not rigid genre labels.

---

# 🎭 Explainable Audio Personas

Raw cluster numbers such as `Cluster 0` or `Cluster 5` are difficult for users to interpret.

Spotify Intelligence therefore transforms cluster profiles into descriptive **audio personas**.

| Cluster | Audio Persona | Dominant Characteristics |
|---:|---|---|
| **0** | 🎛️ Electronic Instrumental Pulse | Instrumentalness, Energy, Loudness |
| **1** | ⚡ High-Energy Drive | Energy, Tempo, Loudness |
| **2** | 💃 Feel-Good Dance | Danceability, Valence, Loudness |
| **3** | 🎤 Live Energy | Liveness, Energy, Loudness |
| **4** | 🌿 Acoustic Mellow | Acousticness, Danceability, Instrumentalness |
| **5** | 🎙️ Rhythmic Vocal Flow | Speechiness, Danceability, Loudness |

These names are **interpretability labels derived from cluster characteristics** rather than official Spotify genres.

---

# 📉 PCA Visualization

High-dimensional clustering results are difficult to visualize directly.

Spotify Intelligence therefore uses:

### Principal Component Analysis (PCA)

The first two principal components explain approximately:

```text
40.67%
```

of the variance represented in the standardized audio-feature space.

PCA is used primarily for **visual exploration of learned segments**, while the clustering itself operates on the standardized audio features.

---

# 🎼 Genre Intelligence

A major component of the application investigates how playlist-defined genres interact with machine-learned audio segments.

The dashboard supports:

- Genre × Cluster analysis
- Playlist × Cluster analysis
- Genre feature comparison
- Genre-specific distributions
- Cluster composition analysis
- Representative-track exploration

This helps demonstrate that traditional genre labels and audio-based segmentation provide **different but complementary views of music structure**.

---

# 🎯 Recommendation Engine

Spotify Intelligence includes a complete **content-based recommendation system**.

Instead of recommending tracks solely because they belong to the same playlist genre, recommendations are generated using similarity in standardized audio-feature space.

### Recommendation Workflow

```text
Selected Song
     │
     ▼
Audio Feature Vector
     │
     ▼
StandardScaler
     │
     ▼
Candidate Track Feature Space
     │
     ▼
Cosine Similarity
     │
     ▼
Duplicate / Self-Match Removal
     │
     ▼
Similarity Ranking
     │
     ▼
Top Recommendations
     │
     ▼
Feature-Level Explanation
```

---

# 🔢 Cosine Similarity

For two standardized audio vectors \(A\) and \(B\), similarity is calculated conceptually as:

```text
                     A · B
cosine_similarity = ─────────
                    ||A||||B||
```

Songs with similar audio characteristics therefore receive higher similarity values.

The recommendation engine can additionally restrict candidates using:

- 🎭 Same learned audio segment
- 🎼 Same playlist genre

---

# 🔍 Explainable Recommendations

Recommendation systems become more useful when users can understand **why** two songs were considered similar.

Spotify Intelligence therefore provides feature-level similarity explanations.

Example:

```text
Recommended Track
├── Danceability       98.4%
├── Energy             97.2%
├── Tempo              96.8%
└── Instrumentalness   95.9%
```

The system highlights the strongest matching characteristics for each recommendation.

> **Important:** These values represent feature similarity. They should not be interpreted as causal explanations of human listening preference.

---

# 🖥️ Interactive Dashboard

The Streamlit application contains **six analytical modules**.

## 🏠 1. Executive Overview

Provides high-level intelligence including:

- Dataset KPIs
- Genre distribution
- Popularity distribution
- Artist analysis
- Learned segment sizes
- Dataset overview

---

## 🔬 2. EDA & Correlations

Provides:

- Interactive audio-feature correlation matrix
- Feature selection
- Genre-based distributions
- Descriptive statistics
- Feature-level exploration

---

## 🎼 3. Genre Intelligence

Analyzes:

- Genre × Cluster relationships
- Playlist-level segmentation
- Genre distributions
- Audio characteristics by genre
- Genre-specific intelligence

---

## 🤖 4. AI Segmentation

Provides:

- PCA cluster visualization
- Audio persona interpretation
- Cluster radar charts
- Cluster composition
- Representative songs
- Learned-segment exploration

---

## 🎯 5. Recommendation Engine

Allows users to:

- Select a song
- Generate similar tracks
- Control recommendation count
- Filter by audio segment
- Filter by playlist genre
- View similarity scores
- Understand feature-level matches
- Download recommendation results

---

## 🔎 6. Model Transparency

Provides visibility into:

- Model architecture
- Cluster-selection methodology
- Silhouette analysis
- Davies-Bouldin analysis
- K-Means inertia
- Audio personas
- Methodology
- Assumptions
- Limitations

This section helps make the machine-learning workflow more transparent instead of treating the model as a black box.

---

# 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Programming | 🐍 Python |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Scientific Computing | SciPy |
| Clustering | K-Means |
| Dimensionality Reduction | PCA |
| Recommendation | Cosine Similarity |
| Visualization | Plotly |
| Web Application | Streamlit |
| Model Persistence | Joblib |
| Version Control | Git |
| Repository | GitHub |
| Deployment | Streamlit Community Cloud |

---

# 📁 Project Architecture

```text
Spotify-Intelligence/
│
├── 📄 app.py
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .gitignore
│
├── 📂 data/
│   ├── raw/
│   │   └── Spotify_songs.csv
│   │
│   └── processed/
│       ├── spotify_cleaned.csv
│       └── spotify_clustered.csv
│
├── 📂 models/
│   ├── scaler.joblib
│   ├── kmeans.joblib
│   ├── pca.joblib
│   ├── cluster_evaluation.csv
│   ├── cluster_ranking.csv
│   ├── cluster_profiles.csv
│   ├── genre_cluster_distribution.csv
│   └── model_metadata.json
│
├── 📂 src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── recommender.py
│   ├── explainability.py
│   └── visualization.py
│
└── 📂 tests/
    └── test_pipeline.py
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Shibaji157/Spotify-Intelligence.git
cd Spotify-Intelligence
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Application

```bash
streamlit run app.py
```

The local application will normally be available at:

```text
http://localhost:8501
```

---

# 📦 Core Dependencies

```text
streamlit
pandas
numpy
scikit-learn
scipy
plotly
joblib
```

---

# 🌐 Deployment

The application is deployed publicly using **Streamlit Community Cloud**.

### Live Production Application

👉 **https://spotify-intelligence.streamlit.app/**

The deployment automatically installs the dependencies defined in `requirements.txt` and launches:

```text
app.py
```

---

# ⚠️ Limitations

Every machine-learning system has assumptions and limitations.

### Current Limitations

- Playlist genres are not perfect ground-truth song genres.
- K-Means assumes centroid-based cluster structures.
- Music characteristics naturally overlap between genres.
- PCA 2D visualization does not retain all information from the original feature space.
- The recommendation engine is content-based.
- User listening history is not currently incorporated.
- Similar audio characteristics do not necessarily imply identical subjective listener preference.
- Recommendations are limited to tracks represented in the available dataset.

These limitations are intentionally documented to keep model interpretation transparent.

---

# 🚀 Future Improvements

Spotify Intelligence can be extended into a larger production-oriented music intelligence platform.

Potential improvements include:

- 🎧 Spotify Web API integration
- 👤 Personalized user profiles
- 🤝 Collaborative filtering
- 🔀 Hybrid recommendation systems
- 🧠 Deep-learning music embeddings
- 🎼 Automated playlist generation
- ❤️ User feedback and preference learning
- 🔎 Approximate nearest-neighbor retrieval
- 📊 Recommendation evaluation metrics
- 🔄 Automated data pipelines
- 📡 Real-time catalog updates
- 🧪 A/B testing framework
- 📈 Model monitoring
- 🐳 Docker deployment
- ☁️ Cloud-native production architecture

---

# 🎓 Key Learning Outcomes

This project demonstrates practical experience in:

- End-to-end machine-learning project development
- Data preprocessing and validation
- Exploratory data analysis
- Feature scaling
- Unsupervised learning
- K-Means clustering
- Multi-metric model evaluation
- Dimensionality reduction
- Cluster interpretation
- Content-based recommendation systems
- Cosine similarity
- Explainable recommendation logic
- Interactive data visualization
- Streamlit application development
- Git/GitHub version control
- Cloud deployment
- Technical documentation

---

# 💼 Portfolio Summary

> **Spotify Intelligence** is an end-to-end explainable music segmentation and recommendation platform built using Python, Scikit-learn, Plotly, and Streamlit. The system analyzes 32K+ Spotify playlist-song records, performs multi-metric K-Means optimization across K=2–10, discovers six interpretable audio segments, visualizes high-dimensional patterns using PCA, and generates content-based song recommendations using cosine similarity with feature-level explanations. The complete system is deployed publicly through Streamlit Community Cloud.

---

# 👨‍💻 Author

## **Shibaji Biswas**

**AI/ML & Data Science Enthusiast | Machine Learning Developer**

I am interested in building practical, data-driven systems that combine **Machine Learning, Artificial Intelligence, Data Analytics, Business Intelligence, and real-world application development**.

This project reflects my focus on moving beyond model training toward complete solutions involving:

```text
Data → Analysis → Machine Learning → Explainability
     → Application → Deployment → Documentation
```

### 📬 Contact

- **Name:** Shibaji Biswas
- **Email:** shibajibiswas.cse@gmail.com
- **GitHub:** [Shibaji157](https://github.com/Shibaji157)
- **Project Repository:** [Spotify-Intelligence](https://github.com/Shibaji157/Spotify-Intelligence)
- **Live Application:** [Spotify Intelligence](https://spotify-intelligence.streamlit.app/)

---

# ⭐ Support

If you find this project useful or interesting, consider giving the repository a **⭐ Star**.

It helps support continued development and future improvements.

---

<p align="center">
  <b>Designed & Developed by Shibaji Biswas</b>
</p>

<p align="center">
  🎧 Turning Music Data into Explainable Intelligence 🤖
</p>

<p align="center">
  <a href="https://spotify-intelligence.streamlit.app/">Live Demo</a>
  •
  <a href="https://github.com/Shibaji157/Spotify-Intelligence">GitHub Repository</a>
</p>