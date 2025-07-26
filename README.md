# **Clustering Spotify Songs for Playlist Generation**

## Project Overview
Building an automated playlist generator using unsupervised machine learning fro Streamify, a cutting-edge music tech startup. By clustering songs based on their musical features (e.g., energy, danceability, tempo), the aim is to improve music discovery, enhance user satisfaction, and power personalized recommendations. Using techniques such as K-Means and Hierarchical Clustering, songs were grouped into meaningful clusters like “Chill Vibes” or “High Energy Dance.” Dimensionality reduction using PCA helped visualize the clustering results, and internal validation metrics (Silhouette Score and Davies-Bouldin Index) ensured cluster quality. A prediction function was also built to assign new songs to their respective clusters, and sample playlists were generated from each group:

* Improve **user satisfaction** through better playlist curation
* Enable **music discovery** through feature-based similarity
* Support **personalized recommendations** for users

This project clusters thousands of Spotify songs into meaningful groups or "vibes" using K-Means and Hierarchical Clustering. The resulting clusters feed into a recommendation engine that suggests similar songs and automatically labels playlists.

---

## Objective

* Cluster Spotify songs into meaningful groups
* Label each cluster (e.g., **“Chill Vibes”**, **“High Energy Dance”**, **“Acoustic Mellow”**)
* Build a function that takes in a song's features and predicts its cluster
* Recommend songs from each cluster for playlist generation
* Build an interactive web application using **Streamlit**

---

## Dataset

The dataset includes thousands of Spotify tracks with audio features retrieved from Spotify’s Web API. Key features include:

* `danceability`, `energy`, `tempo`, `valence`, `acousticness`, `instrumentalness`, `speechiness`, etc.
* Target column `explicit` was label encoded for numerical processing

---

## Steps & Methodology

### 1. Data Preprocessing

* Loaded dataset and checked for missing values and duplicates
* Dropped identifier columns like `track_name`, `artist_name`, `track_id`
* Normalized numeric features using `StandardScaler`
* Encoded boolean column `explicit` using `LabelEncoder`

### 2. Dimensionality Reduction

* Applied **Principal Component Analysis (PCA)**
* Reduced features to 2D for visualization
* Verified number of components to retain 90–95% variance

### 3. K-Means Clustering

* Determined optimal clusters using:

  * Elbow Method
  * Silhouette Score
* Performed K-Means clustering
* Evaluated using:

  * Silhouette Score
  * Davies-Bouldin Index
* Labeled clusters:

  * Cluster 0 → **Chill Vibes**
  * Cluster 1 → **High Energy Dance**
  * Cluster 2 → **Acoustic Mellow**

### 4. Hierarchical Clustering

* Applied Agglomerative Clustering using Euclidean distance & Ward linkage
* Visualized dendrogram to choose optimal number of clusters
* Compared clustering quality with K-Means results

### 5. Model Validation

* Used **Silhouette Score** and **Davies-Bouldin Index** for internal validation
* Cross-checked stability of clusters using different seeds and PCA initializations

---

## Deployment

* Built an interactive **Streamlit app**
* The app allows users to input song features and returns the predicted **vibe/cluster**
* Cluster prediction is mapped using a simple dictionary:

```python
cluster_labels = {
    0: "Chill Vibes",
    1: "High Energy Dance",
    2: "Acoustic Mellow"
}
```

* Model saved using `joblib` for reuse:

```python
import joblib
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

* Streamlit app uses the trained model and scaler to classify new songs based on user input.

---

## Results & Insights

* **K-Means performed better** than Hierarchical Clustering with a higher Silhouette Score and lower Davies-Bouldin Index
* PCA visualizations showed **clear separation** of clusters
* Most popular cluster: **High Energy Dance**
* Least populated but distinct: **Acoustic Mellow**

---

## Recommendations

* Use clustering results as the foundation for **auto-generated playlists**
* Integrate the model into Streamify's recommendation pipeline for **real-time song suggestions**
* Consider dynamic reclustering over time as user behavior changes
* Expand features to include **user listening habits**, genre, and lyrics for improved granularity
* Fine-tune model performance with **t-SNE or UMAP** for even better visual separation

---

## Deliverables

* Well-commented **Jupyter Notebook** with full EDA, modeling, and evaluation
* Final **Streamlit App** for song classification and playlist labeling
* Professional **PowerPoint Report** with results, charts, and deployment guide
* This `README.md` summarizing the entire project

---

## Tech Stack

* Python
* Scikit-learn
* Pandas, NumPy
* Matplotlib, Seaborn
* PCA
* Streamlit
* joblib