from fastapi import APIRouter
from pydantic import BaseModel
import os
import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from umap import UMAP
import matplotlib.pyplot as plt
from backend.logger_config import logger

router = APIRouter()

class FullPipelineRequest(BaseModel):
    dataset_path: str
    num_clusters: int = 3
    sample_size: int = 5000
    max_iter: int = 100
    top_terms: int = 10

@router.post("/clustering")
def full_clustering_pipeline(request: FullPipelineRequest):
    logger.info(f"Clustering start on dataset: {request.dataset_path}")
    safe_path = request.dataset_path.replace("/", "__")
    base_dir = os.path.join("db", safe_path)
    image_path = os.path.join(base_dir, "cluster_plot.png")

    try:
        vectorizer = joblib.load(os.path.join(base_dir, "vectorizer.joblib"))
        tfidf_matrix = joblib.load(os.path.join(base_dir, "tfidf_matrix.joblib"))
    except Exception as e:
        return {"error": f"❌ Failed loading TF-IDF files: {e}"}

    tfidf_matrix = np.array(tfidf_matrix)
    sample_size = min(request.sample_size, tfidf_matrix.shape[0])
    sample_indices = np.random.choice(tfidf_matrix.shape[0], sample_size, replace=False)
    sampled_matrix = tfidf_matrix[sample_indices]

    umap_model = UMAP(n_components=2, n_neighbors=30, min_dist=0.0, metric='cosine')
    reduced_data = umap_model.fit_transform(sampled_matrix)

    model = MiniBatchKMeans(n_clusters=request.num_clusters, max_iter=request.max_iter)
    model.fit(reduced_data)
    labels = model.labels_

    scores = {
        "silhouette_score": round(silhouette_score(reduced_data, labels), 4),
        "davies_bouldin_score": round(davies_bouldin_score(reduced_data, labels), 4),
        "calinski_harabasz_score": round(calinski_harabasz_score(reduced_data, labels), 2)
    }

    # استخراج الكلمات المميزة
    terms = vectorizer.get_feature_names_out()
    top_terms_dict = {}
    for label in np.unique(labels):
        cluster_indices = np.where(labels == label)[0]
        cluster_sum = np.sum(sampled_matrix[cluster_indices], axis=0)
        cluster_sum = cluster_sum if hasattr(cluster_sum, 'A1') else cluster_sum
        ranked = np.argsort(cluster_sum)[::-1][:request.top_terms]
        top_terms_dict[str(label)] = [terms[i] for i in ranked]

    # رسم التجميعات
    try:
        plt.figure(figsize=(10, 8))
        for cluster_id in np.unique(labels):
            plt.scatter(
                reduced_data[labels == cluster_id, 0],
                reduced_data[labels == cluster_id, 1],
                label=f"Cluster {cluster_id}",
                alpha=0.7,
                s=50,
                edgecolors='k'
            )
        plt.title("Clusters Visualization")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(image_path)
        plt.close()
    except Exception as e:
        return {"error": f"Failed to save plot image: {e}"}

    # حفظ النموذج والـ UMAP (اختياري)
    joblib.dump(model, os.path.join(base_dir, "cluster_model.joblib"))
    joblib.dump(umap_model, os.path.join(base_dir, "umap_model.joblib"))

    return {
        "status": "✅ Clustering pipeline complete",
        "scores": scores,
        "top_terms_per_cluster": top_terms_dict,
        "plot_path": image_path
    }