import sys
import os
import numpy as np
import joblib
from fastapi import APIRouter
from pydantic import BaseModel
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt

router = APIRouter()

class ClusteringRequest(BaseModel):
    dataset_path: str
    num_clusters: int = 5
    sample_size: int = 5000
    max_iter: int = 100
    top_terms: int = 10
    queries: list[str] = []  # قائمة الكويريات للتصنيف (اختياري)

def create_clusters(doc_vector, num_clusters=5, max_iter=100):
    model = MiniBatchKMeans(n_clusters=num_clusters, batch_size=2000, max_iter=max_iter)
    model.fit(doc_vector)
    return model

def top_terms_for_clusters(model, tfidf_matrix, vectorizer, num_terms=10):
    labels = model.labels_
    unique_labels = set(labels)
    terms = vectorizer.get_feature_names_out()
    cluster_terms = {}

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        cluster_sum = np.sum(tfidf_matrix[indices], axis=0).A1
        top_indices = np.argsort(cluster_sum)[::-1][:num_terms]
        cluster_terms[label] = [terms[i] for i in top_indices]
    return cluster_terms

def predict_clusters_for_queries(vectorizer, model, svd_model, queries):
    if not queries:
        return []
    X = vectorizer.transform(queries)
    reduced = svd_model.transform(X)
    preds = model.predict(reduced)
    return preds.tolist()

def plot_and_save_clusters(reduced_data, labels, save_path):
    plt.figure(figsize=(10, 8))
    unique_labels = set(labels)
    palette = plt.cm.get_cmap('viridis', len(unique_labels))

    for cluster in unique_labels:
        color = palette(cluster)
        plt.scatter(reduced_data[labels == cluster, 0],
                    reduced_data[labels == cluster, 1],
                    label=f'Cluster {cluster}',
                    s=50,
                    alpha=0.7,
                    edgecolors='k',
                    color=[color])
    plt.title('Clusters Visualization')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

@router.post("/clustering")
def run_clustering(request: ClusteringRequest):
    safe_path = request.dataset_path.replace("/", "__")
    base_dir = os.path.join("db", safe_path)

    try:
        vectorizer = joblib.load(os.path.join(base_dir, "vectorizer.joblib"))
        tfidf_matrix = joblib.load(os.path.join(base_dir, "tfidf_matrix.joblib"))
    except Exception as e:
        return {"error": f"Failed loading TF-IDF files: {e}"}

    tfidf_matrix = csr_matrix(tfidf_matrix)
    sample_size = min(request.sample_size, tfidf_matrix.shape[0])
    np.random.seed(42)
    sample_indices = np.random.choice(tfidf_matrix.shape[0], sample_size, replace=False)
    sampled_tfidf_matrix = tfidf_matrix[sample_indices]

    # تقليل الأبعاد
    svd = TruncatedSVD(n_components=2)
    reduced_tfidf = svd.fit_transform(sampled_tfidf_matrix)

    # تدريب النموذج
    model = create_clusters(reduced_tfidf, num_clusters=request.num_clusters, max_iter=request.max_iter)
    labels = model.labels_

    # 🔁 حفظ النموذجين
    joblib.dump(model, os.path.join(base_dir, "kmeans_model.joblib"))
    joblib.dump(svd, os.path.join(base_dir, "svd_model.joblib"))

    # 🧠 تخزين عنقود كل مستند
    doc_clusters = model.labels_
    joblib.dump(doc_clusters, os.path.join(base_dir, "doc_clusters.joblib"))

    # المقاييس
    silhouette_avg = silhouette_score(reduced_tfidf, labels)
    davies_bouldin = davies_bouldin_score(reduced_tfidf, labels)
    calinski_harabasz = calinski_harabasz_score(reduced_tfidf, labels)

    # استخراج أهم الكلمات
    cluster_terms = top_terms_for_clusters(model, sampled_tfidf_matrix, vectorizer, num_terms=request.top_terms)

    # رسم العناقيد
    image_path = os.path.join(base_dir, "clusters_plot.png")
    plot_and_save_clusters(reduced_tfidf, labels, image_path)

    # تصنيف الاستعلامات (إن وجدت)
    query_preds = predict_clusters_for_queries(vectorizer, model, svd, request.queries)


    return {
        "status": "Done clustering",
        "silhouette_score": float(silhouette_avg),
        "davies_bouldin_index": float(davies_bouldin),
        "calinski_harabasz_index": float(calinski_harabasz),
        "num_clusters": int(request.num_clusters),
        "top_terms_per_cluster": {int(k): v for k, v in cluster_terms.items()},
        "plot_image_path": image_path,
        "query_predictions": list(query_preds)
    }


    
####################################################
def find_optimal_clusters(X, min_k=2, max_k=10, max_iter=100):
    best_k = min_k
    best_score = -1
    all_scores = []

    for k in range(min_k, max_k + 1):
        model = MiniBatchKMeans(n_clusters=k, max_iter=max_iter)
        labels = model.fit_predict(X)

        try:
            silhouette = silhouette_score(X, labels)
            db_score = davies_bouldin_score(X, labels)
            ch_score = calinski_harabasz_score(X, labels)

            score_combo = silhouette - db_score  # مزيج مؤقت للمقارنة

            all_scores.append({
                "k": k,
                "silhouette": silhouette,
                "davies_bouldin": db_score,
                "calinski_harabasz": ch_score,
                "score_combo": score_combo
            })

            if score_combo > best_score:
                best_score = score_combo
                best_k = k

        except Exception:
            continue

    return best_k, all_scores

class EvaluationRequest(BaseModel):
    dataset_path: str
    sample_size: int = 5000
    max_iter: int = 100
    min_k: int = 2
    max_k: int = 10


@router.post("/evaluate_k")
def evaluate_k_range(request: EvaluationRequest):
    safe_path = request.dataset_path.replace("/", "__")
    base_dir = os.path.join("db", safe_path)

    try:
        vectorizer = joblib.load(os.path.join(base_dir, "vectorizer.joblib"))
        tfidf_matrix = joblib.load(os.path.join(base_dir, "tfidf_matrix.joblib"))
    except Exception as e:
        return {"error": f"Failed loading TF-IDF files: {e}"}

    tfidf_matrix = csr_matrix(tfidf_matrix)
    sample_size = min(request.sample_size, tfidf_matrix.shape[0])
    np.random.seed(42)
    sample_indices = np.random.choice(tfidf_matrix.shape[0], sample_size, replace=False)
    sampled_tfidf_matrix = tfidf_matrix[sample_indices]

    svd = TruncatedSVD(n_components=2)
    reduced_tfidf = svd.fit_transform(sampled_tfidf_matrix)

    best_k, score_table = find_optimal_clusters(
        reduced_tfidf,
        min_k=request.min_k,
        max_k=request.max_k,
        max_iter=request.max_iter
    )

    return {
        "best_k": best_k,
        "evaluated_scores": score_table
    }
