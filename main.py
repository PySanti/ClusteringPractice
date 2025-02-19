from utils.basic_preprocess import basic_preprocess
from utils.load_data import load_data
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score, v_measure_score


TARGET = "HeartDisease"
[X_data, Y_data] = basic_preprocess(*load_data("./data/data.csv", TARGET))

best_kmeans_params = {'algorithm': 'elkan', 'copy_x': True, 'init': 'k-means++', 'max_iter': 100, 'n_clusters': 2, 'n_init': 30, 'random_state': None, 'tol': 0.0001, 'verbose': 0}

alg = KMeans(**best_kmeans_params)
clusters = alg.fit_predict(X_data)

print("Resultados de clusters para KMEANS")
print(f"Silhouette : {silhouette_score(X_data, clusters, sample_size=100_000)}")
print(f"Calinski : {calinski_harabasz_score(X_data, clusters)}")
print(f"V-measure : {v_measure_score(Y_data, clusters)}")


