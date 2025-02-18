from utils.basic_preprocess import basic_preprocess
from utils.load_data import load_data
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import v_measure_score, silhouette_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import joblib
import numpy as np  



TARGET = "HeartDisease"
[X_data, Y_data] = basic_preprocess(*load_data("./data/data.csv", TARGET))

# best_kmeans_params = {'algorithm': 'elkan', 'copy_x': True, 'init': 'k-means++', 'max_iter': 100, 'n_clusters': 2, 'n_init': 30, 'random_state': None, 'tol': 0.0001, 'verbose': 0}
# alg = KMeans(**best_params)
# predictions = alg.fit_predict(X_data)
# print(silhouette_score(X_data, predictions, sample_size=100000))



param_distributions = {  
    'eps': np.arange(0.1, 2.0, 0.3),                       # Distancia máxima para considerar vecinos  
    'min_samples': np.arange(1, 15),                      # Número mínimo de puntos para considerar un cluster  
    'metric': ['euclidean', 'manhattan', 'chebyshev'],   # Distancia a utilizar  
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algoritmos de búsqueda  
    'leaf_size': np.arange(1, 50, 5)                      # Tamaño de la hoja (solo para ball_tree y kd_tree)  
}
random_search = RandomizedSearchCV(  
    estimator=DBSCAN(),
    param_distributions=param_distributions,  
    n_iter=100,
    scoring='calinsky',
    random_state=42,
    n_jobs=-1,
    cv=5                    
)
random_search.fit(X_data)
joblib.dump(random_search.best_estimator_, "dbscan.joblib")
