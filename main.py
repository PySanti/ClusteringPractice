from utils.basic_preprocess import basic_preprocess
from utils.load_data import load_data
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import joblib

TARGET = "HeartDisease"
[X_data, Y_data] = basic_preprocess(*load_data("./data/data.csv", TARGET))
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, shuffle=True, random_state=42)
param_distributions = {  
    'n_clusters': [4,5,6],                      # Número de clusters entre 2 y 10  
    'init': ['k-means++', 'random'],                     # Métodos de inicialización  
    'n_init': [10, 20, 30],                              # Número de inicializaciones  
    'max_iter': [100, 300, 500],                        # Máximo número de iteraciones  
    'tol': [1e-4, 1e-3, 1e-2],                           # Tolerancia para la convergencia  
    'algorithm': ['lloyd', 'elkan']              # Algoritmos a utilizar  
}  
random_search = RandomizedSearchCV(KMeans(), param_distributions, n_iter=100, random_state=42, n_jobs=-1)  
random_search.fit(X_train)
joblib.dump(random_search.best_estimator_, "k_means.joblib")
