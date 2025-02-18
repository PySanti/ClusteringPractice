from utils.basic_preprocess import basic_preprocess
from utils.load_data import load_data
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import joblib

TARGET = "HeartDisease"
[X_data, Y_data] = basic_preprocess(*load_data("./data/data.csv", TARGET))
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, shuffle=True, random_state=42)
model = joblib.load("./k_means.joblib")
