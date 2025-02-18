import pandas as pd
from preprocess.encoding import CustomOneHotEncoding
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


important_features = ['BMI', 'SleepTime', 'PhysicalHealth', 'MentalHealth', 'AgeCategory_80 or older', 'DiffWalking_Yes', 'Stroke_No', 'Stroke_Yes', 'PhysicalActivity_Yes', 'PhysicalActivity_No', 'AgeCategory_70-74', 'DiffWalking_No', 'GenHealth_Poor', 'Race_White', 'GenHealth_Fair', 'AgeCategory_75-79', 'Diabetic_Yes', 'Diabetic_No', 'Asthma_Yes', 'Asthma_No', 'AgeCategory_65-69', 'AgeCategory_60-64']


def basic_preprocess(X_data, Y_data):
    X_data = Pipeline([
        ("encoding", CustomOneHotEncoding()),
        ]).fit_transform(X_data)
    X_data = X_data[important_features]
    pca = PCA(n_components=0.99)
    X_data = pd.DataFrame(pca.fit_transform(X_data))
    return [X_data, Y_data]

