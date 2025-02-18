# Clustering Practice : Heart Disease

En este proyecto tomare un dataset diseniago para realizar practicas de clustering. 

El objetivo sera tratar de lograr la generacion de clusters lo mas precisos posibles teniendo las etiquetas.

Las fases seran las siguientes:

1- **Preprocesamiento**

&nbsp;1.1- Analisis inicial del conjunto

&nbsp;1.2- Cargar datos

&nbsp;1.3- Manejo de Nans e infinitos: no hay nans ni infinitos

&nbsp;1.4- Manejo de desequilibrio de datos: mucho desequilibrio de datos en muchas variables. Sin embargo, se probara continuar con el proceso sin aplicar tecnicas como SMOTE.


&nbsp;1.5- Manejo de variables categoricas: se aplico OneHotEncoding exitosamente.

&nbsp;1.6- Manejo de correlaciones: el estudio de correlaciones resulto que las siguientes caracteristicas tienen menos de 0.2 y mas de -0.2 porcentaje de correlacion con el target:

['GenHealth', 'KidneyDisease', 'AgeCategory', 'Smoking', 'PhysicalActivity', 'SkinCancer', 'Sex', 'BMI', 'Asthma', 'Race', 'AlcoholDrinking', 'MentalHealth', 'Diabetic', 'SleepTime']

Se utilizara random forest como metodo de seleccion de caracteristicas para confirmar lo anterior.

Despues de implementar Random Forest para seleccion de caracteristicas, resulto que las caracteristicas con mas de 1% de relevancia son las siguientes:

['BMI', 'SleepTime', 'PhysicalHealth', 'MentalHealth', 'AgeCategory_80 or older', 'DiffWalking_Yes', 'Stroke_No', 'Stroke_Yes', 'PhysicalActivity_Yes', 'PhysicalActivity_No', 'AgeCategory_70-74', 'DiffWalking_No', 'GenHealth_Poor', 'Race_White', 'GenHealth_Fair', 'AgeCategory_75-79', 'Diabetic_Yes', 'Diabetic_No', 'Asthma_Yes', 'Asthma_No', 'AgeCategory_65-69', 'AgeCategory_60-64']

Utilizaremos estas ultimas.


&nbsp;1.7- Uso de PCA: despues de implementar PCA, la **cantidad de features se redujo a 5**.

&nbsp;1.8- *En caso de no usar PCA: Normalizacion*

2- **Entrenamiento**

&nbsp;2.1-Division del conjunto de datos en entrenamiento y test.

&nbsp;2.2-Uso de estrategias de seleccion de modelo: despues de haber utilizado RandomizedSearch para seleccion de modelo de KMEANS, concluimos que los siguientes parametros son los mas optimos para este dataset:

{'algorithm': 'elkan', 'copy_x': True, 'init': 'k-means++', 'max_iter': 100, 'n_clusters': 2, 'n_init': 30, 'random_state': None, 'tol': 0.0001, 'verbose': 0}

Ahora dejaremos al RandomizedSearch buscando la mejor combinacion de hiperparametros para DBSCAN para despues evaluar cual de los algoritmos de clustering funciona mejor para este dataset.

&nbsp;2.3-Almacenar modelos en disco: en este proceso nos dimos cuenta que, al utilizar algoritmos de aprendizaje no supervisado como son los algoritmos de clustering, dumpear los modelos usando joblib no van a dumpear los clusters sino el algoritmo utilizado para ello. Por eso es que hay que utilzar el metodo **fit_predict**.

3- **Evaluacion**

&nbsp;3.1- Comparar resultados de para coeficiente de silhouette e indice de calinski.
