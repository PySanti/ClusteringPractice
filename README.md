# Clustering Practice: Heart Disease

En este proyecto tomaré un dataset diseñado para realizar prácticas de clustering. 

El objetivo será tratar de lograr la generación de clusters lo más precisos posible teniendo las etiquetas.

Las fases serán las siguientes:

1- **Preprocesamiento**

&nbsp;1.1- Análisis inicial del conjunto

&nbsp;1.2- Cargar datos

&nbsp;1.3- Manejo de Nans e infinitos: no hay nans ni infinitos

&nbsp;1.4- Manejo de desequilibrio de datos: mucho desequilibrio de datos en muchas variables. Sin embargo, se probará continuar con el proceso sin aplicar técnicas como SMOTE.


&nbsp;1.5- Manejo de variables categóricas: se aplicó OneHotEncoding exitosamente.

&nbsp;1.6- Manejo de correlaciones: el estudio de correlaciones resulto que las siguientes características tienen menos de 0.2 y más de -0.2 porcentaje de correlación con el target:

['GenHealth', 'KidneyDisease', 'AgeCategory', 'Smoking', 'PhysicalActivity', 'SkinCancer', 'Sex', 'BMI', 'Asthma', 'Race', 'AlcoholDrinking', 'MentalHealth', 'Diabetic', 'SleepTime']

Se utilizará random forest como método de selección de características para confirmar lo anterior.

Después de implementar Random Forest para selección de características, resulto que las características con más de 1% de relevancia son las siguientes:

['BMI', 'SleepTime', 'PhysicalHealth', 'MentalHealth', 'AgeCategory_80 or older', 'DiffWalking_Yes', 'Stroke_No', 'Stroke_Yes', 'PhysicalActivity_Yes', 'PhysicalActivity_No', 'AgeCategory_70-74', 'DiffWalking_No', 'GenHealth_Poor', 'Race_White', 'GenHealth_Fair', 'AgeCategory_75-79', 'Diabetic_Yes', 'Diabetic_No', 'Asthma_Yes', 'Asthma_No', 'AgeCategory_65-69', 'AgeCategory_60-64']

Utilizaremos estas últimas.


&nbsp;1.7- Uso de PCA: después de implementar PCA, la **cantidad de features se redujo a 5**.

2- **Entrenamiento**

&nbsp;2.1-División del conjunto de datos en entrenamiento y test.

&nbsp;2.2-Uso de estrategias de selección de modelo: después de haber utilizado RandomizedSearch para selección de modelo de KMEANS, concluimos que los siguientes parámetros son los más óptimos para este dataset:

{'algorithm': 'elkan', 'copy_x': True, 'init': 'k-means++', 'max_iter': 100, 'n_clusters': 2, 'n_init': 30, 'random_state': None, 'tol': 0.0001, 'verbose': 0}

Ahora dejaremos al RandomizedSearch buscando la mejor combinación de hiperparámetros para DBSCAN para después evaluar cuál de los algoritmos de clustering funciona mejor para este dataset.

Después de todo no pudimos encontrar los hiperparámetros más óptimos para DBSCAN debido a su gran exigencia computacional, de todas formas, la diferencia en rendimiento es abismal con relación a KMEANS.

&nbsp;2.3-Almacenar modelos en disco: en este proceso nos dimos cuenta de que, al utilizar algoritmos de aprendizaje no supervisado como son los algoritmos de clustering, dumpear los modelos usando joblib no van a dumpear los clusters sino el algoritmo utilizado para ello. Por eso es que hay que utilizar el método **fit_predict**.



3- **Evaluación**

&nbsp;3.1- Comparar resultados de para coeficiente de silhouette e índice de calinski.

Estos son los resultados finales:

Resultados de clusters para KMEANS

silhouette : 0.6084890376348987
calinski : 196135.7319261467

Resultados de clusters para DBSCAN

silhouette : -0.6709401119693488
calinski : 264.00824658655273
V-measure : 0.01607311019499402


Podemos concluir que KMEANS es el mejor algoritmo para este caso concreto. Ademas, vemos que a pesar de obtener un buen coeficiente de silhouette, obtenemos ademas un bajo V-measure. Esto se debe a que existe una fuerte relacion entre los datos de los clusters, pero esa relacion no se corresponde exactamente con las clases, o en otras palabras, existen dos subconjuntos de personas principales dentro del conjunto de datos estudiados que guardan una fuerte relacion entre ellos, sin embargo, esa relacion no se corresponde con que tuvieron un problemas del corazon o no. Interesante.
