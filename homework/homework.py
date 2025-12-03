#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

def load_data():
    import pandas as pd

    train = pd.read_csv("./files/input/train_data.csv.zip", compression="zip")
    test = pd.read_csv("./files/input/test_data.csv.zip", compression="zip" )

    return train, test

def preprocess_data(df):
    df = df.copy()
    df["Age"] = 2021 - df["Year"]
    df = df.drop(columns=["Year", "Car_Name"])
    return df

train_data, test_data = load_data()

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

x_train = train_data.drop(columns=["Present_Price"])
y_train = train_data["Selling_Price"]
x_test = test_data.drop(columns=["Present_Price"])
y_test = test_data["Selling_Price"]

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

def build_pipeline():
    categorical_features = [col for col in x_train.columns if x_train[col].dtype == "object"]
    numeric_features = [col for col in x_train.columns if x_train[col].dtype != "object"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("select_k_best", SelectKBest(score_func=f_regression)),
        ("linear_model", LinearRegression())
    ])

    return pipeline

def optimize_params(pipeline):
    from sklearn.model_selection import GridSearchCV
    num_features = len(x_train.columns)
    param_grid = {
        "select_k_best__k": list(range(1, num_features + 1)),
        "select_k_best__score_func": [f_regression, mutual_info_regression],
        "preprocessing__num__feature_range": [(0, 1), (-1, 1)],
        "linear_model__fit_intercept": [True, False]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="neg_mean_absolute_error"
    )

    grid_search.fit(x_train, y_train)
    return grid_search

pipeline = build_pipeline()
model = optimize_params(pipeline)

def save_model(estimator):
    import gzip
    import pickle
    import os

    os.makedirs("./files/models", exist_ok=True)
    with gzip.open("./files/models/model.pkl.gz", "wb") as f:
        pickle.dump(estimator, f)
save_model(model)

def calc_metrics(estimator):
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import os
    import json



    os.makedirs("./files/output", exist_ok=True)
    with open("./files/output/metrics.json", "w") as f:
        y_train_pred = estimator.predict(x_train)
        y_test_pred = estimator.predict(x_test)

        train_metrics = {
            "type": "metrics",
            "dataset": "train",
            "r2": r2_score(y_train, y_train_pred),
            "mse": mean_squared_error(y_train, y_train_pred),
            "mad": mean_absolute_error(y_train, y_train_pred)
        }

        f.write(json.dumps(train_metrics) + "\n")

        test_metrics = {
            "type": "metrics",
            "dataset": "test",
            "r2": r2_score(y_test, y_test_pred),
            "mse": mean_squared_error(y_test, y_test_pred),
            "mad": mean_absolute_error(y_test, y_test_pred)
        }

        f.write(json.dumps(test_metrics) + "\n")

calc_metrics(model)