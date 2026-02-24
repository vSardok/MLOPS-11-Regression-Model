import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

np.random.seed(42)

# Leemos el dataset
DATA_PATH = "insurance.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"No encontré {DATA_PATH} en el folder actual.")

df = pd.read_csv(DATA_PATH)
if "charges" not in df.columns:
    raise ValueError("No existe la columna objetivo 'charges' en tu CSV.")

# separamos la variable objetivo y las características
y = df["charges"]
X = df.drop(columns=["charges"]).copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# separamos las columnas categóricas y numéricas
cat_cols = ["sex", "smoker", "region"]
num_cols = [c for c in X.columns if c not in cat_cols]

# convertimos en binario las variables categóricas y dejamos las numéricas tal cual
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# entrenamos el modelo
model = GradientBoostingRegressor(random_state=42)

# creamos un pipeline que primero preprocesa los datos y luego entrena el modelo
pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model),
])

pipe.fit(X_train, y_train)

# evaluamos el modelo en el set de test
preds = pipe.predict(X_test)
test_mse = mean_squared_error(y_test, preds)

print("\nTEST loss (MSE) in dollars:", float(test_mse))

if test_mse <= 19_000_000:
    print("Cumple: MSE <= 19,000,000")
else:
    print("No cumple aún")

# creamos un reporte con las predicciones, los valores reales y el error
report = pd.DataFrame({
    "predicted_charges": preds,
    "real_charges": y_test.values,
    "error": preds - y_test.values
})

report.to_csv("regression_report.csv", index=False)
print("\nGuardado: regression_report.csv")
print(report.head(10))