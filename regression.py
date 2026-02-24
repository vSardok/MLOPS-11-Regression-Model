import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("data/insurance.csv")

X = df.drop(columns=["charges"])
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

cat_cols = ["sex", "smoker", "region"]
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LinearRegression()),
])

pipe.fit(X_train, y_train)

preds = pipe.predict(X_test)
mse = mean_squared_error(y_test, preds)

print("TEST loss (MSE):", float(mse))
print("Cumple (<= 19,000,000):", mse <= 19_000_000)