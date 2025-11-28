import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

BASE = r"C:\Users\aksha\OneDrive\Desktop\PROJECTS\AI-hyper"

X_train = pd.read_csv(BASE + r"\data\X_train.csv")
X_test = pd.read_csv(BASE + r"\data\X_test.csv")
y_train = pd.read_csv(BASE + r"\data\y_train.csv")
y_test = pd.read_csv(BASE + r"\data\y_test.csv")

numeric = ["SAR", "E_field", "Frequency_Index", "Depth"]
categorical = ["AMC_Presence", "Tissue_Region"]

preprocess = ColumnTransformer(
    [("num", StandardScaler(), numeric),
     ("cat", OneHotEncoder(handle_unknown='ignore'), categorical)]
)

models = {
    "LR": (LinearRegression(), {}),
    "DT": (DecisionTreeRegressor(), {
        "model__max_depth": [5, 10, 20]
    }),
    "MLP": (MLPRegressor(max_iter=2000), {
        "model__hidden_layer_sizes": [(64,32), (128,64)],
        "model__learning_rate_init": [0.001, 0.01]
    })
}

best_mse = float("inf")
best_model = None
best_name = ""

for name, (m, params) in models.items():
    pipe = Pipeline([("pre", preprocess), ("model", m)])
    
    grid = GridSearchCV(pipe, params, cv=3, scoring="neg_mean_squared_error")
    grid.fit(X_train, y_train.values.ravel())

    preds = grid.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    
    if mse < best_mse:
        best_mse = mse
        best_model = grid.best_estimator_
        best_name = name

joblib.dump(best_model, BASE + r"\models\hyperthermia_temp_model.pkl")
print(f"✅ Best model saved: {best_name}")
