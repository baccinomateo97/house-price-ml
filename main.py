import pandas as pd

df = pd.read_excel("data/houses.xlsx")

X = df[[
    "X2 house age",
    "X3 distance to the nearest MRT station",
    "X4 number of convenience stores",
    "X5 latitude",
    "X6 longitude"
]]

y = df["Y house price of unit area"]

print(X.head())
print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Train:")
print(X_train.shape)

print("Test:")
print(X_test.shape)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

print("Coeficiente:", model.coef_[0])
print("Intercept:", model.intercept_)

predicciones = model.predict(X_test)

resultados = pd.DataFrame({
    "Real": y_test,
    "Prediccion": predicciones
})

print("\nPredicciones:")
print(resultados.head(10))

from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, predicciones)
r2 = r2_score(y_test, predicciones)

coeficientes = pd.DataFrame({
    "Feature": X.columns,
    "Coeficiente": model.coef_
})

print("\nCoeficientes:")
print(coeficientes)

print("\nMAE:", mae)
print("R²:", r2)

from sklearn.dummy import DummyRegressor

dummy = DummyRegressor(strategy="mean")

dummy.fit(X_train, y_train)

dummy_pred = dummy.predict(X_test)

dummy_mae = mean_absolute_error(y_test, dummy_pred)
dummy_r2 = r2_score(y_test, dummy_pred)

print("\n--- BASELINE ---")
print("MAE:", dummy_mae)
print("R²:", dummy_r2)

print("\n--- LINEAR REGRESSION ---")
print("MAE:", mae)
print("R²:", r2)

from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(
    random_state=42
)

tree.fit(X_train, y_train)

tree_pred = tree.predict(X_test)

tree_mae = mean_absolute_error(y_test, tree_pred)
tree_r2 = r2_score(y_test, tree_pred)

print("\n--- DECISION TREE ---")
print("MAE:", tree_mae)
print("R²:", tree_r2)

# Predicciones sobre TRAIN
tree_train_pred = tree.predict(X_train)

tree_train_mae = mean_absolute_error(
    y_train,
    tree_train_pred
)

tree_train_r2 = r2_score(
    y_train,
    tree_train_pred
)

print("\n--- DECISION TREE: TRAIN vs TEST ---")

print("TRAIN")
print("MAE:", tree_train_mae)
print("R²:", tree_train_r2)

print("\nTEST")
print("MAE:", tree_mae)
print("R²:", tree_r2)

print("\n--- PRUEBA DE PROFUNDIDADES ---")

for depth in [2, 3, 4, 5, 6, 10, None]:

    tree_depth = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    tree_depth.fit(X_train, y_train)

    train_pred = tree_depth.predict(X_train)
    test_pred = tree_depth.predict(X_test)

    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)

    print(
        f"Depth: {str(depth):>4} | "
        f"Train MAE: {train_mae:.2f} | "
        f"Test MAE: {test_mae:.2f}"
    )

    from sklearn.model_selection import cross_val_score

print("\n--- 5-FOLD CROSS-VALIDATION ---")

for depth in [2, 3, 4, 5, 6, 10, None]:

    tree_cv = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    scores = cross_val_score(
        tree_cv,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_absolute_error"
    )

    mae_scores = -scores

    print(
        f"Depth: {str(depth):>4} | "
        f"CV MAE: {mae_scores.mean():.2f} | "
        f"Std: {mae_scores.std():.2f}"
    )

    # Modelo final elegido mediante Cross-Validation
final_tree = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)

# Entrenamos usando todo el conjunto de TRAIN
final_tree.fit(X_train, y_train)

# Ahora hacemos la evaluación final sobre TEST
final_pred = final_tree.predict(X_test)

# Calculamos las métricas
final_mae = mean_absolute_error(y_test, final_pred)
final_r2 = r2_score(y_test, final_pred)

print("\n--- FINAL DECISION TREE ---")
print("max_depth: 3")
print("Test MAE:", final_mae)
print("Test R²:", final_r2)


from sklearn.ensemble import RandomForestRegressor

random_forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

random_forest.fit(X_train, y_train)

rf_pred = random_forest.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("\n--- RANDOM FOREST ---")
print("Número de árboles: 100")
print("Test MAE:", rf_mae)
print("Test R²:", rf_r2)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": random_forest.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n--- FEATURE IMPORTANCE ---")
print(feature_importance)

from sklearn.inspection import permutation_importance

perm = permutation_importance(
    random_forest,
    X_test,
    y_test,
    n_repeats=20,
    random_state=42,
    scoring="neg_mean_absolute_error"
)

perm_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": perm.importances_mean
})

perm_importance = perm_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n--- PERMUTATION IMPORTANCE ---")
print(perm_importance)