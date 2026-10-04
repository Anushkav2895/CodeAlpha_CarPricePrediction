"""
CodeAlpha Task: Car Price Prediction with Machine Learning
----------------------------------------------------------
Predicts a used car's selling price from its features: pre-processing,
feature engineering, several regression models, evaluation and charts.

Run:  python car_price_prediction.py
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).parent
FIG = BASE / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")
SEED = 42
TARGET = "Selling_Price"          # in lakhs (INR 100,000)
CURRENT_YEAR = 2020               # reference year for car age (data runs to 2018)


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=150)
    plt.close()


# ------------------------------------------------------------ 1. LOAD & CLEAN
def load_and_clean():
    df = pd.read_csv(BASE / "data" / "car_data.csv")
    df.columns = df.columns.str.strip()
    print("Shape:", df.shape)
    print("Missing values:", int(df.isna().sum().sum()))
    n_dup = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicate rows removed: {n_dup} -> {len(df)} rows left")
    for c in df.select_dtypes(exclude="number"):
        df[c] = df[c].astype(str).str.strip()
    return df


# ------------------------------------------------------------ 2. FEATURES
def engineer(df):
    df = df.copy()
    df["Car_Age"] = CURRENT_YEAR - df["Year"]
    df["Kms_per_Year"] = df["Driven_kms"] / df["Car_Age"].clip(lower=1)
    df["Log_Driven_kms"] = np.log1p(df["Driven_kms"])
    # Car_Name is a model name (ritz, city, fortuner...). Keep common ones and
    # lump the rest so the encoder does not memorise one-off names.
    counts = df["Car_Name"].value_counts()
    df["Model"] = df["Car_Name"].where(df["Car_Name"].map(counts) >= 5, "other")
    return df


NUMERIC = ["Present_Price", "Driven_kms", "Car_Age", "Kms_per_Year", "Log_Driven_kms", "Owner"]
CATEGORICAL = ["Fuel_Type", "Selling_type", "Transmission", "Model"]


# ------------------------------------------------------------ 3. EDA
def explore(df):
    print("\nSummary statistics:\n", df[[TARGET, "Present_Price", "Driven_kms", "Car_Age"]].describe().round(2))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df[TARGET], kde=True, ax=ax[0], color="steelblue")
    ax[0].set_title("Selling price (lakhs)")
    sns.histplot(np.log1p(df[TARGET]), kde=True, ax=ax[1], color="seagreen")
    ax[1].set_title("log(1 + selling price)")
    save("01_price_distribution")

    sns.heatmap(df[[TARGET, "Present_Price", "Driven_kms", "Car_Age", "Owner"]].corr(),
                annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation matrix")
    save("02_correlation")

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    sns.boxplot(data=df, x="Fuel_Type", y=TARGET, ax=ax[0, 0])
    sns.boxplot(data=df, x="Transmission", y=TARGET, ax=ax[0, 1])
    sns.boxplot(data=df, x="Selling_type", y=TARGET, ax=ax[1, 0])
    sns.boxplot(data=df, x="Owner", y=TARGET, ax=ax[1, 1])
    fig.suptitle("Selling price by category")
    save("03_price_by_category")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    sns.scatterplot(data=df, x="Present_Price", y=TARGET, hue="Fuel_Type", ax=ax[0])
    sns.scatterplot(data=df, x="Car_Age", y=TARGET, hue="Transmission", ax=ax[1])
    sns.scatterplot(data=df, x="Driven_kms", y=TARGET, ax=ax[2])
    ax[0].set_title("Price vs showroom price")
    ax[1].set_title("Price vs car age")
    ax[2].set_title("Price vs kms driven")
    save("04_price_relationships")


# ------------------------------------------------------------ 4. MODELS
def make_pipe(model):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("prep", pre), ("model", model)])


def log_target(model):
    """Train on log(1+price) so expensive cars do not dominate the error."""
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)


def evaluate(df):
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    # Stratify on price band so cheap AND expensive cars appear in train and test.
    bands = pd.qcut(y, 5, labels=False, duplicates="drop")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=bands)
    print(f"\nTrain: {len(X_tr)} rows | Test: {len(X_te)} rows")

    models = {
        "Linear Regression": make_pipe(LinearRegression()),
        "Ridge": make_pipe(Ridge(alpha=1.0)),
        "Random Forest": log_target(make_pipe(RandomForestRegressor(n_estimators=300, random_state=SEED))),
        "Gradient Boosting": log_target(make_pipe(GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=SEED))),
    }
    # 301 rows is small, so one split is noisy: the main score is repeated 5-fold CV
    # on ALL data; the held-out test score is reported next to it as a sanity check.
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}
    rows, fitted = [], {}
    for name, m in models.items():
        cvr = cross_validate(m, X, y, cv=cv, scoring=scoring)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        rows.append({
            "Model": name,
            "CV R2": cvr["test_r2"].mean(),
            "CV R2 std": cvr["test_r2"].std(),
            "CV MAE (lakhs)": -cvr["test_mae"].mean(),
            "CV RMSE (lakhs)": -cvr["test_rmse"].mean(),
            "Test R2": r2_score(y_te, pred),
            "Test MAE (lakhs)": mean_absolute_error(y_te, pred),
        })
        fitted[name] = m
    res = pd.DataFrame(rows).set_index("Model").round(3).sort_values("CV R2", ascending=False)
    print("\nMODEL COMPARISON (sorted by cross-validated R2)\n", res.to_string())
    res.to_csv(BASE / "outputs" / "model_comparison.csv")

    plt.figure(figsize=(8, 4))
    res[["CV R2", "Test R2"]].sort_values("CV R2").plot(kind="barh", figsize=(8, 4))
    plt.xlabel("R2"); plt.title("Model comparison (higher is better)")
    save("05_model_comparison")
    return res, fitted, X_te, y_te


# ------------------------------------------------------------ 5. BEST MODEL
def report_best(res, fitted, X_te, y_te):
    best = res.index[0]
    pipe = fitted[best]
    pred = pipe.predict(X_te)
    print(f"\nBest model: {best} (CV R2 = {res.loc[best, 'CV R2']:.3f}, "
          f"Test R2 = {res.loc[best, 'Test R2']:.3f}, Test MAE = {res.loc[best, 'Test MAE (lakhs)']:.2f} lakhs)")

    lim = [0, max(y_te.max(), pred.max()) * 1.05]
    plt.figure(figsize=(6, 6))
    plt.scatter(y_te, pred, alpha=0.7, color="navy")
    plt.plot(lim, lim, "r--", label="Perfect prediction")
    plt.xlabel("Actual price (lakhs)"); plt.ylabel("Predicted price (lakhs)")
    plt.title(f"Actual vs predicted - {best}"); plt.legend()
    save("06_actual_vs_predicted")

    resid = y_te - pred
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(pred, resid, alpha=0.7)
    ax[0].axhline(0, color="red", ls="--")
    ax[0].set(xlabel="Predicted", ylabel="Residual", title="Residuals vs predicted")
    sns.histplot(resid, kde=True, ax=ax[1])
    ax[1].set_title("Residual distribution")
    save("07_residuals")

    # Feature importance (tree models only)
    inner = getattr(pipe, "regressor_", pipe)       # unwrap log-target wrapper
    model = inner.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        names = inner.named_steps["prep"].get_feature_names_out()
        names = [n.split("__", 1)[1] for n in names]
        imp = pd.Series(model.feature_importances_, index=names).sort_values(ascending=False).head(12)
        print("\nTop features:\n", imp.round(3).to_string())
        plt.figure(figsize=(8, 5))
        imp.sort_values().plot(kind="barh", color="darkorange")
        plt.title(f"Top feature importances - {best}")
        save("08_feature_importance")

    joblib.dump(pipe, BASE / "outputs" / "best_model.joblib")
    return pipe, best


def demo_prediction(pipe):
    sample = pd.DataFrame([{
        "Present_Price": 9.5, "Driven_kms": 40000, "Owner": 0, "Fuel_Type": "Petrol",
        "Selling_type": "Dealer", "Transmission": "Manual", "Model": "other",
        "Car_Age": 6, "Kms_per_Year": 40000 / 6, "Log_Driven_kms": np.log1p(40000),
    }])[NUMERIC + CATEGORICAL]
    print(f"\nExample: 6-year-old petrol manual, 40,000 km, showroom 9.5 lakhs "
          f"-> predicted resale price {pipe.predict(sample)[0]:.2f} lakhs")


def main():
    df = engineer(load_and_clean())
    explore(df)
    res, fitted, X_te, y_te = evaluate(df)
    pipe, best = report_best(res, fitted, X_te, y_te)
    demo_prediction(pipe)
    print(f"\nFigures saved in {FIG}")


if __name__ == "__main__":
    main()
