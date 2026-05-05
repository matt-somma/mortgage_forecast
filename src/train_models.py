import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from project_paths import MODEL_RESULTS_DIR, MODELING_DATA_DIR


DATASET_PATH = MODELING_DATA_DIR / "dgs10_long_1w.csv"
OUTPUT_DIR = MODEL_RESULTS_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [0.1, 1, 10, 50, 100, 250, 500, 1000]

CORE_DGS10_FEATURES = [
    "DGS10",
    "DGS2",
    "DGS3MO",
    "DGS5",
    "DGS30",
    "DFII10",
    "T10YIE",
    "yield_curve_10y_2y",
    "yield_curve_10y_3mo",
    "yield_curve_30y_10y",
    "dgs10_chg_1w",
    "dgs10_chg_4w",
    "dgs2_chg_1w",
    "dgs5_chg_1w",
    "dgs30_chg_1w",
    "dgs10_roll_std_4w",
    "dgs10_roll_std_12w",
    "VIXCLS",
    "vixcls_chg_1w",
    "vixcls_roll_std_4w",
]


def evaluate_predictions(y_true, y_pred):
    return {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def train_ridge_grid(df, dataset_label, feature_cols):
    X = df[feature_cols]
    y = df["target"]

    X_train = X[df["split"] == "train"]
    y_train = y[df["split"] == "train"]

    X_val = X[df["split"] == "validation"]
    y_val = y[df["split"] == "validation"]

    X_test = X[df["split"] == "test"]
    y_test = y[df["split"] == "test"]

    baseline_pred = X_test["dgs10_chg_1w"]
    baseline_metrics = evaluate_predictions(y_test, baseline_pred)

    print(f"\n=== {dataset_label} ===")
    print("Baseline Test RMSE:", baseline_metrics["rmse"])
    print("Baseline Test MAE:", baseline_metrics["mae"])

    results = []
    feature_importance_frames = []

    for alpha in ALPHAS:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )

        model.fit(X_train, y_train)

        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)

        val_metrics = evaluate_predictions(y_val, y_pred_val)
        test_metrics = evaluate_predictions(y_test, y_pred_test)

        results.append(
            {
                "dataset_label": dataset_label,
                "alpha": alpha,
                "baseline_test_rmse": baseline_metrics["rmse"],
                "baseline_test_mae": baseline_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"],
            }
        )

        ridge = model.named_steps["ridge"]

        feature_importance = (
            pd.Series(ridge.coef_, index=X.columns)
            .sort_values(key=abs, ascending=False)
            .reset_index()
        )

        feature_importance.columns = ["feature_name", "coefficient"]
        feature_importance["dataset_label"] = dataset_label
        feature_importance["alpha"] = alpha

        feature_importance_frames.append(feature_importance)

    results_df = pd.DataFrame(results)
    importance_df = pd.concat(feature_importance_frames, ignore_index=True)

    print("\nResults:")
    print(results_df.sort_values("test_rmse").to_string(index=False))

    best = results_df.sort_values("test_rmse").iloc[0]
    best_alpha = best["alpha"]

    print("\nBest model by test RMSE:")
    print(best.to_string())

    print("\nTop 15 features for best alpha:")
    print(
        importance_df[importance_df["alpha"] == best_alpha]
        .head(15)
        [["feature_name", "coefficient"]]
        .to_string(index=False)
    )

    return results_df, importance_df


def main():
    df = pd.read_csv(DATASET_PATH, parse_dates=["week_date"])

    full_features = [
        c for c in df.columns
        if c not in ["week_date", "dataset_name", "target", "split"]
    ]

    full_results, full_importance = train_ridge_grid(
        df=df,
        dataset_label="dgs10_full_features",
        feature_cols=full_features,
    )

    core_results, core_importance = train_ridge_grid(
        df=df,
        dataset_label="dgs10_core_features",
        feature_cols=CORE_DGS10_FEATURES,
    )

    results = pd.concat([full_results, core_results], ignore_index=True)
    importance = pd.concat([full_importance, core_importance], ignore_index=True)

    results_path = OUTPUT_DIR / "dgs10_ridge_results.csv"
    importance_path = OUTPUT_DIR / "dgs10_ridge_feature_importance.csv"

    results.to_csv(results_path, index=False)
    importance.to_csv(importance_path, index=False)

    print("\nSaved:")
    print(results_path)
    print(importance_path)


if __name__ == "__main__":
    main()
