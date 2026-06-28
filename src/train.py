import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "../data/Software_feature_shipment_data_set.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "../outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)



# 1. LOAD & PREPROCESS


def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df)} rows × {len(df.columns)} columns")
    print(f"[INFO] Missing values:\n{df.isnull().sum()}\n")

    # Parse date → temporal features
    df["planned_shipment_date"] = pd.to_datetime(df["planned_shipment_date"])
    df["shipment_month"]        = df["planned_shipment_date"].dt.month
    df["shipment_quarter"]      = df["planned_shipment_date"].dt.quarter
    df["shipment_dayofweek"]    = df["planned_shipment_date"].dt.dayofweek   # Mon=0
    df["shipment_week"]         = df["planned_shipment_date"].dt.isocalendar().week.astype(int)
    df["is_end_of_quarter"]     = df["shipment_month"].isin([3, 6, 9, 12]).astype(int)

    # Interaction / ratio features
    df["complexity_per_dev"]     = df["feature_complexity"] / df["team_size"].clip(lower=1)
    df["dependency_blocker_sum"] = df["num_dependencies"] + df["num_blockers"]
    df["effective_sprint_days"]  = df["sprint_length_weeks"] * 5 - df["holidays_in_sprint"]

    # Drop raw date column
    df.drop(columns=["planned_shipment_date"], inplace=True)

    print(f"[INFO] After feature engineering: {df.shape[1]} columns")
    print(f"[INFO] Columns: {list(df.columns)}\n")
    return df



# 2. EDA PLOTS


def run_eda(df: pd.DataFrame):
    print("[INFO] Running EDA...")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("EDA — Software Feature Shipment Dataset", fontsize=16, fontweight="bold", y=1.01)

    # 1. Target distribution
    axes[0, 0].hist(df["delay_days"], bins=28, color="#4C72B0", edgecolor="white", linewidth=0.6)
    axes[0, 0].set_title("Target Distribution: delay_days", fontweight="bold")
    axes[0, 0].set_xlabel("Delay (days)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].axvline(df["delay_days"].mean(), color="red", linestyle="--", label=f"Mean = {df['delay_days'].mean():.1f}")
    axes[0, 0].axvline(df["delay_days"].median(), color="orange", linestyle="--", label=f"Median = {df['delay_days'].median():.1f}")
    axes[0, 0].legend()

    # 2. Correlation heatmap (vs target only — keeps it readable)
    corr = df[numeric_cols].corr()[["delay_days"]].drop("delay_days").sort_values("delay_days", ascending=False)
    sns.heatmap(corr, ax=axes[0, 1], annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, cbar=True, vmin=-1, vmax=1)
    axes[0, 1].set_title("Feature Correlation with delay_days", fontweight="bold")

    # 3. Feature complexity vs delay
    axes[0, 2].scatter(df["feature_complexity"], df["delay_days"], alpha=0.3, color="#DD8452", s=15)
    axes[0, 2].set_title("Feature Complexity vs Delay", fontweight="bold")
    axes[0, 2].set_xlabel("feature_complexity")
    axes[0, 2].set_ylabel("delay_days")
    m, b = np.polyfit(df["feature_complexity"], df["delay_days"], 1)
    x_line = np.linspace(df["feature_complexity"].min(), df["feature_complexity"].max(), 100)
    axes[0, 2].plot(x_line, m * x_line + b, color="red", lw=2, label="Trend")
    axes[0, 2].legend()

    # 4. Team size vs delay
    axes[1, 0].scatter(df["team_size"], df["delay_days"], alpha=0.3, color="#55A868", s=15)
    axes[1, 0].set_title("Team Size vs Delay", fontweight="bold")
    axes[1, 0].set_xlabel("team_size")
    axes[1, 0].set_ylabel("delay_days")

    # 5. Past avg delay vs actual delay
    axes[1, 1].scatter(df["past_avg_delay_days"], df["delay_days"], alpha=0.3, color="#C44E52", s=15)
    axes[1, 1].set_title("Past Avg Delay vs Actual Delay", fontweight="bold")
    axes[1, 1].set_xlabel("past_avg_delay_days")
    axes[1, 1].set_ylabel("delay_days")
    m2, b2 = np.polyfit(df["past_avg_delay_days"], df["delay_days"], 1)
    x2 = np.linspace(df["past_avg_delay_days"].min(), df["past_avg_delay_days"].max(), 100)
    axes[1, 1].plot(x2, m2 * x2 + b2, color="red", lw=2)

    # 6. Delay by quarter (box plot)
    quarters = sorted(df["shipment_quarter"].unique())
    data_by_q = [df[df["shipment_quarter"] == q]["delay_days"].values for q in quarters]
    bp = axes[1, 2].boxplot(data_by_q, patch_artist=True,
                             tick_labels=[f"Q{q}" for q in quarters])
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 2].set_title("Delay Distribution by Quarter", fontweight="bold")
    axes[1, 2].set_xlabel("Quarter")
    axes[1, 2].set_ylabel("delay_days")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "eda_plots.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] EDA plots saved → {out}\n")



# 3. BASELINE MODELS


def run_baselines(X_tr, X_te, y_tr, y_te) -> dict:
    print("[INFO] Running baseline models...")
    models = {
        "Random Forest (default)":      RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting (default)":  GradientBoostingRegressor(n_estimators=100, random_state=42),
        "XGBoost (default)":            xgb.XGBRegressor(objective="reg:squarederror",
                                                          n_estimators=100, random_state=42, verbosity=0),
    }
    results = {}
    for name, mdl in models.items():
        mdl.fit(X_tr, y_tr)
        preds = mdl.predict(X_te)
        mae  = mean_absolute_error(y_te, preds)
        rmse = mean_squared_error(y_te, preds) ** 0.5
        r2   = r2_score(y_te, preds)
        results[name] = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}
        print(f"  {name:<35}  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
    print()
    return results



# 4. FINE-TUNE XGBOOST


def finetune_xgboost(X_tr, y_tr):
    print("[INFO] Fine-tuning XGBoost with GridSearchCV (5-fold CV)...")
    print("[INFO] This may take 3–8 minutes depending on your machine...\n")

    param_grid = {
        "n_estimators":     [200, 400, 600],
        "max_depth":        [3, 5, 7],
        "learning_rate":    [0.01, 0.05, 0.1],
        "subsample":        [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
        "reg_alpha":        [0, 0.1],
        "reg_lambda":       [1, 1.5],
    }

    base = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        eval_metric="mae",
    )

    gs = GridSearchCV(
        base,
        param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    gs.fit(X_tr, y_tr)

    print(f"\n[INFO] Best CV MAE  : {-gs.best_score_:.4f}")
    print(f"[INFO] Best params  :\n{json.dumps(gs.best_params_, indent=4)}\n")
    return gs.best_estimator_, gs.best_params_, -gs.best_score_



# 5. FEATURE IMPORTANCE PLOT


def plot_feature_importance(model, feature_names: list):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    top_n = min(15, len(feature_names))

    top_features = [feature_names[i] for i in idx[:top_n]][::-1]
    top_values   = importances[idx[:top_n]][::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top_features, top_values, color="#4C72B0", edgecolor="white", linewidth=0.5)

    # Value labels on bars
    for bar, val in zip(bars, top_values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    ax.set_title("XGBoost Feature Importances (Top 15)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.set_xlim(0, top_values.max() * 1.15)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Feature importance plot saved → {out}")


# 6. PREDICTION PLOTS


def plot_predictions(y_te, preds):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Fine-Tuned XGBoost — Test Set Results", fontsize=14, fontweight="bold")

    # Actual vs Predicted scatter
    axes[0].scatter(y_te, preds, alpha=0.4, color="#4C72B0", s=20)
    mn = min(float(y_te.min()), float(preds.min()))
    mx = max(float(y_te.max()), float(preds.max()))
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect prediction")
    axes[0].set_xlabel("Actual delay_days")
    axes[0].set_ylabel("Predicted delay_days")
    axes[0].set_title("Actual vs Predicted")
    axes[0].legend()

    mae  = mean_absolute_error(y_te, preds)
    r2   = r2_score(y_te, preds)
    axes[0].text(0.05, 0.92, f"MAE = {mae:.3f}\nR² = {r2:.3f}",
                 transform=axes[0].transAxes, fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # Residuals distribution
    residuals = preds - y_te
    axes[1].hist(residuals, bins=30, color="#DD8452", edgecolor="white", linewidth=0.6)
    axes[1].axvline(0, color="red", linestyle="--", lw=2, label="Zero error")
    axes[1].axvline(residuals.mean(), color="blue", linestyle="--", lw=1.5,
                    label=f"Mean residual = {residuals.mean():.2f}")
    axes[1].set_title("Residuals Distribution")
    axes[1].set_xlabel("Predicted − Actual (days)")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "predictions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Prediction plots saved → {out}")



# 7. BASELINE COMPARISON PLOT


def plot_baseline_comparison(baseline_results: dict, fine_tuned_mae: float):
    names = list(baseline_results.keys()) + ["XGBoost (fine-tuned)"]
    maes  = [v["MAE"] for v in baseline_results.values()] + [round(fine_tuned_mae, 4)]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, maes, color=colors, edgecolor="white", linewidth=0.5, width=0.5)

    for bar, val in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title("Model Comparison — Test MAE (lower is better)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean Absolute Error (days)")
    ax.set_ylim(0, max(maes) * 1.2)
    ax.set_xticklabels(names, rotation=12, ha="right")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Model comparison plot saved → {out}")



# MAIN


def main():
    print("=" * 65)
    print("   Britha AI — Feature Shipment Delay Predictor")
    print("=" * 65 + "\n")

    # 1. Load & preprocess
    df = load_and_preprocess(DATA_PATH)

    # 2. EDA
    run_eda(df)

    # 3. Split features / target
    FEATURE_COLS = [c for c in df.columns if c != "delay_days"]
    X = df[FEATURE_COLS].values
    y = df["delay_days"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[INFO] Train size = {len(X_tr)}  |  Test size = {len(X_te)}\n")

    # 4. Baselines
    baseline_results = run_baselines(X_tr, X_te, y_tr, y_te)

    # 5. Fine-tune XGBoost
    best_xgb, best_params, cv_mae = finetune_xgboost(X_tr, y_tr)

    # 6. Final evaluation on test set
    final_preds = best_xgb.predict(X_te)
    final_metrics = {
        "MAE":  round(mean_absolute_error(y_te, final_preds), 4),
        "RMSE": round(mean_squared_error(y_te, final_preds) ** 0.5, 4),
        "R2":   round(r2_score(y_te, final_preds), 4),
    }

    print("── Fine-Tuned XGBoost — Final Test Results ──")
    print(f"  MAE  = {final_metrics['MAE']}")
    print(f"  RMSE = {final_metrics['RMSE']}")
    print(f"  R²   = {final_metrics['R2']}")
    print(f"  5-Fold CV MAE = {cv_mae:.4f}\n")

    # 7. Plots
    plot_feature_importance(best_xgb, FEATURE_COLS)
    plot_predictions(y_te, final_preds)
    plot_baseline_comparison(baseline_results, final_metrics["MAE"])

    # 8. Save model
    model_path = os.path.join(OUTPUT_DIR, "xgb_shipment_predictor.json")
    best_xgb.save_model(model_path)
    print(f"\n[INFO] Model saved → {model_path}")

    # 9. Save all results to JSON
    results = {
        "feature_columns": FEATURE_COLS,
        "train_size": len(X_tr),
        "test_size":  len(X_te),
        "baseline_results": baseline_results,
        "best_params": best_params,
        "cv_mae": round(cv_mae, 4),
        "fine_tuned_test_metrics": final_metrics,
    }
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Results JSON saved → {results_path}")

    print("\n" + "=" * 65)
    print("Pipeline complete. Check the outputs/ folder.")
    print("=" * 65)


if __name__ == "__main__":
    main()