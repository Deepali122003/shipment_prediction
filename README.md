# 🚀 Feature Shipment Delay Predictor
### Britha AI — AI/ML Engineer Technical Assessment

A machine learning pipeline that predicts **how many days a software feature will be delayed** from its planned shipment date, trained on historical American Express feature delivery data.

---

## 📊 Results Summary

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Random Forest (default) | 1.185 | 1.485 | 0.911 |
| Gradient Boosting (default) | 0.933 | 1.182 | 0.943 |
| XGBoost (default) | 1.099 | 1.410 | 0.919 |
| **XGBoost (fine-tuned)** | **0.949** | **1.185** | **0.943** |

> **The fine-tuned XGBoost model achieves an R² of 0.943**, meaning it explains 94.3% of variance in shipment delays, with a mean prediction error of less than 1 day.

---

## 📁 Project Structure

shipment_prediction/

├── data/

│   └── Software_feature_shipment_data_set.csv

├── src/

│   └── train.py

├── outputs/

│   ├── eda_plots.png

│   ├── feature_importance.png

│   ├── predictions.png

│   ├── model_comparison.png

│   ├── xgb_shipment_predictor.json

│   └── results.json

├── requirements.txt

└── README.md


---

## 🛠️ Setup & Run

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/shipment-prediction.git
cd shipment-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the pipeline**
```bash
python src/train.py
```

All outputs (plots, model, metrics) will be saved to the `outputs/` folder automatically.

---

## 📋 Dataset Overview

- **Rows:** 1,300 historical feature shipments
- **Missing values:** None
- **Target variable:** `delay_days` — number of days a feature shipped after its planned date
- **Date range:** 2021 to 2045

| Feature | Description |
|---------|-------------|
| `planned_shipment_date` | Originally planned delivery date |
| `team_size` | Number of developers on the feature |
| `feature_complexity` | Complexity score (1–10) |
| `num_dependencies` | Number of external dependencies |
| `sprint_length_weeks` | Length of the sprint in weeks |
| `num_blockers` | Number of active blockers |
| `holidays_in_sprint` | Holiday days within the sprint |
| `priority_encoded` | Feature priority (encoded) |
| `past_avg_delay_days` | Team's historical average delay |
| `estimated_bug_count` | Bugs estimated during planning |

---

## ⚙️ Feature Engineering

| Engineered Feature | Formula | Rationale |
|-------------------|---------|-----------|
| `shipment_month` | extracted from date | Captures seasonal delivery patterns |
| `shipment_quarter` | extracted from date | Q4 often sees more delays due to holidays |
| `shipment_dayofweek` | extracted from date | End-of-week shipments may be rushed |
| `shipment_week` | ISO week number | Captures annual sprint calendar patterns |
| `is_end_of_quarter` | months 3,6,9,12 → 1 else 0 | End-of-quarter pressure affects timelines |
| `complexity_per_dev` | `feature_complexity / team_size` | Captures per-person workload |
| `dependency_blocker_sum` | `num_dependencies + num_blockers` | Combined external risk signal |
| `effective_sprint_days` | `sprint_length_weeks × 5 − holidays_in_sprint` | Actual working days available |

---

## 🧠 Model Selection & Methodology

### Why XGBoost?
- Handles tabular data extremely well without scaling
- Built-in regularization (L1/L2) prevents overfitting
- Fast training with `n_jobs=-1` parallelization
- Gradient boosting captures non-linear feature interactions naturally

### Baseline First
Three baseline models were evaluated before any tuning. Gradient Boosting already performed strongly (MAE 0.933), confirming that the boosting approach was the right direction.

### Fine-Tuning with GridSearchCV
A 5-fold cross-validated grid search over 432 hyperparameter combinations was run:

```python
Best Parameters:
{
    "colsample_bytree": 0.9,
    "learning_rate":    0.05,
    "max_depth":        3,
    "n_estimators":     200,
    "reg_alpha":        0,
    "reg_lambda":       1.5,
    "subsample":        0.7
}
```

**Key insight:** A shallow `max_depth=3` with a low `learning_rate=0.05` and 200 trees outperformed deeper trees — this indicates the relationship between features and delay is relatively smooth, and deeper trees were overfitting noise.

---

## 📈 Output Plots

| Plot | Description |
|------|-------------|
| `eda_plots.png` | 6-panel EDA: delay distribution, correlation heatmap, scatter plots, quarterly box plot |
| `feature_importance.png` | Top 15 features ranked by XGBoost gain importance |
| `predictions.png` | Actual vs predicted scatter with R² annotation + residuals distribution |
| `model_comparison.png` | Bar chart comparing MAE across all 4 models |

---

## 🔮 What I Would Improve With More Time

1. **More data & external signals** — integrating team velocity history, Jira ticket data, or release calendar events could meaningfully improve predictions
2. **Time-series aware validation** — use `TimeSeriesSplit` instead of random train/test split to respect temporal ordering and prevent data leakage
3. **SHAP explanations** — add SHAP value analysis for per-prediction explainability, useful for product teams to understand *why* a feature might be delayed
4. **Deployment as an API** — wrap the saved model in a FastAPI endpoint so engineering teams can query predicted delay in real time during sprint planning
5. **Optuna instead of GridSearchCV** — Bayesian hyperparameter optimization would search the space more efficiently, especially as the param grid grows
6. **Anomaly detection** — flag unusual shipments (e.g. `delay_days > 25`) separately and model them with a specialized classifier

---

## 📦 Requirements
pandas

numpy

scikit-learn

xgboost

matplotlib

seaborn

joblib

Install with:
```bash
pip install -r requirements.txt
```

---

## 👩‍💻 Author

**Deepali**
B.Tech Computer Science & Engineering (AI & ML), University of Delhi
