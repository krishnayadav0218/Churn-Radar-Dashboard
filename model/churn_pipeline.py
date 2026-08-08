"""
Customer Churn Prediction Pipeline
-----------------------------------
1. Load data
2. Preprocess (encode categoricals, scale numerics)
3. Train Logistic Regression + Random Forest, compare
4. Evaluate (ROC-AUC, precision/recall, confusion matrix)
5. Feature importance
6. Business translation: high-risk customers & revenue at risk
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    RocCurveDisplay, precision_recall_curve
)

RANDOM_STATE = 42

# All inputs/outputs are resolved relative to this script's location, so the
# pipeline works no matter where the repo is checked out or which directory
# you run it from (as long as you `cd model` first, per the README).
MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODEL_DIR.parent

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv(MODEL_DIR / "telecom_churn.csv")

target = "churn"
id_col = "customer_id"
revenue_col = "monthly_charges"  # proxy for revenue per customer

X = df.drop(columns=[target, id_col])
y = df[target]

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------
# 2. Preprocessing pipeline
# ---------------------------------------------------------------
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_cols),
])

# ---------------------------------------------------------------
# 3. Models
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE
    ),
}

results = {}
fitted_pipelines = {}

for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", model)])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    preds = pipe.predict(X_test)
    auc = roc_auc_score(y_test, proba)
    results[name] = {
        "auc": auc,
        "report": classification_report(y_test, preds, output_dict=True),
        "proba": proba,
        "preds": preds,
    }
    fitted_pipelines[name] = pipe

    # 5-fold stratified CV on the training split — a single train/test AUC can
    # be noisy; this gives a mean +/- std that's more trustworthy for model
    # selection and reporting.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    results[name]["cv_auc_mean"] = float(cv_scores.mean())
    results[name]["cv_auc_std"] = float(cv_scores.std())

    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print(f"Held-out test ROC-AUC: {auc:.4f}")
    print(f"5-fold CV ROC-AUC:     {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))

# ---------------------------------------------------------------
# 4. Pick best model by AUC
# ---------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["auc"])
best_pipe = fitted_pipelines[best_name]
best_proba = results[best_name]["proba"]
print(f"\nBest model: {best_name} (AUC={results[best_name]['auc']:.4f})")

# ---------------------------------------------------------------
# 5. ROC curve comparison plot
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
for name in models:
    RocCurveDisplay.from_predictions(y_test, results[name]["proba"], name=name, ax=ax)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
ax.set_title("ROC Curve Comparison — Churn Models")
ax.legend()
plt.tight_layout()
plt.savefig(MODEL_DIR / "charts" / "roc_curves.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 6. Confusion matrix for best model (and all models, for the diagnostics view)
# ---------------------------------------------------------------
cm = confusion_matrix(y_test, results[best_name]["preds"])
fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
ax.set_xticks([0, 1]); ax.set_xticklabels(["No Churn", "Churn"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["No Churn", "Churn"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig(MODEL_DIR / "charts" / "confusion_matrix.png", dpi=150)
plt.close()

# Confusion matrix for every model (not just the winner) so the dashboard's
# diagnostics view can let a user flip between models.
confusion_matrices_all = {
    name: confusion_matrix(y_test, results[name]["preds"]).tolist()
    for name in models
}

# ---------------------------------------------------------------
# 7. Feature importance (Random Forest / Gradient Boosting)
# ---------------------------------------------------------------
if best_name in ("Random Forest", "Gradient Boosting"):
    feature_names = (
        numeric_cols +
        list(best_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(categorical_cols))
    )

    # Impurity-based importances (fast, but biased toward high-cardinality
    # / continuous features). We report them but also compute permutation
    # importance on the held-out test set, which measures the actual drop in
    # AUC when a feature is shuffled — a more reliable signal for one-hot
    # encoded categoricals in particular. We rank by permutation importance
    # and keep impurity importance as a secondary column.
    impurity_importances = best_pipe.named_steps["clf"].feature_importances_

    perm_result = permutation_importance(
        best_pipe, X_test, y_test, scoring="roc_auc",
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    # permutation_importance works on the raw X columns (pre-transform), not
    # the expanded one-hot feature names, so we align on the original
    # categorical/numeric column list.
    perm_df = pd.DataFrame({
        "feature": numeric_cols + categorical_cols,
        "perm_importance": perm_result.importances_mean,
    }).sort_values("perm_importance", ascending=False)

    imp_df = pd.DataFrame({"feature": feature_names, "importance": impurity_importances}).sort_values(
        "importance", ascending=False
    ).head(12)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#2563eb")
    ax.set_title(f"Top Feature Importances (impurity-based) — {best_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "charts" / "feature_importance.png", dpi=150)
    plt.close()
    imp_df.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
    perm_df.to_csv(MODEL_DIR / "permutation_importance.csv", index=False)
    print("\nTop features (impurity-based):\n", imp_df)
    print("\nTop features (permutation importance, on held-out test set):\n", perm_df.head(12))

# ---------------------------------------------------------------
# 8. BUSINESS ANALYSIS — high-risk customers & revenue at risk
# ---------------------------------------------------------------
test_df = df.loc[idx_test].copy()
test_df["churn_probability"] = best_proba
test_df["annual_revenue"] = test_df[revenue_col] * 12
test_df["churn_actual"] = y_test.values

# Risk tiers
def risk_tier(p):
    if p >= 0.7:
        return "High Risk"
    elif p >= 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"

test_df["risk_tier"] = test_df["churn_probability"].apply(risk_tier)

total_revenue = test_df["annual_revenue"].sum()
tier_summary = test_df.groupby("risk_tier").agg(
    customers=("customer_id", "count"),
    annual_revenue=("annual_revenue", "sum"),
    avg_churn_prob=("churn_probability", "mean"),
).reset_index()
tier_summary["pct_of_customers"] = 100 * tier_summary["customers"] / len(test_df)
tier_summary["pct_of_revenue"] = 100 * tier_summary["annual_revenue"] / total_revenue
tier_summary = tier_summary.sort_values("avg_churn_prob", ascending=False)

print("\n" + "=" * 60)
print("BUSINESS SUMMARY — Risk Tiers (on held-out test set)")
print("=" * 60)
print(tier_summary.to_string(index=False))

# Empirical churn rate by segment, computed on the FULL dataset (not just
# the test split) since this is describing observed behavior, not a model
# prediction — more data is strictly better here.
def segment_churn_rates(col):
    g = df.groupby(col)["churn"].agg(["mean", "count"]).reset_index()
    g.columns = [col, "churn_rate", "customers"]
    return g.sort_values("churn_rate", ascending=False).to_dict(orient="records")

segment_churn = {
    "contract": segment_churn_rates("contract"),
    "internet_service": segment_churn_rates("internet_service"),
    "payment_method": segment_churn_rates("payment_method"),
}

# Churn rate by tenure bucket — shows the classic "early tenure = highest
# risk" curve. Computed on the full dataset, same reasoning as segment_churn.
tenure_bins = [0, 6, 12, 24, 36, 48, 60, 200]
tenure_labels = ["0-6mo", "6-12mo", "12-24mo", "24-36mo", "36-48mo", "48-60mo", "60mo+"]
df["_tenure_bucket"] = pd.cut(df["tenure_months"], bins=tenure_bins, labels=tenure_labels, right=False)
tenure_cohort = df.groupby("_tenure_bucket", observed=True)["churn"].agg(["mean", "count"]).reindex(tenure_labels).reset_index()
tenure_cohort.columns = ["tenure_bucket", "churn_rate", "customers"]
tenure_cohort_export = [
    {"tenure_bucket": r.tenure_bucket, "churn_rate": round(float(r.churn_rate), 4), "customers": int(r.customers)}
    for r in tenure_cohort.itertuples() if pd.notna(r.churn_rate)
]

# Kaplan-Meier survival curve: treats tenure_months as the observed duration
# and churn==1 as the event. Every customer in this snapshot dataset is
# either censored (still active, churn==0, tenure = time observed so far) or
# has an event (churned, tenure = time-to-churn) — a standard KM setup, no
# extra dependency needed since it's a simple product-limit estimator.
def kaplan_meier(durations, events):
    order = np.argsort(durations)
    durations = np.asarray(durations)[order]
    events = np.asarray(events)[order]
    unique_times = np.unique(durations)
    n_at_risk = len(durations)
    survival = 1.0
    curve = []
    for t in unique_times:
        mask = durations == t
        d = int(events[mask].sum())          # events at time t
        n = n_at_risk                        # at risk just before t
        if n > 0 and d > 0:
            survival *= (1 - d / n)
        n_at_risk -= int(mask.sum())
        curve.append({"tenure_months": int(t), "survival_prob": round(float(survival), 4), "at_risk": int(n)})
    return curve

km_curve = kaplan_meier(df["tenure_months"].values, df["churn"].values)
# Downsample for a lighter JSON — keep every Nth point plus the last.
km_step = max(1, len(km_curve) // 80)
km_curve_export = km_curve[::km_step]
if km_curve and km_curve_export[-1] != km_curve[-1]:
    km_curve_export.append(km_curve[-1])

high_risk = test_df[test_df["risk_tier"] == "High Risk"]
high_risk_revenue = high_risk["annual_revenue"].sum()
pct_customers_high = 100 * len(high_risk) / len(test_df)
pct_revenue_high = 100 * high_risk_revenue / total_revenue

print(f"\nHigh-risk customers: {len(high_risk)} ({pct_customers_high:.1f}% of customer base)")
print(f"Revenue represented by high-risk customers: ${high_risk_revenue:,.0f} "
      f"({pct_revenue_high:.1f}% of total test-set annual revenue)")

# Save the scored customer list (top 25 highest risk, by revenue impact)
top_at_risk = high_risk.sort_values("annual_revenue", ascending=False)[
    ["customer_id", "churn_probability", "annual_revenue", "contract",
     "tenure_months", "num_support_calls", "tech_support"]
].head(25)
top_at_risk.to_csv(MODEL_DIR / "top_25_at_risk_customers.csv", index=False)

tier_summary.to_csv(MODEL_DIR / "risk_tier_summary.csv", index=False)
test_df[["customer_id", "churn_probability", "risk_tier", "annual_revenue"]].to_csv(
    MODEL_DIR / "all_scored_customers.csv", index=False
)

# Fuller per-customer export (every test-set customer, not just the top 25
# high-risk-by-revenue) for the dashboard's filterable/sortable table and the
# threshold explorer. churn_actual (0/1) lets the browser compute real
# precision/recall at any threshold without re-calling the model.
customer_directory = test_df.sort_values("annual_revenue", ascending=False)[
    ["customer_id", "churn_probability", "churn_actual", "risk_tier", "annual_revenue",
     "contract", "internet_service", "tenure_months", "num_support_calls", "tech_support",
     "monthly_charges", "total_charges", "senior_citizen", "online_security",
     "paperless_billing", "payment_method", "partner", "dependents"]
].head(300)

# Compact, FULL-coverage arrays (all 1,250 test customers, not just the 300
# above) so the threshold explorer's precision/recall/revenue-at-risk numbers
# are computed over the whole held-out set rather than a revenue-biased
# subset. Parallel arrays keep this small in JSON.
threshold_explorer_data = {
    "probability": [round(float(p), 4) for p in test_df["churn_probability"]],
    "actual": [int(a) for a in test_df["churn_actual"]],
    "annual_revenue": [round(float(r), 2) for r in test_df["annual_revenue"]],
}

# ---------------------------------------------------------------
# 9. Revenue-at-risk chart
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = {"High Risk": "#dc2626", "Medium Risk": "#f59e0b", "Low Risk": "#16a34a"}
order = ["High Risk", "Medium Risk", "Low Risk"]
ts = tier_summary.set_index("risk_tier").loc[order]

axes[0].bar(order, ts["pct_of_customers"], color=[colors[t] for t in order])
axes[0].set_title("% of Customers by Risk Tier")
axes[0].set_ylabel("% of customers")
for i, v in enumerate(ts["pct_of_customers"]):
    axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")

axes[1].bar(order, ts["pct_of_revenue"], color=[colors[t] for t in order])
axes[1].set_title("% of Annual Revenue by Risk Tier")
axes[1].set_ylabel("% of revenue")
for i, v in enumerate(ts["pct_of_revenue"]):
    axes[1].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")

plt.suptitle("Customers vs. Revenue Concentration by Churn-Risk Tier", fontweight="bold")
plt.tight_layout()
plt.savefig(MODEL_DIR / "charts" / "revenue_at_risk.png", dpi=150)
plt.close()

print("\nSaved charts to model/charts/, and top_25_at_risk_customers.csv,")
print("risk_tier_summary.csv, all_scored_customers.csv to model/.")

# Save model comparison summary for report
with open(MODEL_DIR / "model_summary.txt", "w") as f:
    f.write(f"Best model: {best_name}\n")
    f.write(f"AUC scores (held-out test): { {k: round(v['auc'], 4) for k, v in results.items()} }\n")
    cv_summary = {
        k: f"{v['cv_auc_mean']:.4f} +/- {v['cv_auc_std']:.4f}" for k, v in results.items()
    }
    f.write(f"AUC scores (5-fold CV mean +/- std): {cv_summary}\n")
    f.write(f"High risk customers: {len(high_risk)} ({pct_customers_high:.1f}% of base)\n")
    f.write(f"High risk revenue: ${high_risk_revenue:,.0f} ({pct_revenue_high:.1f}% of revenue)\n")
    f.write(f"Total test-set customers: {len(test_df)}\n")
    f.write(f"Total test-set annual revenue: ${total_revenue:,.0f}\n")

# Persist the fitted best-model pipeline so it can be reused for batch
# scoring (e.g. a monthly re-score job) without retraining from scratch.
joblib.dump(best_pipe, MODEL_DIR / "best_model_pipeline.joblib")
print(f"Saved fitted pipeline to {MODEL_DIR / 'best_model_pipeline.joblib'}")

# ---------------------------------------------------------------
# 10. EXPORT everything needed for the static web dashboard
# ---------------------------------------------------------------
import json

# 10a. ROC curve points per model (downsampled for a lightweight JSON)
roc_data = {}
pr_data = {}
for name in models:
    fpr, tpr, _ = __import__("sklearn.metrics", fromlist=["roc_curve"]).roc_curve(
        y_test, results[name]["proba"]
    )
    step = max(1, len(fpr) // 60)
    roc_data[name] = {
        "fpr": [round(float(x), 4) for x in fpr[::step]],
        "tpr": [round(float(x), 4) for x in tpr[::step]],
        "auc": round(float(results[name]["auc"]), 4),
    }

    # Precision-recall curve — more informative than ROC for an imbalanced
    # target like churn, since it doesn't reward true negatives.
    precision, recall, _ = precision_recall_curve(y_test, results[name]["proba"])
    step_pr = max(1, len(precision) // 60)
    pr_data[name] = {
        "precision": [round(float(x), 4) for x in precision[::step_pr]],
        "recall": [round(float(x), 4) for x in recall[::step_pr]],
    }

# 10b. Metrics table per model
metrics_table = []
for name in models:
    rep = results[name]["report"]
    metrics_table.append({
        "model": name,
        "auc": round(results[name]["auc"], 4),
        "cv_auc_mean": round(results[name]["cv_auc_mean"], 4),
        "cv_auc_std": round(results[name]["cv_auc_std"], 4),
        "accuracy": round(rep["accuracy"], 4),
        "churn_precision": round(rep["1"]["precision"], 4),
        "churn_recall": round(rep["1"]["recall"], 4),
        "churn_f1": round(rep["1"]["f1-score"], 4),
    })

# 10c. Confusion matrix (best model) + all models for the diagnostics tabs
cm_export = {
    "labels": ["No Churn", "Churn"],
    "matrix": cm.tolist(),
    "model": best_name,
}
confusion_matrices_export = {
    name: {"labels": ["No Churn", "Churn"], "matrix": confusion_matrices_all[name]}
    for name in models
}

# 10d. Feature importance (best model, if tree-based)
feature_importance_export = []
if best_name in ("Random Forest", "Gradient Boosting"):
    feature_importance_export = [
        {"feature": row.feature, "importance": round(float(row.importance), 4)}
        for row in imp_df.itertuples()
    ]

# 10e. Risk tier summary
risk_tier_export = [
    {
        "tier": row.risk_tier,
        "customers": int(row.customers),
        "annual_revenue": round(float(row.annual_revenue), 2),
        "avg_churn_prob": round(float(row.avg_churn_prob), 4),
        "pct_of_customers": round(float(row.pct_of_customers), 2),
        "pct_of_revenue": round(float(row.pct_of_revenue), 2),
    }
    for row in tier_summary.itertuples()
]

# 10f. Top at-risk customers (for the table)
top_at_risk_export = json.loads(
    top_at_risk.round(4).to_json(orient="records")
)

# 10f-2. Fuller customer directory (up to 300 customers, all risk tiers) for
# the filterable/sortable table, and the full-coverage arrays for the
# threshold explorer.
customer_directory_export = json.loads(
    customer_directory.round(4).to_json(orient="records")
)

# 10g. Overall headline numbers
headline = {
    "total_test_customers": int(len(test_df)),
    "total_test_annual_revenue": round(float(total_revenue), 2),
    "high_risk_customers": int(len(high_risk)),
    "high_risk_revenue": round(float(high_risk_revenue), 2),
    "pct_customers_high": round(float(pct_customers_high), 2),
    "pct_revenue_high": round(float(pct_revenue_high), 2),
    "best_model": best_name,
    "best_auc": round(float(results[best_name]["auc"]), 4),
}

# 10h. Logistic Regression coefficients -> for a fully client-side (JS) churn calculator
lr_pipe = fitted_pipelines["Logistic Regression"]
lr_model = lr_pipe.named_steps["clf"]
scaler = lr_pipe.named_steps["prep"].named_transformers_["num"]
cat_encoder = lr_pipe.named_steps["prep"].named_transformers_["cat"]

lr_coefs = lr_model.coef_[0]
lr_intercept = float(lr_model.intercept_[0])

numeric_export = [
    {
        "name": col,
        "mean": round(float(scaler.mean_[i]), 4),
        "scale": round(float(scaler.scale_[i]), 4),
        "coef": round(float(lr_coefs[i]), 4),
    }
    for i, col in enumerate(numeric_cols)
]

categories_list = cat_encoder.categories_
drop_idx = cat_encoder.drop_idx_

categorical_export = []
ptr = len(numeric_cols)
for i, col in enumerate(categorical_cols):
    cats = categories_list[i]
    d_idx = drop_idx[i] if drop_idx is not None else None
    base_cat = str(cats[d_idx]) if d_idx is not None else None
    coefs_map = {}
    for j, c in enumerate(cats):
        if d_idx is not None and j == d_idx:
            continue
        coefs_map[str(c)] = round(float(lr_coefs[ptr]), 4)
        ptr += 1
    categorical_export.append({"feature": col, "base": base_cat, "coefs": coefs_map})

calculator_model = {
    "intercept": round(lr_intercept, 4),
    "numeric": numeric_export,
    "categorical": categorical_export,
    "note": "Standalone logistic-regression export (portable, runs fully client-side). "
            "The full analysis/report uses the Random Forest model.",
}

dashboard_data = {
    "headline": headline,
    "metrics_table": metrics_table,
    "roc_data": roc_data,
    "pr_data": pr_data,
    "confusion_matrix": cm_export,
    "confusion_matrices": confusion_matrices_export,
    "feature_importance": feature_importance_export,
    "risk_tiers": risk_tier_export,
    "top_at_risk": top_at_risk_export,
    "customer_directory": customer_directory_export,
    "threshold_explorer": threshold_explorer_data,
    "segment_churn": segment_churn,
    "tenure_cohort": tenure_cohort_export,
    "survival_curve": km_curve_export,
    "calculator_model": calculator_model,
}

# Written straight to the repo root as data.json — the exact filename the
# dashboard fetches — so there's no manual "rename and copy" step. Re-run
# this script and refresh the page (or redeploy) to see the update.
out_path = REPO_ROOT / "data.json"
with open(out_path, "w") as f:
    json.dump(dashboard_data, f, indent=2)

print(f"\nExported {out_path} for the web dashboard.")
