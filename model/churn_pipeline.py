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
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    RocCurveDisplay, precision_recall_curve
)

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv("/home/claude/telecom_churn.csv")

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
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print(f"ROC-AUC: {auc:.4f}")
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
plt.savefig("/home/claude/roc_curves.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 6. Confusion matrix for best model
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
plt.savefig("/home/claude/confusion_matrix.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 7. Feature importance (Random Forest / Gradient Boosting)
# ---------------------------------------------------------------
if best_name in ("Random Forest", "Gradient Boosting"):
    feature_names = (
        numeric_cols +
        list(best_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(categorical_cols))
    )
    importances = best_pipe.named_steps["clf"].feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False
    ).head(12)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#2563eb")
    ax.set_title(f"Top Feature Importances — {best_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("/home/claude/feature_importance.png", dpi=150)
    plt.close()
    imp_df.to_csv("/home/claude/feature_importance.csv", index=False)
    print("\nTop features:\n", imp_df)

# ---------------------------------------------------------------
# 8. BUSINESS ANALYSIS — high-risk customers & revenue at risk
# ---------------------------------------------------------------
test_df = df.loc[idx_test].copy()
test_df["churn_probability"] = best_proba
test_df["annual_revenue"] = test_df[revenue_col] * 12

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
top_at_risk.to_csv("/home/claude/top_25_at_risk_customers.csv", index=False)

tier_summary.to_csv("/home/claude/risk_tier_summary.csv", index=False)
test_df[["customer_id", "churn_probability", "risk_tier", "annual_revenue"]].to_csv(
    "/home/claude/all_scored_customers.csv", index=False
)

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
plt.savefig("/home/claude/revenue_at_risk.png", dpi=150)
plt.close()

print("\nSaved: roc_curves.png, confusion_matrix.png, feature_importance.png,")
print("revenue_at_risk.png, top_25_at_risk_customers.csv, risk_tier_summary.csv, all_scored_customers.csv")

# Save model comparison summary for report
with open("/home/claude/model_summary.txt", "w") as f:
    f.write(f"Best model: {best_name}\n")
    f.write(f"AUC scores: { {k: round(v['auc'],4) for k,v in results.items()} }\n")
    f.write(f"High risk customers: {len(high_risk)} ({pct_customers_high:.1f}% of base)\n")
    f.write(f"High risk revenue: ${high_risk_revenue:,.0f} ({pct_revenue_high:.1f}% of revenue)\n")
    f.write(f"Total test-set customers: {len(test_df)}\n")
    f.write(f"Total test-set annual revenue: ${total_revenue:,.0f}\n")

# ---------------------------------------------------------------
# 10. EXPORT everything needed for the static web dashboard
# ---------------------------------------------------------------
import json

# 10a. ROC curve points per model (downsampled for a lightweight JSON)
roc_data = {}
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

# 10b. Metrics table per model
metrics_table = []
for name in models:
    rep = results[name]["report"]
    metrics_table.append({
        "model": name,
        "auc": round(results[name]["auc"], 4),
        "accuracy": round(rep["accuracy"], 4),
        "churn_precision": round(rep["1"]["precision"], 4),
        "churn_recall": round(rep["1"]["recall"], 4),
        "churn_f1": round(rep["1"]["f1-score"], 4),
    })

# 10c. Confusion matrix (best model)
cm_export = {
    "labels": ["No Churn", "Churn"],
    "matrix": cm.tolist(),
    "model": best_name,
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
    "confusion_matrix": cm_export,
    "feature_importance": feature_importance_export,
    "risk_tiers": risk_tier_export,
    "top_at_risk": top_at_risk_export,
    "calculator_model": calculator_model,
}

with open("/home/claude/dashboard_data.json", "w") as f:
    json.dump(dashboard_data, f, indent=2)

print("\nExported dashboard_data.json for the web dashboard.")
