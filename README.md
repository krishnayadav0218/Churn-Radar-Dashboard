# ChurnRadar — Customer Churn Prediction Dashboard

A live, interactive customer-churn dashboard: model comparison, feature importance,
revenue-at-risk breakdown, a client-side risk calculator, and a prioritized
retention call list — built from a scikit-learn pipeline.

**No backend.** The dashboard is a static site (`index.html` + `style.css` + `script.js` +
`data.json`). The churn-risk calculator runs entirely in the browser using exported
logistic-regression coefficients — no API calls, no server, nothing to keep alive.

## What's in this repo

```
.
├── index.html          # dashboard markup
├── style.css            # design system (all styling)
├── script.js             # rendering + client-side prediction logic
├── data.json             # model results & coefficients, exported from the Python pipeline
├── vercel.json            # static-site config
└── model/                  # the actual ML work (reference / reproducibility)
    ├── generate_data.py      # synthetic dataset generator (swap for your real data loader)
    ├── churn_pipeline.py       # full scikit-learn pipeline: preprocess → train → evaluate → export
    ├── telecom_churn.csv         # dataset used
    ├── risk_tier_summary.csv       # risk-tier business summary
    ├── top_25_at_risk_customers.csv  # retention call list
    └── charts/                         # ROC curve, confusion matrix, feature importance, revenue chart
```

## Deploy to Vercel (2 minutes)

1. **Push this folder to a new GitHub repo:**
   ```bash
   git init
   git add .
   git commit -m "ChurnRadar dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```

2. **Import into Vercel:**
   - Go to [vercel.com/new](https://vercel.com/new)
   - Select your GitHub repo
   - Framework preset: **Other** (it's a static site — no build step needed)
   - Root directory: leave as `.`
   - Click **Deploy**

That's it — Vercel serves `index.html` directly. There's no build command, no
environment variables, and no serverless functions to configure.

### Alternative: Vercel CLI
```bash
npm i -g vercel
vercel --prod
```
Run this from the repo root and follow the prompts.

## Using your own data

The dashboard reads everything from `data.json`. To rebuild it from your real
customer data:

1. Replace `model/telecom_churn.csv` with your export (needs a binary `churn`
   column, a customer ID, and a revenue/charges column).
2. In `model/churn_pipeline.py`, update `revenue_col` if your revenue field has
   a different name, and adjust `categorical_cols`/`numeric_cols` detection if needed.
3. Run it:
   ```bash
   cd model
   pip install -r requirements.txt
   python churn_pipeline.py
   ```
4. That's it — the script writes straight to `../data.json` (the repo root),
   which is the exact file the dashboard fetches. Just redeploy
   (`git push` — Vercel auto-redeploys on push) or refresh if running locally.

The pipeline also reports 5-fold cross-validated AUC (not just a single
train/test split) and both impurity-based and permutation feature importance
— the latter is computed by shuffling each feature on the held-out test set
and measuring the AUC drop, which is more reliable for one-hot encoded
categoricals than the default scikit-learn importances. It also saves the
fitted best-model pipeline to `model/best_model_pipeline.joblib` so you can
reuse it for batch scoring without retraining.

## Local preview

No build tools needed — just serve the folder:
```bash
python3 -m http.server 8080
# open http://localhost:8080
```

## Model summary

| Model | ROC-AUC | Churn Recall | Churn Precision |
|---|---|---|---|
| Logistic Regression | 0.785 | 75% | 50% |
| **Random Forest (best)** | **0.786** | 72% | 51% |
| Gradient Boosting | 0.779 | 39% | 61% |

High-risk customers (≥70% predicted churn probability) are ~9.2% of the customer
base but represent ~10.7% of annual revenue — the business case for prioritizing
retention spend on this segment. Full write-up in `model/churn_pipeline.py` output
and the dashboard itself.

## Dashboard features

- **Model diagnostics** — flip between the three trained models to compare live ROC curves and confusion matrices.
- **Segment comparison** — empirical (not model-predicted) churn rate by contract, internet service, and payment method.
- **Threshold explorer** — drag the churn-probability cutoff and watch precision, recall, F1, and revenue coverage recompute live against the full 1,250-customer held-out test set.
- **Live calculator** with a **"copy link to this profile"** button that encodes every input into the URL, so a specific what-if scenario can be shared or bookmarked.
- **Batch CSV scoring** — upload a CSV of customers and score them entirely in the browser using the same exported logistic-regression coefficients as the calculator. Nothing is uploaded anywhere; missing columns fall back to dataset averages/baseline categories.
- **Filterable customer directory** — search, filter by risk tier and contract type, sort any column, export to CSV, or print a clean call-list view.
- **Dark mode** toggle (persisted, and respects OS preference on first visit).
- **Mobile nav** with a hamburger menu below 860px.
- Friendly on-page error message if `data.json` fails to load (most commonly
  from opening `index.html` directly via `file://` instead of serving it —
  see **Local preview** below).
- Fully keyboard-accessible: sortable table headers, filter chips, calculator sliders, and
  the theme toggle all work without a mouse.
