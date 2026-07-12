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
   pip install scikit-learn pandas numpy matplotlib
   python churn_pipeline.py
   ```
4. Copy the newly generated `dashboard_data.json` over the root `data.json`
   and redeploy (`git push` — Vercel auto-redeploys on push).

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
