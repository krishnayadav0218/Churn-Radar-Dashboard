"""
Generates a realistic synthetic telecom customer-churn dataset.
(No real dataset was uploaded, so we simulate one with realistic
correlations: tenure, contract type, support calls, and charges
all drive churn probability, mimicking the well-known
Telco Customer Churn dataset structure.)
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 5000

tenure = np.random.exponential(scale=24, size=n).clip(0, 72).astype(int)
contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20]
)
internet_service = np.random.choice(
    ["Fiber optic", "DSL", "No"], size=n, p=[0.45, 0.35, 0.20]
)
monthly_charges = np.round(
    np.random.normal(65, 25, size=n).clip(18, 120), 2
)
tech_support = np.random.choice(["Yes", "No"], size=n, p=[0.4, 0.6])
online_security = np.random.choice(["Yes", "No"], size=n, p=[0.35, 0.65])
paperless_billing = np.random.choice(["Yes", "No"], size=n, p=[0.6, 0.4])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    size=n, p=[0.35, 0.2, 0.25, 0.2]
)
num_support_calls = np.random.poisson(1.5, size=n).clip(0, 10)
senior_citizen = np.random.choice([0, 1], size=n, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], size=n, p=[0.48, 0.52])
dependents = np.random.choice(["Yes", "No"], size=n, p=[0.3, 0.7])

total_charges = np.round(monthly_charges * (tenure + 1) * np.random.uniform(0.9, 1.05, n), 2)

# ---- Build churn probability from a realistic logit combination ----
logit = (
    -3.0
    + (-0.045 * tenure)
    + np.where(contract == "Month-to-month", 1.1, 0)
    + np.where(contract == "One year", 0.1, 0)
    + np.where(contract == "Two year", -0.9, 0)
    + np.where(internet_service == "Fiber optic", 0.55, 0)
    + (0.012 * monthly_charges)
    + np.where(tech_support == "No", 0.4, -0.1)
    + np.where(online_security == "No", 0.35, -0.1)
    + np.where(paperless_billing == "Yes", 0.25, 0)
    + np.where(payment_method == "Electronic check", 0.45, 0)
    + (0.28 * num_support_calls)
    + np.where(senior_citizen == 1, 0.3, 0)
    + np.where(partner == "No", 0.15, 0)
    + np.where(dependents == "No", 0.1, 0)
    + np.random.normal(0, 0.5, n)  # noise
)
prob = 1 / (1 + np.exp(-logit))
churn = (np.random.uniform(0, 1, n) < prob).astype(int)

df = pd.DataFrame({
    "customer_id": [f"CUST-{i:05d}" for i in range(n)],
    "tenure_months": tenure,
    "contract": contract,
    "internet_service": internet_service,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "tech_support": tech_support,
    "online_security": online_security,
    "paperless_billing": paperless_billing,
    "payment_method": payment_method,
    "num_support_calls": num_support_calls,
    "senior_citizen": senior_citizen,
    "partner": partner,
    "dependents": dependents,
    "churn": churn,
})

df.to_csv("/home/claude/telecom_churn.csv", index=False)
print(df.shape)
print(df["churn"].value_counts(normalize=True))
print(df.head())
