/* ChurnRadar dashboard logic — no build step, no backend. */

let DATA = null;

async function loadData() {
  const res = await fetch("data.json");
  DATA = await res.json();
  renderHero();
  renderModelCards();
  renderImportance();
  renderTiers();
  renderCustomerTable();
  renderFooter();
  setupCalculator();
  setupReveal();
}

function fmtMoney(n) {
  return Math.round(n).toLocaleString("en-US");
}

/* ---------------- HERO ---------------- */
function renderHero() {
  const h = DATA.headline;
  document.getElementById("heroPctCustomers").textContent = h.pct_customers_high.toFixed(1) + "%";
  document.getElementById("heroPctRevenue").textContent = h.pct_revenue_high.toFixed(1) + "%";
  document.getElementById("heroRevenueAmt").textContent = fmtMoney(h.high_risk_revenue);
  document.getElementById("heroTotalCustomers").textContent = h.total_test_customers.toLocaleString("en-US");
}

/* ---------------- MODEL CARDS ---------------- */
function renderModelCards() {
  const wrap = document.getElementById("modelCards");
  const best = DATA.headline.best_model;
  wrap.innerHTML = DATA.metrics_table.map(m => `
    <div class="model-card reveal ${m.model === best ? "best" : ""}">
      ${m.model === best ? '<span class="model-card-best-badge">Best model</span>' : ""}
      <h3>${m.model}</h3>
      <div class="model-auc">${m.auc.toFixed(3)} <span style="font-size:0.9rem;color:var(--muted);font-weight:400;">AUC</span></div>
      ${metricBar("Churn recall", m.churn_recall)}
      ${metricBar("Churn precision", m.churn_precision)}
      ${metricBar("Accuracy", m.accuracy)}
    </div>
  `).join("");
}
function metricBar(label, val) {
  return `
    <div class="model-metric-row"><span>${label}</span><span>${(val * 100).toFixed(0)}%</span></div>
    <div class="model-bar-track"><div class="model-bar-fill" style="width:${val * 100}%"></div></div>
  `;
}

/* ---------------- FEATURE IMPORTANCE ---------------- */
function renderImportance() {
  const wrap = document.getElementById("importanceList");
  const items = DATA.feature_importance;
  const max = Math.max(...items.map(i => i.importance));
  wrap.innerHTML = items.map(i => `
    <div class="imp-row reveal">
      <span class="imp-name">${niceFeatureName(i.feature)}</span>
      <div class="imp-track"><div class="imp-fill" data-width="${(i.importance / max) * 100}"></div></div>
      <span class="imp-val">${(i.importance * 100).toFixed(1)}%</span>
    </div>
  `).join("");
}
function niceFeatureName(f) {
  return f.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

/* ---------------- RISK TIERS ---------------- */
const TIER_COLOR = { "High Risk": "var(--red)", "Medium Risk": "var(--amber)", "Low Risk": "var(--green)" };
const TIER_ORDER = ["High Risk", "Medium Risk", "Low Risk"];

function renderTiers() {
  const tiers = [...DATA.risk_tiers].sort((a, b) => TIER_ORDER.indexOf(a.tier) - TIER_ORDER.indexOf(b.tier));

  const compareWrap = document.getElementById("tierCompare");
  compareWrap.innerHTML = tiers.map(t => `
    <div class="tier-card reveal">
      <div class="tier-card-head">
        <span class="tier-dot" style="background:${TIER_COLOR[t.tier]}"></span>
        <h4>${t.tier}</h4>
      </div>
      <div class="tier-bar-label"><span>% of customers</span><span>${t.pct_of_customers.toFixed(1)}%</span></div>
      <div class="tier-bar-track"><div class="tier-bar-fill" style="background:${TIER_COLOR[t.tier]}" data-width="${t.pct_of_customers}"></div></div>
      <div class="tier-bar-label"><span>% of revenue</span><span>${t.pct_of_revenue.toFixed(1)}%</span></div>
      <div class="tier-bar-track"><div class="tier-bar-fill" style="background:${TIER_COLOR[t.tier]}" data-width="${t.pct_of_revenue}"></div></div>
    </div>
  `).join("");

  const tbody = document.querySelector("#tierTable tbody");
  tbody.innerHTML = tiers.map(t => `
    <tr>
      <td><span class="tier-pill"><span class="tier-dot" style="background:${TIER_COLOR[t.tier]}"></span>${t.tier}</span></td>
      <td>${t.customers}</td>
      <td>${t.pct_of_customers.toFixed(1)}%</td>
      <td>$${fmtMoney(t.annual_revenue)}</td>
      <td>${t.pct_of_revenue.toFixed(1)}%</td>
      <td>${(t.avg_churn_prob * 100).toFixed(0)}%</td>
    </tr>
  `).join("");
}

/* ---------------- CUSTOMER TABLE ---------------- */
function renderCustomerTable() {
  const tbody = document.querySelector("#custTable tbody");
  tbody.innerHTML = DATA.top_at_risk.map(c => `
    <tr>
      <td>${c.customer_id}</td>
      <td style="color:var(--red);font-weight:600;">${(c.churn_probability * 100).toFixed(0)}%</td>
      <td>$${fmtMoney(c.annual_revenue)}</td>
      <td>${c.contract}</td>
      <td>${c.tenure_months} mo</td>
      <td>${c.num_support_calls}</td>
    </tr>
  `).join("");
}

/* ---------------- FOOTER ---------------- */
function renderFooter() {
  document.getElementById("footerModel").textContent = "Best model: " + DATA.headline.best_model;
  document.getElementById("footerAuc").textContent = "AUC: " + DATA.headline.best_auc.toFixed(3);
}

/* ---------------- CALCULATOR (client-side logistic regression) ---------------- */
function predictChurn(inputs) {
  const m = DATA.calculator_model;
  let z = m.intercept;

  m.numeric.forEach(f => {
    const raw = inputs[f.name];
    const scaled = (raw - f.mean) / f.scale;
    z += scaled * f.coef;
  });

  m.categorical.forEach(f => {
    const chosen = inputs[f.feature];
    if (chosen !== f.base && f.coefs.hasOwnProperty(chosen)) {
      z += f.coefs[chosen];
    }
  });

  return 1 / (1 + Math.exp(-z));
}

function setupCalculator() {
  const tenure = document.getElementById("tenure");
  const monthly = document.getElementById("monthly");
  const calls = document.getElementById("calls");
  const contract = document.getElementById("contract");
  const internet = document.getElementById("internet");
  const techSupport = document.getElementById("techSupport");
  const onlineSecurity = document.getElementById("onlineSecurity");
  const paperless = document.getElementById("paperless");
  const senior = document.getElementById("senior");
  const payment = document.getElementById("payment");

  const tenureVal = document.getElementById("tenureVal");
  const monthlyVal = document.getElementById("monthlyVal");
  const callsVal = document.getElementById("callsVal");

  const bars = Array.from(document.querySelectorAll(".signal-bar"));
  const probEl = document.getElementById("calcProb");
  const tierEl = document.getElementById("calcTier");

  function totalCharges() {
    return Number(monthly.value) * (Number(tenure.value) + 1);
  }

  function update() {
    tenureVal.textContent = tenure.value;
    monthlyVal.textContent = monthly.value;
    callsVal.textContent = calls.value;

    const inputs = {
      tenure_months: Number(tenure.value),
      monthly_charges: Number(monthly.value),
      total_charges: totalCharges(),
      num_support_calls: Number(calls.value),
      senior_citizen: senior.checked ? 1 : 0,
      contract: contract.value,
      internet_service: internet.value,
      tech_support: techSupport.checked ? "Yes" : "No",
      online_security: onlineSecurity.checked ? "Yes" : "No",
      paperless_billing: paperless.checked ? "Yes" : "No",
      payment_method: payment.value,
      partner: "No",
      dependents: "No",
    };

    const p = predictChurn(inputs);
    const pct = Math.round(p * 100);
    probEl.textContent = pct + "%";

    let tier, colorClass, litBars;
    if (p >= 0.7) { tier = "HIGH RISK"; colorClass = "on-red"; litBars = 5; }
    else if (p >= 0.4) { tier = "MEDIUM RISK"; colorClass = "on-amber"; litBars = 3; }
    else { tier = "LOW RISK"; colorClass = "on-green"; litBars = 1; }
    litBars = Math.max(1, Math.min(5, Math.round(p * 5) + 1));

    tierEl.textContent = tier;
    bars.forEach((bar, i) => {
      bar.classList.remove("on-red", "on-amber", "on-green");
      if (i < litBars) bar.classList.add(colorClass);
    });
  }

  [tenure, monthly, calls, contract, internet, techSupport, onlineSecurity, paperless, senior, payment]
    .forEach(el => el.addEventListener("input", update));

  update();
}

/* ---------------- SCROLL REVEAL ---------------- */
function setupReveal() {
  const items = document.querySelectorAll(".reveal");
  const io = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add("in");
        const fills = e.target.querySelectorAll("[data-width]");
        fills.forEach(f => { f.style.width = f.dataset.width + "%"; });
        io.unobserve(e.target);
      }
    });
  }, { threshold: 0.15 });
  items.forEach(i => io.observe(i));

  // also animate any data-width bars not wrapped individually (importance/tier bars)
  const bars = document.querySelectorAll("[data-width]");
  const io2 = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.style.width = e.target.dataset.width + "%";
        io2.unobserve(e.target);
      }
    });
  }, { threshold: 0.2 });
  bars.forEach(b => io2.observe(b));
}

document.addEventListener("DOMContentLoaded", loadData);
