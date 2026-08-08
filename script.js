/* ChurnRadar dashboard logic — no build step, no backend. */

let DATA = null;
let custSort = { key: "annual_revenue", dir: "desc" };
let custQuery = "";

document.addEventListener("DOMContentLoaded", () => {
  setupTheme();
  setupMobileNav();
  loadData();
});

async function loadData() {
  try {
    const res = await fetch("data.json", { cache: "no-store" });
    if (!res.ok) throw new Error(`data.json responded with ${res.status}`);
    DATA = await res.json();
  } catch (err) {
    console.error("Failed to load dashboard data:", err);
    hideSkeleton();
    showDataError();
    return;
  }

  renderHero();
  renderModelCards();
  setupDiagnostics();
  renderImportance();
  renderSegments();
  renderTiers();
  renderCohortChart();
  renderSurvivalChart();
  setupThresholdExplorer();
  setupRoiCalculator();
  setupCustomerTableControls();
  renderCustomerTable();
  renderFooter();
  setupCalculator();
  setupCompareMode();
  setupBatchScoring();
  setupExplainModal();
  setupScrollSpy();
  setupKeyboardShortcuts();
  hideSkeleton();
  setupReveal();
}

function hideSkeleton() {
  document.body.classList.remove("is-loading");
  const skel = document.getElementById("skeletonWrap");
  if (skel) skel.remove();
}

function showDataError() {
  const main = document.querySelector("main");
  const banner = document.createElement("div");
  banner.className = "data-error";
  banner.setAttribute("role", "alert");
  banner.innerHTML = `
    <strong>Couldn't load dashboard data.</strong>
    <span>data.json failed to load — if you're running this locally, serve the folder
    over HTTP (e.g. <code>python3 -m http.server</code>) rather than opening index.html
    directly, since browsers block fetch() on the file:// protocol.</span>
  `;
  main.prepend(banner);
}

function fmtMoney(n) {
  return Math.round(n).toLocaleString("en-US");
}

/* Escape any string before interpolating into innerHTML. Data currently
   comes from our own trusted data.json, but the pipeline is designed to be
   re-pointed at arbitrary customer exports (see README), so treat every
   field as untrusted rather than assuming that will always hold. */
function esc(str) {
  const div = document.createElement("div");
  div.textContent = String(str ?? "");
  return div.innerHTML;
}

function showToast(message) {
  const container = document.getElementById("toastContainer");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  container.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add("show"));
  setTimeout(() => {
    toast.classList.remove("show");
    setTimeout(() => toast.remove(), 300);
  }, 2600);
}

/* ---------------- HERO ---------------- */


function renderHero() {
  const h = DATA.headline;
  document.getElementById("heroPctCustomers").textContent = h.pct_customers_high.toFixed(1) + "%";
  document.getElementById("heroPctRevenue").textContent = h.pct_revenue_high.toFixed(1) + "%";
  document.getElementById("heroRevenueAmt").textContent = fmtMoney(h.high_risk_revenue);
  document.getElementById("heroTotalCustomers").textContent = h.total_test_customers.toLocaleString("en-US");
  const badge = document.getElementById("heroBadgeText");
  if (badge) badge.textContent = `MODEL LIVE — scikit-learn ${h.best_model}`;
}

/* ---------------- MODEL CARDS ---------------- */
function renderModelCards() {
  const wrap = document.getElementById("modelCards");
  const best = DATA.headline.best_model;
  wrap.innerHTML = DATA.metrics_table.map(m => `
    <div class="model-card reveal ${m.model === best ? "best" : ""}">
      ${m.model === best ? '<span class="model-card-best-badge">Best model</span>' : ""}
      <h3>${esc(m.model)}</h3>
      <div class="model-auc">${m.auc.toFixed(3)} <span style="font-size:0.9rem;color:var(--muted);font-weight:400;">AUC</span></div>
      ${cvNote(m)}
      ${metricBar("Churn recall", m.churn_recall)}
      ${metricBar("Churn precision", m.churn_precision)}
      ${metricBar("Accuracy", m.accuracy)}
    </div>
  `).join("");
}
function cvNote(m) {
  if (typeof m.cv_auc_mean !== "number") return "";
  return `<div class="model-cv-note">5-fold CV: ${m.cv_auc_mean.toFixed(3)} &plusmn; ${m.cv_auc_std.toFixed(3)}</div>`;
}
function metricBar(label, val) {
  return `
    <div class="model-metric-row"><span>${label}</span><span>${(val * 100).toFixed(0)}%</span></div>
    <div class="model-bar-track"><div class="model-bar-fill" style="width:${val * 100}%"></div></div>
  `;
}

/* ---------------- MODEL DIAGNOSTICS (ROC curve + confusion matrix, per model) ---------------- */
let diagActiveModel = null;

function setupDiagnostics() {
  if (!DATA.confusion_matrices || !DATA.roc_data) return;
  diagActiveModel = DATA.headline.best_model;

  const tabsWrap = document.getElementById("diagTabs");
  const modelNames = Object.keys(DATA.roc_data);
  tabsWrap.innerHTML = modelNames.map(name => `
    <button class="diag-tab ${name === diagActiveModel ? "active" : ""}" type="button" role="tab"
      aria-selected="${name === diagActiveModel}" data-model="${esc(name)}">${esc(name)}</button>
  `).join("");

  tabsWrap.querySelectorAll(".diag-tab").forEach(btn => {
    btn.addEventListener("click", () => {
      diagActiveModel = btn.dataset.model;
      tabsWrap.querySelectorAll(".diag-tab").forEach(b => {
        b.classList.toggle("active", b === btn);
        b.setAttribute("aria-selected", b === btn ? "true" : "false");
      });
      renderDiagnostics();
    });
  });

  renderDiagnostics();
  renderModelScatter();
}

function renderDiagnostics() {
  renderRocChart(diagActiveModel);
  renderPrChart(diagActiveModel);
  renderConfusionChart(diagActiveModel);
}

function renderRocChart(modelName) {
  const wrap = document.getElementById("rocChart");
  const d = DATA.roc_data[modelName];
  if (!d) { wrap.innerHTML = ""; return; }

  const W = 320, H = 320, PAD = 34;
  const plotW = W - PAD * 2, plotH = H - PAD * 2;
  const toX = (fpr) => PAD + fpr * plotW;
  const toY = (tpr) => PAD + (1 - tpr) * plotH;

  const points = d.fpr.map((f, i) => `${toX(f).toFixed(1)},${toY(d.tpr[i]).toFixed(1)}`).join(" ");
  const diagonal = `${toX(0)},${toY(0)} ${toX(1)},${toY(1)}`;

  wrap.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="ROC curve for ${esc(modelName)}, AUC ${d.auc.toFixed(3)}">
      <line x1="${PAD}" y1="${PAD}" x2="${PAD}" y2="${H - PAD}" stroke="var(--line)" stroke-width="1"/>
      <line x1="${PAD}" y1="${H - PAD}" x2="${W - PAD}" y2="${H - PAD}" stroke="var(--line)" stroke-width="1"/>
      <polyline points="${diagonal}" fill="none" stroke="var(--line)" stroke-width="1.5" stroke-dasharray="4,4"/>
      <polyline points="${points}" fill="none" stroke="var(--deep)" stroke-width="2.5"/>
      <text x="${PAD}" y="${H - PAD + 16}" font-size="9">0</text>
      <text x="${W - PAD - 6}" y="${H - PAD + 16}" font-size="9">1</text>
      <text x="${PAD - 22}" y="${H - PAD + 3}" font-size="9">0</text>
      <text x="${PAD - 22}" y="${PAD + 3}" font-size="9">1</text>
      <text x="${W / 2 - 22}" y="${H - 6}" font-size="9">False Positive Rate</text>
      <text x="8" y="${H / 2}" font-size="9" transform="rotate(-90 8 ${H / 2})">True Positive Rate</text>
    </svg>
    <div class="diag-auc-label">AUC = ${d.auc.toFixed(3)}</div>
  `;
}

function renderPrChart(modelName) {
  const wrap = document.getElementById("prChart");
  const d = DATA.pr_data && DATA.pr_data[modelName];
  if (!d) { wrap.innerHTML = ""; return; }

  const W = 320, H = 320, PAD = 34;
  const plotW = W - PAD * 2, plotH = H - PAD * 2;
  const toX = (recall) => PAD + recall * plotW;
  const toY = (precision) => PAD + (1 - precision) * plotH;

  const points = d.recall.map((r, i) => `${toX(r).toFixed(1)},${toY(d.precision[i]).toFixed(1)}`).join(" ");

  // Baseline = the fraction of actual churners in the test set — what a
  // random classifier would score on precision, regardless of recall.
  let baseline = 0.3;
  if (DATA.threshold_explorer && DATA.threshold_explorer.actual.length) {
    const a = DATA.threshold_explorer.actual;
    baseline = a.reduce((s, v) => s + v, 0) / a.length;
  }
  const baselineY = toY(baseline).toFixed(1);

  wrap.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Precision-recall curve for ${esc(modelName)}">
      <line x1="${PAD}" y1="${PAD}" x2="${PAD}" y2="${H - PAD}" stroke="var(--line)" stroke-width="1"/>
      <line x1="${PAD}" y1="${H - PAD}" x2="${W - PAD}" y2="${H - PAD}" stroke="var(--line)" stroke-width="1"/>
      <line x1="${PAD}" y1="${baselineY}" x2="${W - PAD}" y2="${baselineY}" stroke="var(--line)" stroke-width="1.5" stroke-dasharray="4,4"/>
      <polyline points="${points}" fill="none" stroke="var(--red)" stroke-width="2.5"/>
      <text x="${PAD}" y="${H - PAD + 16}" font-size="9">0</text>
      <text x="${W - PAD - 6}" y="${H - PAD + 16}" font-size="9">1</text>
      <text x="${PAD - 22}" y="${H - PAD + 3}" font-size="9">0</text>
      <text x="${PAD - 22}" y="${PAD + 3}" font-size="9">1</text>
      <text x="${W / 2 - 16}" y="${H - 6}" font-size="9">Recall</text>
      <text x="8" y="${H / 2}" font-size="9" transform="rotate(-90 8 ${H / 2})">Precision</text>
    </svg>
    <div class="diag-auc-label">Baseline (random) = ${(baseline * 100).toFixed(0)}% precision</div>
  `;
}

function renderConfusionChart(modelName) {
  const wrap = document.getElementById("confusionChart");
  const cm = DATA.confusion_matrices[modelName];
  if (!cm) { wrap.innerHTML = ""; return; }

  const [[tn, fp], [fn, tp]] = cm.matrix;
  const max = Math.max(tn, fp, fn, tp);
  const cellColor = (v) => {
    const alpha = 0.15 + 0.65 * (v / max);
    return `rgba(27,75,67,${alpha.toFixed(2)})`;
  };

  const cells = [
    { label: "Predicted No Churn\n(Actual No Churn)", v: tn, x: 0, y: 0 },
    { label: "Predicted Churn\n(Actual No Churn)", v: fp, x: 1, y: 0 },
    { label: "Predicted No Churn\n(Actual Churn)", v: fn, x: 0, y: 1 },
    { label: "Predicted Churn\n(Actual Churn)", v: tp, x: 1, y: 1 },
  ];

  const cellSize = 130, gap = 6;
  const svgCells = cells.map(c => `
    <rect x="${c.x * (cellSize + gap)}" y="${c.y * (cellSize + gap)}" width="${cellSize}" height="${cellSize}"
      rx="8" fill="${cellColor(c.v)}" stroke="var(--line)" />
    <text x="${c.x * (cellSize + gap) + cellSize / 2}" y="${c.y * (cellSize + gap) + cellSize / 2 - 4}"
      font-size="26" font-weight="700" fill="var(--ink)" text-anchor="middle" font-family="var(--font-mono)">${c.v}</text>
  `).join("");

  wrap.innerHTML = `
    <svg viewBox="0 0 ${cellSize * 2 + gap} ${cellSize * 2 + gap}" role="img" aria-label="Confusion matrix for ${esc(modelName)}">
      ${svgCells}
    </svg>
    <div class="cm-legend">
      <span>◤ True Negative: ${tn}</span> · <span>◥ False Positive: ${fp}</span><br>
      <span>◣ False Negative: ${fn}</span> · <span>◢ True Positive: ${tp}</span>
    </div>
  `;
}

function renderModelScatter() {
  const wrap = document.getElementById("modelScatter");
  if (!wrap) return;
  const models = DATA.metrics_table;
  const best = DATA.headline.best_model;

  const W = 340, H = 300, PAD = 40;
  const plotW = W - PAD * 2, plotH = H - PAD * 2;
  const toX = (recall) => PAD + recall * plotW;
  const toY = (precision) => PAD + (1 - precision) * plotH;

  const colors = ["#1B4B43", "#B45309", "#5B3A8E"];
  const dots = models.map((m, i) => `
    <circle cx="${toX(m.churn_recall).toFixed(1)}" cy="${toY(m.churn_precision).toFixed(1)}" r="${m.model === best ? 8 : 6}"
      fill="${colors[i % colors.length]}" stroke="${m.model === best ? "var(--ink)" : "none"}" stroke-width="2"/>
    <text x="${toX(m.churn_recall).toFixed(1)}" y="${(toY(m.churn_precision) - 14).toFixed(1)}"
      font-size="10" text-anchor="middle" font-family="var(--font-mono)" fill="var(--ink)">${esc(m.model)}</text>
  `).join("");

  wrap.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Precision vs recall scatter for all models">
      <line x1="${PAD}" y1="${PAD}" x2="${PAD}" y2="${H - PAD}" stroke="var(--line)" stroke-width="1"/>
      <line x1="${PAD}" y1="${H - PAD}" x2="${W - PAD}" y2="${H - PAD}" stroke="var(--line)" stroke-width="1"/>
      <text x="${PAD}" y="${H - PAD + 16}" font-size="9" fill="var(--muted)">0</text>
      <text x="${W - PAD - 6}" y="${H - PAD + 16}" font-size="9" fill="var(--muted)">1</text>
      <text x="${PAD - 22}" y="${H - PAD + 3}" font-size="9" fill="var(--muted)">0</text>
      <text x="${PAD - 22}" y="${PAD + 3}" font-size="9" fill="var(--muted)">1</text>
      <text x="${W / 2 - 16}" y="${H - 6}" font-size="9" fill="var(--muted)">Recall</text>
      <text x="8" y="${H / 2}" font-size="9" fill="var(--muted)" transform="rotate(-90 8 ${H / 2})">Precision</text>
      ${dots}
    </svg>
  `;
}

/* ---------------- FEATURE IMPORTANCE ---------------- */
function renderImportance() {
  const wrap = document.getElementById("importanceList");
  const items = DATA.feature_importance;
  if (!items || !items.length) {
    wrap.innerHTML = `<p class="empty-note">No feature-importance data available for this model.</p>`;
    return;
  }
  const max = Math.max(...items.map(i => i.importance));
  wrap.innerHTML = items.map(i => `
    <div class="imp-row reveal">
      <span class="imp-name">${esc(niceFeatureName(i.feature))}</span>
      <div class="imp-track"><div class="imp-fill" data-width="${(i.importance / max) * 100}"></div></div>
      <span class="imp-val">${(i.importance * 100).toFixed(1)}%</span>
    </div>
  `).join("");
}
function niceFeatureName(f) {
  return f.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

/* ---------------- SEGMENT COMPARISON ---------------- */
function renderSegments() {
  const wrap = document.getElementById("segmentGrid");
  if (!wrap || !DATA.segment_churn) return;

  const titles = {
    contract: "By contract type",
    internet_service: "By internet service",
    payment_method: "By payment method",
  };

  wrap.innerHTML = Object.entries(DATA.segment_churn).map(([key, rows]) => {
    const max = Math.max(...rows.map(r => r.churn_rate));
    return `
      <div class="segment-card reveal">
        <h4>${esc(titles[key] || niceFeatureName(key))}</h4>
        ${rows.map(r => `
          <div class="segment-row">
            <div class="segment-row-label">
              <span>${esc(r[key])} <span style="opacity:0.6">(${r.customers})</span></span>
              <b>${(r.churn_rate * 100).toFixed(1)}%</b>
            </div>
            <div class="segment-track"><div class="segment-fill" data-width="${(r.churn_rate / max) * 100}"></div></div>
          </div>
        `).join("")}
      </div>
    `;
  }).join("");
}

/* ---------------- TENURE COHORT + SURVIVAL CURVE ---------------- */
function renderCohortChart() {
  const wrap = document.getElementById("cohortChart");
  const rows = DATA.tenure_cohort;
  if (!wrap || !rows || !rows.length) return;

  const W = 340, H = 300, PAD = 40;
  const plotW = W - PAD * 2, plotH = H - PAD * 2;
  const maxRate = Math.max(...rows.map(r => r.churn_rate));
  const stepX = plotW / (rows.length - 1 || 1);
  const toX = (i) => PAD + i * stepX;
  const toY = (rate) => PAD + (1 - rate / (maxRate * 1.15)) * plotH;

  const points = rows.map((r, i) => `${toX(i).toFixed(1)},${toY(r.churn_rate).toFixed(1)}`).join(" ");
  const dots = rows.map((r, i) => `
    <circle cx="${toX(i).toFixed(1)}" cy="${toY(r.churn_rate).toFixed(1)}" r="4" fill="var(--red)"/>
    <text x="${toX(i).toFixed(1)}" y="${(toY(r.churn_rate) - 10).toFixed(1)}" font-size="9" text-anchor="middle" font-family="var(--font-mono)" fill="var(--ink)">${(r.churn_rate * 100).toFixed(0)}%</text>
    <text x="${toX(i).toFixed(1)}" y="${H - PAD + 16}" font-size="8" text-anchor="middle" fill="var(--muted)">${esc(r.tenure_bucket)}</text>
  `).join("");

  wrap.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Churn rate by tenure cohort">
      <line x1="${PAD}" y1="${PAD}" x2="${PAD}" y2="${H - PAD}" stroke="var(--line)" stroke-width="1"/>
      <line x1="${PAD}" y1="${H - PAD}" x2="${W - PAD}" y2="${H - PAD}" stroke="var(--line)" stroke-width="1"/>
      <polyline points="${points}" fill="none" stroke="var(--red)" stroke-width="2"/>
      ${dots}
    </svg>
  `;
}

function renderSurvivalChart() {
  const wrap = document.getElementById("survivalChart");
  const rows = DATA.survival_curve;
  if (!wrap || !rows || !rows.length) return;

  const W = 340, H = 300, PAD = 40;
  const plotW = W - PAD * 2, plotH = H - PAD * 2;
  const maxT = rows[rows.length - 1].tenure_months;
  const toX = (t) => PAD + (t / maxT) * plotW;
  const toY = (s) => PAD + (1 - s) * plotH;

  const points = rows.map(r => `${toX(r.tenure_months).toFixed(1)},${toY(r.survival_prob).toFixed(1)}`).join(" ");
  const lastRow = rows[rows.length - 1];

  wrap.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Kaplan-Meier survival curve by tenure">
      <line x1="${PAD}" y1="${PAD}" x2="${PAD}" y2="${H - PAD}" stroke="var(--line)" stroke-width="1"/>
      <line x1="${PAD}" y1="${H - PAD}" x2="${W - PAD}" y2="${H - PAD}" stroke="var(--line)" stroke-width="1"/>
      <polyline points="${points}" fill="none" stroke="var(--deep)" stroke-width="2.5"/>
      <text x="${PAD}" y="${H - PAD + 16}" font-size="9" fill="var(--muted)">0mo</text>
      <text x="${W - PAD - 16}" y="${H - PAD + 16}" font-size="9" fill="var(--muted)">${maxT}mo</text>
      <text x="${PAD - 24}" y="${H - PAD + 3}" font-size="9" fill="var(--muted)">0%</text>
      <text x="${PAD - 30}" y="${PAD + 3}" font-size="9" fill="var(--muted)">100%</text>
    </svg>
    <div class="diag-auc-label">${(lastRow.survival_prob * 100).toFixed(0)}% still active at ${maxT} months</div>
  `;
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
        <h4>${esc(t.tier)}</h4>
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
      <td><span class="tier-pill"><span class="tier-dot" style="background:${TIER_COLOR[t.tier]}"></span>${esc(t.tier)}</span></td>
      <td>${t.customers}</td>
      <td>${t.pct_of_customers.toFixed(1)}%</td>
      <td>$${fmtMoney(t.annual_revenue)}</td>
      <td>${t.pct_of_revenue.toFixed(1)}%</td>
      <td>${(t.avg_churn_prob * 100).toFixed(0)}%</td>
    </tr>
  `).join("");
}

/* ---------------- THRESHOLD EXPLORER ---------------- */
let lastThresholdStats = null;

function setupThresholdExplorer() {
  const slider = document.getElementById("thresholdSlider");
  const valEl = document.getElementById("thresholdVal");
  const statsEl = document.getElementById("thresholdStats");
  if (!slider || !DATA.threshold_explorer) return;

  const { probability, actual, annual_revenue } = DATA.threshold_explorer;
  const totalRevenue = annual_revenue.reduce((a, b) => a + b, 0);
  const totalActualChurners = actual.reduce((a, b) => a + b, 0);

  function update() {
    const cutoff = Number(slider.value) / 100;
    valEl.textContent = slider.value + "%";

    let tp = 0, fp = 0, fn = 0, tn = 0, flaggedRevenue = 0;
    for (let i = 0; i < probability.length; i++) {
      const predicted = probability[i] >= cutoff;
      const isChurner = actual[i] === 1;
      if (predicted && isChurner) tp++;
      else if (predicted && !isChurner) fp++;
      else if (!predicted && isChurner) fn++;
      else tn++;
      if (predicted) flaggedRevenue += annual_revenue[i];
    }

    const flagged = tp + fp;
    const precision = flagged > 0 ? tp / flagged : 0;
    const recall = totalActualChurners > 0 ? tp / totalActualChurners : 0;
    const f1 = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    const pctRevenueFlagged = totalRevenue > 0 ? (flaggedRevenue / totalRevenue) * 100 : 0;
    const pctCustomersFlagged = (flagged / probability.length) * 100;

    lastThresholdStats = { tp, fp, fn, tn, flagged, flaggedRevenue, precision, recall, f1 };

    statsEl.innerHTML = `
      <div class="threshold-stat"><div class="threshold-stat-val">${flagged}</div><div class="threshold-stat-label">Customers flagged (${pctCustomersFlagged.toFixed(1)}%)</div></div>
      <div class="threshold-stat"><div class="threshold-stat-val">${(precision * 100).toFixed(0)}%</div><div class="threshold-stat-label">Precision</div></div>
      <div class="threshold-stat"><div class="threshold-stat-val">${(recall * 100).toFixed(0)}%</div><div class="threshold-stat-label">Recall</div></div>
      <div class="threshold-stat"><div class="threshold-stat-val">${(f1 * 100).toFixed(0)}%</div><div class="threshold-stat-label">F1 score</div></div>
      <div class="threshold-stat"><div class="threshold-stat-val">$${fmtMoney(flaggedRevenue)}</div><div class="threshold-stat-label">Revenue flagged (${pctRevenueFlagged.toFixed(1)}%)</div></div>
    `;

    updateRoi();
  }

  slider.addEventListener("input", update);
  update();
}

/* ---------------- RETENTION ROI ---------------- */
function setupRoiCalculator() {
  const costSlider = document.getElementById("roiCost");
  const successSlider = document.getElementById("roiSuccess");
  if (!costSlider || !successSlider) return;

  costSlider.addEventListener("input", updateRoi);
  successSlider.addEventListener("input", updateRoi);
  updateRoi();
}

function updateRoi() {
  const costSlider = document.getElementById("roiCost");
  const successSlider = document.getElementById("roiSuccess");
  const statsEl = document.getElementById("roiStats");
  if (!costSlider || !statsEl || !lastThresholdStats) return;

  document.getElementById("roiCostVal").textContent = costSlider.value;
  document.getElementById("roiSuccessVal").textContent = successSlider.value;

  const cost = Number(costSlider.value);
  const successRate = Number(successSlider.value) / 100;
  const { flagged, tp, flaggedRevenue } = lastThresholdStats;

  // Approximation: assume revenue is spread evenly across flagged customers,
  // so "true positive share of flagged revenue" scales with tp/flagged.
  // Only true positives (actual churners who were correctly flagged) can
  // actually be "saved" by an offer — calling a false positive costs money
  // with no churn to prevent.
  const avgRevenuePerFlagged = flagged > 0 ? flaggedRevenue / flagged : 0;
  const revenueAtStakeAmongTp = avgRevenuePerFlagged * tp;
  const totalCampaignCost = cost * flagged;
  const customersSaved = tp * successRate;
  const revenueSaved = revenueAtStakeAmongTp * successRate;
  const netValue = revenueSaved - totalCampaignCost;

  const statsEl2 = statsEl;
  statsEl2.innerHTML = `
    <div class="roi-stat"><div class="roi-stat-val">${flagged}</div><div class="roi-stat-label">Offers sent</div></div>
    <div class="roi-stat"><div class="roi-stat-val">$${fmtMoney(totalCampaignCost)}</div><div class="roi-stat-label">Campaign cost</div></div>
    <div class="roi-stat"><div class="roi-stat-val">${customersSaved.toFixed(1)}</div><div class="roi-stat-label">Expected customers retained</div></div>
    <div class="roi-stat"><div class="roi-stat-val">$${fmtMoney(revenueSaved)}</div><div class="roi-stat-label">Expected revenue saved</div></div>
    <div class="roi-stat"><div class="roi-stat-val ${netValue < 0 ? "negative" : ""}">${netValue < 0 ? "-" : ""}$${fmtMoney(Math.abs(netValue))}</div><div class="roi-stat-label">Net value</div></div>
  `;
}

/* ---------------- CUSTOMER TABLE (search + filter + sort + CSV export + print) ---------------- */
let custTierFilter = "All";
let custContractFilter = "All";

function customerSource() {
  // Fall back to the smaller top_at_risk list if an older data.json (before
  // the customer_directory export existed) is being served.
  return DATA.customer_directory && DATA.customer_directory.length ? DATA.customer_directory : DATA.top_at_risk;
}

function setupCustomerTableControls() {
  const searchInput = document.getElementById("custSearch");
  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      custQuery = e.target.value.trim().toLowerCase();
      renderCustomerTable();
    });
  }

  setupFilterChips("tierChips", ["All", "High Risk", "Medium Risk", "Low Risk"], (val) => {
    custTierFilter = val;
    renderCustomerTable();
  });

  const contracts = ["All", ...new Set(customerSource().map(c => c.contract))];
  setupFilterChips("contractChips", contracts, (val) => {
    custContractFilter = val;
    renderCustomerTable();
  });

  document.querySelectorAll("#custTable thead th[data-sort-key]").forEach(th => {
    const activate = () => {
      const key = th.dataset.sortKey;
      if (custSort.key === key) {
        custSort.dir = custSort.dir === "asc" ? "desc" : "asc";
      } else {
        custSort = { key, dir: "desc" };
      }
      renderCustomerTable();
    };
    th.addEventListener("click", activate);
    th.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        activate();
      }
    });
  });

  const exportBtn = document.getElementById("custExportBtn");
  if (exportBtn) exportBtn.addEventListener("click", exportCustomerCsv);

  const printBtn = document.getElementById("custPrintBtn");
  if (printBtn) printBtn.addEventListener("click", () => window.print());
}

function setupFilterChips(containerId, values, onChange) {
  const wrap = document.getElementById(containerId);
  if (!wrap) return;
  wrap.innerHTML = values.map((v, i) => `
    <button class="filter-chip ${i === 0 ? "active" : ""}" type="button" data-val="${esc(v)}">${esc(v)}</button>
  `).join("");
  wrap.querySelectorAll(".filter-chip").forEach(btn => {
    btn.addEventListener("click", () => {
      wrap.querySelectorAll(".filter-chip").forEach(b => b.classList.toggle("active", b === btn));
      onChange(btn.dataset.val);
    });
  });
}

function getFilteredSortedCustomers() {
  let rows = customerSource();

  if (custTierFilter !== "All") {
    rows = rows.filter(c => c.risk_tier === custTierFilter);
  }
  if (custContractFilter !== "All") {
    rows = rows.filter(c => c.contract === custContractFilter);
  }
  if (custQuery) {
    rows = rows.filter(c =>
      String(c.customer_id).toLowerCase().includes(custQuery) ||
      String(c.contract).toLowerCase().includes(custQuery)
    );
  }
  const { key, dir } = custSort;
  const sign = dir === "asc" ? 1 : -1;
  rows = [...rows].sort((a, b) => {
    const av = a[key], bv = b[key];
    if (typeof av === "string") return sign * av.localeCompare(bv);
    return sign * (av - bv);
  });
  return rows;
}

const TIER_TEXT_COLOR = { "High Risk": "var(--red)", "Medium Risk": "var(--amber)", "Low Risk": "var(--green)" };

function renderCustomerTable() {
  const tbody = document.querySelector("#custTable tbody");
  const rows = getFilteredSortedCustomers();
  const total = customerSource().length;

  updateSortIndicators();

  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-note">No customers match the current search/filters.</td></tr>`;
    const countEl = document.getElementById("custCount");
    if (countEl) countEl.textContent = `0 of ${total} customers`;
    return;
  }

  tbody.innerHTML = rows.map(c => `
    <tr data-cust-id="${esc(c.customer_id)}">
      <td>${esc(c.customer_id)}</td>
      <td style="color:${TIER_TEXT_COLOR[c.risk_tier] || "var(--ink)"};font-weight:600;">${(c.churn_probability * 100).toFixed(0)}%</td>
      <td>${esc(c.risk_tier)}</td>
      <td>$${fmtMoney(c.annual_revenue)}</td>
      <td>${esc(c.contract)}</td>
      <td>${c.tenure_months} mo</td>
      <td>${c.num_support_calls}</td>
    </tr>
  `).join("");

  const countEl = document.getElementById("custCount");
  if (countEl) countEl.textContent = `${rows.length} of ${total} customers`;
}

function updateSortIndicators() {
  document.querySelectorAll("#custTable thead th[data-sort-key]").forEach(th => {
    th.classList.toggle("sorted", th.dataset.sortKey === custSort.key);
    th.setAttribute("aria-sort",
      th.dataset.sortKey === custSort.key
        ? (custSort.dir === "asc" ? "ascending" : "descending")
        : "none"
    );
    const arrow = th.querySelector(".sort-arrow");
    if (arrow) arrow.textContent = th.dataset.sortKey === custSort.key ? (custSort.dir === "asc" ? "\u2191" : "\u2193") : "";
  });
}

function exportCustomerCsv() {
  const rows = getFilteredSortedCustomers();
  const headers = ["customer_id", "churn_probability", "risk_tier", "annual_revenue", "contract", "tenure_months", "num_support_calls"];
  const csvLines = [headers.join(",")];
  rows.forEach(c => {
    csvLines.push(headers.map(h => {
      const v = c[h];
      return typeof v === "string" && v.includes(",") ? `"${v}"` : v;
    }).join(","));
  });
  const blob = new Blob([csvLines.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "churnradar_at_risk_customers.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  showToast(`Exported ${rows.length} customers to CSV.`);
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

function riskTierFor(p) {
  if (p >= 0.7) return { tier: "HIGH RISK", label: "High Risk", colorClass: "on-red" };
  if (p >= 0.4) return { tier: "MEDIUM RISK", label: "Medium Risk", colorClass: "on-amber" };
  return { tier: "LOW RISK", label: "Low Risk", colorClass: "on-green" };
}

let lastCalcResult = null;

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
  const readout = document.getElementById("signalMeter")?.closest(".calc-readout");

  function totalCharges() {
    return Number(monthly.value) * (Number(tenure.value) + 1);
  }

  function update() {
    tenureVal.textContent = tenure.value;
    monthlyVal.textContent = monthly.value;
    callsVal.textContent = calls.value;
    tenure.setAttribute("aria-valuetext", `${tenure.value} months`);
    monthly.setAttribute("aria-valuetext", `$${monthly.value}`);
    calls.setAttribute("aria-valuetext", `${calls.value} calls`);

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

    // A single formula now drives both the tier label and the number of lit
    // bars, so they can never disagree (previously the tier thresholds at
    // 0.7/0.4 and the bar-count formula were computed independently and
    // could show, e.g., a "MEDIUM RISK" label with all 5 bars lit).
    const { tier, colorClass } = riskTierFor(p);
    const litBars = Math.max(1, Math.min(5, Math.round(p * 5) + 1));

    lastCalcResult = { probability: p, tier, inputs: { ...inputs } };

    tierEl.textContent = tier;
    if (readout) {
      readout.setAttribute("aria-label", `Predicted churn probability ${pct}%, ${tier.toLowerCase()}`);
    }
    bars.forEach((bar, i) => {
      bar.classList.remove("on-red", "on-amber", "on-green");
      if (i < litBars) bar.classList.add(colorClass);
    });
  }

  [tenure, monthly, calls, contract, internet, techSupport, onlineSecurity, paperless, senior, payment]
    .forEach(el => el.addEventListener("input", update));

  // Prefill from URL query params if present (e.g. ?tenure=5&monthly=90&contract=Month-to-month),
  // so a "copy link" can be shared to reproduce an exact calculator scenario.
  const params = new URLSearchParams(window.location.search);
  if (params.has("tenure")) tenure.value = clampNum(params.get("tenure"), 0, 72, tenure.value);
  if (params.has("monthly")) monthly.value = clampNum(params.get("monthly"), 18, 120, monthly.value);
  if (params.has("calls")) calls.value = clampNum(params.get("calls"), 0, 10, calls.value);
  if (params.has("contract") && [...contract.options].some(o => o.value === params.get("contract"))) contract.value = params.get("contract");
  if (params.has("internet") && [...internet.options].some(o => o.value === params.get("internet"))) internet.value = params.get("internet");
  if (params.has("payment") && [...payment.options].some(o => o.value === params.get("payment"))) payment.value = params.get("payment");
  if (params.has("techSupport")) techSupport.checked = params.get("techSupport") === "1";
  if (params.has("onlineSecurity")) onlineSecurity.checked = params.get("onlineSecurity") === "1";
  if (params.has("paperless")) paperless.checked = params.get("paperless") === "1";
  if (params.has("senior")) senior.checked = params.get("senior") === "1";

  update();

  const copyBtn = document.getElementById("calcCopyLink");
  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      const url = new URL(window.location.href);
      url.search = new URLSearchParams({
        tenure: tenure.value,
        monthly: monthly.value,
        calls: calls.value,
        contract: contract.value,
        internet: internet.value,
        payment: payment.value,
        techSupport: techSupport.checked ? "1" : "0",
        onlineSecurity: onlineSecurity.checked ? "1" : "0",
        paperless: paperless.checked ? "1" : "0",
        senior: senior.checked ? "1" : "0",
      }).toString();

      try {
        await navigator.clipboard.writeText(url.toString());
      } catch {
        // Clipboard API can fail (permissions, insecure context) — fall back
        // to selecting the URL isn't possible without an input, so just
        // surface the link in a prompt as a last resort.
        window.prompt("Copy this link:", url.toString());
      }
      copyBtn.textContent = "✓ Link copied";
      copyBtn.classList.add("copied");
      setTimeout(() => {
        copyBtn.textContent = "🔗 Copy link to this profile";
        copyBtn.classList.remove("copied");
      }, 2000);
    });
  }
}

function clampNum(raw, min, max, fallback) {
  const n = Number(raw);
  if (Number.isNaN(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

/* ---------------- CALCULATOR: COMPARE MODE (Profile A vs Profile B) ---------------- */
let compareProfiles = { A: null, B: null };

function setupCompareMode() {
  const saveA = document.getElementById("compareSaveA");
  const saveB = document.getElementById("compareSaveB");
  const clearBtn = document.getElementById("compareClear");
  if (!saveA || !saveB) return;

  saveA.addEventListener("click", () => {
    if (!lastCalcResult) return;
    compareProfiles.A = { ...lastCalcResult };
    renderComparePanel();
    showToast("Saved current settings as Profile A.");
  });
  saveB.addEventListener("click", () => {
    if (!lastCalcResult) return;
    compareProfiles.B = { ...lastCalcResult };
    renderComparePanel();
    showToast("Saved current settings as Profile B.");
  });
  clearBtn.addEventListener("click", () => {
    compareProfiles = { A: null, B: null };
    renderComparePanel();
  });
}

function renderComparePanel() {
  const panel = document.getElementById("comparePanel");
  if (!panel) return;
  const { A, B } = compareProfiles;

  if (!A && !B) { panel.innerHTML = ""; return; }

  const cards = [];
  if (A) cards.push(compareCardHtml("Profile A", A));
  if (B) cards.push(compareCardHtml("Profile B", B));

  let deltaCard = "";
  if (A && B) {
    const delta = (B.probability - A.probability) * 100;
    const sign = delta > 0 ? "+" : "";
    deltaCard = `
      <div class="compare-delta">
        <div class="compare-delta-val">${sign}${delta.toFixed(1)} pts</div>
        <div class="compare-delta-label">A → B change</div>
      </div>
    `;
  }

  panel.innerHTML = cards.join("") + deltaCard;
}

function compareCardHtml(label, result) {
  return `
    <div class="compare-card">
      <h5>${esc(label)}</h5>
      <div class="compare-prob">${Math.round(result.probability * 100)}%</div>
      <div class="compare-tier">${esc(result.tier)}</div>
      <div class="compare-note">${esc(result.inputs.contract)} · ${result.inputs.tenure_months}mo tenure · $${result.inputs.monthly_charges}/mo</div>
    </div>
  `;
}

/* ---------------- EXPLAIN-CUSTOMER MODAL ---------------- */
function setupExplainModal() {
  const overlay = document.getElementById("explainOverlay");
  const closeBtn = document.getElementById("explainClose");
  if (!overlay) return;

  const tbody = document.querySelector("#custTable tbody");
  tbody.addEventListener("click", (e) => {
    const tr = e.target.closest("tr");
    if (!tr || !tr.dataset.custId) return;
    openExplainModal(tr.dataset.custId);
  });

  const close = () => { overlay.hidden = true; };
  closeBtn.addEventListener("click", close);
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !overlay.hidden) close();
  });
}

function openExplainModal(customerId) {
  const overlay = document.getElementById("explainOverlay");
  const body = document.getElementById("explainBody");
  const title = document.getElementById("explainTitle");

  const cust = customerSource().find(c => String(c.customer_id) === String(customerId));
  if (!cust) return;

  title.textContent = customerId;

  const contributions = explainCustomer(cust);
  const maxAbs = Math.max(...contributions.map(c => Math.abs(c.contribution)), 0.001);

  body.innerHTML = `
    <div class="explain-summary">
      Predicted churn probability: <b>${(cust.churn_probability * 100).toFixed(0)}%</b> (${esc(cust.risk_tier)})
      &nbsp;·&nbsp; ${esc(cust.contract)}, ${cust.tenure_months}mo tenure, $${cust.monthly_charges}/mo
    </div>
    <div class="explain-summary" style="margin-top:-12px;">Top factors pushing this score up (red) or down (green), from the
      logistic-regression model:</div>
    ${contributions.slice(0, 8).map(c => `
      <div class="explain-factor">
        <span class="explain-factor-name">${esc(c.label)}</span>
        <div class="explain-factor-track">
          <div class="explain-factor-fill ${c.contribution >= 0 ? "up" : "down"}"
            style="width:${(Math.abs(c.contribution) / maxAbs / 2 * 100).toFixed(1)}%"></div>
        </div>
        <span class="explain-factor-val">${c.contribution >= 0 ? "+" : ""}${c.contribution.toFixed(2)}</span>
      </div>
    `).join("")}
  `;

  overlay.hidden = false;
}

/* Decompose a customer's logistic-regression logit into per-feature
   contributions, using the same exported coefficients as the calculator.
   Each numeric feature contributes (scaled_value * coef); each categorical
   feature contributes its category's coefficient (0 if it's the baseline). */
function explainCustomer(cust) {
  const m = DATA.calculator_model;
  const contributions = [];

  m.numeric.forEach(f => {
    const raw = cust[f.name];
    if (raw === undefined) return;
    const scaled = (raw - f.mean) / f.scale;
    contributions.push({ label: niceFeatureName(f.name), contribution: scaled * f.coef });
  });

  m.categorical.forEach(f => {
    const chosen = cust[f.feature];
    if (chosen === undefined) return;
    const coef = (chosen !== f.base && f.coefs.hasOwnProperty(chosen)) ? f.coefs[chosen] : 0;
    contributions.push({ label: `${niceFeatureName(f.feature)}: ${chosen}`, contribution: coef });
  });

  return contributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
}

/* ---------------- SCROLL SPY ---------------- */
function setupScrollSpy() {
  const navLinks = Array.from(document.querySelectorAll(".nav-links a[data-nav]"));
  if (!navLinks.length) return;
  const sections = navLinks
    .map(a => document.getElementById(a.dataset.nav))
    .filter(Boolean);

  const io = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      const link = navLinks.find(a => a.dataset.nav === entry.target.id);
      if (!link) return;
      if (entry.isIntersecting) {
        navLinks.forEach(a => a.classList.remove("active"));
        link.classList.add("active");
      }
    });
  }, { rootMargin: "-40% 0px -55% 0px", threshold: 0 });

  sections.forEach(s => io.observe(s));
}

/* ---------------- KEYBOARD SHORTCUTS (customer directory) ---------------- */
function setupKeyboardShortcuts() {
  document.addEventListener("keydown", (e) => {
    const tag = (e.target.tagName || "").toLowerCase();
    const isTyping = tag === "input" || tag === "textarea" || tag === "select" || e.target.isContentEditable;

    if (e.key === "/" && !isTyping) {
      e.preventDefault();
      const search = document.getElementById("custSearch");
      if (search) {
        if (typeof search.scrollIntoView === "function") {
          search.scrollIntoView({ behavior: "smooth", block: "center" });
        }
        search.focus();
      }
    }

    if (e.key === "Escape" && isTyping && e.target.id === "custSearch") {
      e.target.value = "";
      custQuery = "";
      renderCustomerTable();
      e.target.blur();
    }
  });
}

/* ---------------- BATCH SCORING (client-side CSV upload + scoring) ---------------- */
function getCalculatorDefaults() {
  const m = DATA.calculator_model;
  const defaults = {};
  m.numeric.forEach(f => { defaults[f.name] = f.mean; });
  m.categorical.forEach(f => { defaults[f.feature] = f.base; });
  return defaults;
}

function parseCsv(text) {
  const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
  if (!lines.length) return { headers: [], rows: [] };
  const splitLine = (line) => line.split(",").map(cell => cell.trim().replace(/^"|"$/g, ""));
  const headers = splitLine(lines[0]);
  const rows = lines.slice(1).map(line => {
    const cells = splitLine(line);
    const obj = {};
    headers.forEach((h, i) => { obj[h] = cells[i]; });
    return obj;
  });
  return { headers, rows };
}

function scoreCsvRow(row, defaults) {
  const num = (key) => {
    const v = row[key];
    const n = Number(v);
    return v !== undefined && v !== "" && !Number.isNaN(n) ? n : defaults[key];
  };
  const cat = (key) => {
    const v = row[key];
    return v !== undefined && v !== "" ? v : defaults[key];
  };

  const tenure_months = num("tenure_months");
  const monthly_charges = num("monthly_charges");

  const inputs = {
    tenure_months,
    monthly_charges,
    total_charges: row.total_charges !== undefined && row.total_charges !== "" ? num("total_charges") : monthly_charges * (tenure_months + 1),
    num_support_calls: num("num_support_calls"),
    senior_citizen: row.senior_citizen !== undefined && row.senior_citizen !== "" ? num("senior_citizen") : defaults.senior_citizen,
    contract: cat("contract"),
    internet_service: cat("internet_service"),
    tech_support: cat("tech_support"),
    online_security: cat("online_security"),
    paperless_billing: cat("paperless_billing"),
    payment_method: cat("payment_method"),
    partner: cat("partner"),
    dependents: cat("dependents"),
  };

  return predictChurn(inputs);
}

function setupBatchScoring() {
  const fileInput = document.getElementById("batchFile");
  const dropZone = document.getElementById("batchDrop");
  const dropText = document.getElementById("batchDropText");
  const statusEl = document.getElementById("batchStatus");
  const previewEl = document.getElementById("batchPreview");
  if (!fileInput || !dropZone) return;

  const REQUIRED_HINT = "tenure_months, monthly_charges, num_support_calls, contract, internet_service, tech_support, online_security, paperless_billing, payment_method, senior_citizen";

  function handleFile(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) {
      statusEl.textContent = "Please choose a .csv file.";
      statusEl.className = "batch-status error";
      return;
    }
    statusEl.textContent = `Reading ${file.name}…`;
    statusEl.className = "batch-status";
    dropText.textContent = file.name;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const { headers, rows } = parseCsv(String(e.target.result));
        if (!rows.length) throw new Error("No data rows found.");
        if (!headers.includes("customer_id")) {
          statusEl.textContent = `Warning: no "customer_id" column found — rows will be numbered instead. Recognized columns: ${REQUIRED_HINT}`;
        }

        const defaults = getCalculatorDefaults();
        const scored = rows.map((row, i) => {
          const p = scoreCsvRow(row, defaults);
          return {
            customer_id: row.customer_id || `ROW-${i + 1}`,
            churn_probability: Math.round(p * 1000) / 1000,
            risk_tier: riskTierFor(p).label,
          };
        });

        renderBatchPreview(scored, rows.length);
        statusEl.textContent = `Scored ${scored.length} customers. Missing columns fell back to dataset averages/baseline categories.`;
        statusEl.className = "batch-status ok";
        setupBatchDownload(scored);
      } catch (err) {
        console.error(err);
        statusEl.textContent = `Couldn't parse that CSV: ${err.message}`;
        statusEl.className = "batch-status error";
        previewEl.innerHTML = "";
      }
    };
    reader.onerror = () => {
      statusEl.textContent = "Couldn't read that file.";
      statusEl.className = "batch-status error";
    };
    reader.readAsText(file);
  }

  fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

  ["dragover", "dragenter"].forEach(evt => {
    dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });
  });
  ["dragleave", "dragend", "drop"].forEach(evt => {
    dropZone.addEventListener(evt, () => dropZone.classList.remove("dragover"));
  });
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) handleFile(file);
  });
}

function renderBatchPreview(scored, totalCount) {
  const previewEl = document.getElementById("batchPreview");
  const preview = scored.slice(0, 15);
  previewEl.innerHTML = `
    <table>
      <thead><tr><th>Customer ID</th><th>Churn Probability</th><th>Risk Tier</th></tr></thead>
      <tbody>
        ${preview.map(r => `<tr><td>${esc(r.customer_id)}</td><td>${(r.churn_probability * 100).toFixed(1)}%</td><td>${esc(r.risk_tier)}</td></tr>`).join("")}
      </tbody>
    </table>
    ${totalCount > preview.length ? `<p class="empty-note" style="text-align:left !important;padding:10px 4px !important;">…and ${totalCount - preview.length} more. Download the full CSV below.</p>` : ""}
  `;
}

function setupBatchDownload(scored) {
  let btn = document.getElementById("batchDownloadBtn");
  if (!btn) {
    btn = document.createElement("button");
    btn.id = "batchDownloadBtn";
    btn.className = "btn-export";
    btn.type = "button";
    btn.style.marginTop = "14px";
    document.getElementById("batchPreview").after(btn);
  }
  btn.textContent = "⭳ Download scored CSV";
  btn.onclick = () => {
    const headers = ["customer_id", "churn_probability", "risk_tier"];
    const lines = [headers.join(",")];
    scored.forEach(r => lines.push(headers.map(h => r[h]).join(",")));
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "churnradar_batch_scored.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    showToast(`Downloaded ${scored.length} scored customers.`);
  };
}

/* ---------------- THEME (light/dark, persisted) ---------------- */
function setupTheme() {
  const toggle = document.getElementById("themeToggle");
  const stored = localStorage.getItem("churnradar-theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const theme = stored || (prefersDark ? "dark" : "light");
  applyTheme(theme);

  if (toggle) {
    toggle.addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      applyTheme(next);
      localStorage.setItem("churnradar-theme", next);
    });
  }
}
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const toggle = document.getElementById("themeToggle");
  if (toggle) {
    toggle.setAttribute("aria-pressed", theme === "dark" ? "true" : "false");
    toggle.textContent = theme === "dark" ? "\u2600\ufe0f Light" : "\ud83c\udf19 Dark";
  }
}

/* ---------------- MOBILE NAV ---------------- */
function setupMobileNav() {
  const toggle = document.getElementById("navToggle");
  const links = document.getElementById("navLinks");
  if (!toggle || !links) return;

  toggle.addEventListener("click", () => {
    const open = links.classList.toggle("open");
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
  });

  links.querySelectorAll("a").forEach(a => {
    a.addEventListener("click", () => {
      links.classList.remove("open");
      toggle.setAttribute("aria-expanded", "false");
    });
  });
}

/* ---------------- SCROLL REVEAL ---------------- */
function setupReveal() {
  const prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const items = document.querySelectorAll(".reveal");
  const io = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add("in");
        io.unobserve(e.target);
      }
    });
  }, { threshold: 0.15 });

  if (prefersReduced) {
    items.forEach(i => i.classList.add("in"));
  } else {
    items.forEach(i => io.observe(i));
  }

  // Animate width-based bars (importance/tier bars) independently of the
  // fade-in reveal above, since some live outside a .reveal wrapper.
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
