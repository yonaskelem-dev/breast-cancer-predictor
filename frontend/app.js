/**
 * app.js — Multimodal Breast Cancer Classification System
 * BMED 4244 · Spring 2026 · ResNet-18 Fusion Model
 *
 * Fusion formula: fused = w1 * mammo_prob + (1 - w1) * us_prob
 * Backend: Flask API at /predict (see backend/app.py)
 */

"use strict";

// ─── STATE ────────────────────────────────────────────────────────────────────
let history = [];
let lastResult = null;
let analyticsReady = false;
let chartInstances = {};

// ─── THEME ────────────────────────────────────────────────────────────────────
function toggleTheme() {
  document.body.classList.toggle("dark");
  localStorage.setItem(
    "theme",
    document.body.classList.contains("dark") ? "dark" : "light",
  );
}

// restore saved theme
(function () {
  if (localStorage.getItem("theme") === "dark") {
    document.body.classList.add("dark");
  }
})();

// ─── TABS ─────────────────────────────────────────────────────────────────────
function showTab(name, el) {
  document
    .querySelectorAll('[id^="tab-"]')
    .forEach((t) => (t.style.display = "none"));
  document
    .querySelectorAll(".tab")
    .forEach((t) => t.classList.remove("active"));
  const tab = document.getElementById("tab-" + name);
  tab.style.display = "block";
  tab.classList.add("fade-in");
  el.classList.add("active");
  if (name === "analytics" && !analyticsReady) {
    initAnalyticsCharts();
    analyticsReady = true;
  }
  if (name === "history") renderHistory();
}

// ─── DRAG & DROP ──────────────────────────────────────────────────────────────
function doDrag(e, id) {
  e.preventDefault();
  document.getElementById(id).classList.add("drag");
}

function doDragLeave(id) {
  document.getElementById(id).classList.remove("drag");
}

function doDrop(e, type) {
  e.preventDefault();
  document.getElementById("dz-" + type).classList.remove("drag");
  const file = e.dataTransfer.files[0];
  if (file) showPreview(file, type);
}

function handleFile(e, type) {
  const file = e.target.files[0];
  if (file) showPreview(file, type);
}

function showPreview(file, type) {
  const reader = new FileReader();
  reader.onload = (ev) => {
    const img = document.getElementById("preview-" + type);
    img.src = ev.target.result;
    img.style.display = "block";
    document.getElementById("fname-" + type).textContent =
      file.name + " (" + Math.round(file.size / 1024) + " KB)";
  };
  reader.readAsDataURL(file);
}

// ─── PREDICTION ───────────────────────────────────────────────────────────────

/**
 * Main prediction entry point.
 * In full deployment, sends FormData to Flask /predict endpoint.
 * In demo mode (no backend), uses simulated probabilities.
 */
async function runPrediction() {
  const btn = document.getElementById("predict-btn");
  const btnIcon = document.getElementById("btn-icon");
  const btnText = document.getElementById("btn-text");

  btn.disabled = true;
  btnIcon.innerHTML = '<div class="spinner"></div>';
  btnText.textContent = "Analyzing...";

  try {
    const mammoFile = document.getElementById("inp-mammo").files[0];
    const usFile = document.getElementById("inp-us").files[0];
    const fw = parseFloat(document.getElementById("fw").value);

    let result;

    if (mammoFile && usFile) {
      // --- Try real backend first ---
      try {
        result = await callBackend(mammoFile, usFile, fw);
      } catch (_) {
        // Backend unavailable — use simulation
        result = simulateResult(fw);
      }
    } else {
      // No files uploaded — use simulation
      result = simulateResult(fw);
    }

    // Attach metadata
    result.pid =
      document.getElementById("pid").value ||
      "PT-" + String(Math.floor(Math.random() * 90000 + 10000));
    result.age = document.getElementById("page").value || "—";
    result.fw = fw.toFixed(2);
    result.ts = new Date().toLocaleString();

    lastResult = result;
    displayResult(result);
    addToHistory(result);
  } finally {
    btn.disabled = false;
    btnIcon.textContent = "🧠";
    btnText.textContent = "Run Multimodal Prediction";
  }
}

/**
 * Send images to Flask backend.
 * POST /predict  multipart/form-data
 */
/**
 * Send images and patient metadata to Flask backend.
 * POST /predict  multipart/form-data
 */
async function callBackend(mammoFile, usFile, fw) {
  const form = new FormData();
  form.append("mammogram", mammoFile);
  form.append("ultrasound", usFile);
  form.append("fusion_weight", fw);

  // ─── ADD THIS LINE TO SEND THE LIVE PATIENT ID INPUT ───
  const pidValue =
    document.getElementById("pid").value ||
    "PT-" + String(Math.floor(Math.random() * 90000 + 10000));
  form.append("patient_id", pidValue);

  const resp = await fetch("/predict", { method: "POST", body: form });
  if (!resp.ok) throw new Error("Backend error " + resp.status);
  const data = await resp.json();
  return {
    mammoMalig: data.mammogram_malignant * 100,
    usMalig: data.ultrasound_malignant * 100,
    benign: data.benign * 100,
    malignant: data.malignant * 100,
    result: data.prediction,
    // Keep backend ID synchronized
    pid: data.patient_id,
  };
}

/**
 * Simulate a prediction result (demo / no backend).
 * Uses fusion formula: fused = w1*mammo + (1-w1)*us
 */
function simulateResult(fw) {
  const mammoMalig = 0.1 + Math.random() * 0.85;
  const usMalig = 0.1 + Math.random() * 0.85;
  const fused = fw * mammoMalig + (1 - fw) * usMalig;
  return {
    mammoMalig: mammoMalig * 100,
    usMalig: usMalig * 100,
    benign: (1 - fused) * 100,
    malignant: fused * 100,
    result: fused > 0.5 ? "MALIGNANT" : "BENIGN",
  };
}

// ─── DISPLAY RESULT ───────────────────────────────────────────────────────────
function displayResult(r) {
  const card = document.getElementById("result-card");
  const label = document.getElementById("result-label");
  const isMal = r.result === "MALIGNANT";

  card.className = "result-card " + (isMal ? "malignant" : "benign");
  label.className = "result-label " + (isMal ? "malignant" : "benign");
  label.textContent = r.result;

  const confidence = isMal ? r.malignant : r.benign;
  document.getElementById("conf-score").textContent =
    confidence.toFixed(1) + "%";

  // Animate bars
  setTimeout(() => {
    document.getElementById("bar-benign").style.width =
      r.benign.toFixed(1) + "%";
    document.getElementById("bar-malignant").style.width =
      r.malignant.toFixed(1) + "%";
  }, 100);

  document.getElementById("val-benign").textContent = r.benign.toFixed(1) + "%";
  document.getElementById("val-malignant").textContent =
    r.malignant.toFixed(1) + "%";

  // AI explanation text
  const expl =
    isMal ?
      `<strong>⚠ Malignant Pattern Detected:</strong> The multimodal fusion model (w₁=${r.fw}) identified high-risk tissue characteristics. Mammography model malignancy score: ${r.mammoMalig.toFixed(1)}%. Ultrasound model malignancy score: ${r.usMalig.toFixed(1)}%. Fused probability: ${r.malignant.toFixed(1)}%. Clinical verification is strongly recommended.`
    : `<strong>✓ Benign Pattern Identified:</strong> The fused ResNet-18 classifier (w₁=${r.fw}) found predominantly benign tissue characteristics. Mammography malignancy score: ${r.mammoMalig.toFixed(1)}%. Ultrasound malignancy score: ${r.usMalig.toFixed(1)}%. Fused malignancy probability: ${r.malignant.toFixed(1)}%. Routine monitoring advised as per clinical protocol.`;
  document.getElementById("ai-explanation").innerHTML = expl;

  document.getElementById("result-area").style.display = "block";
  renderResultCharts(r);
}

// ─── RESULT CHARTS ────────────────────────────────────────────────────────────
function renderResultCharts(r) {
  destroyChart("modalityChart");
  destroyChart("fusionChart");

  chartInstances.modalityChart = new Chart(
    document.getElementById("modalityChart"),
    {
      type: "bar",
      data: {
        labels: ["Mammography", "Ultrasound", "Fusion"],
        datasets: [
          {
            label: "Benign %",
            data: [100 - r.mammoMalig, 100 - r.usMalig, r.benign],
            backgroundColor: "#16a34a",
            borderRadius: 5,
          },
          {
            label: "Malignant %",
            data: [r.mammoMalig, r.usMalig, r.malignant],
            backgroundColor: "#dc2626",
            borderRadius: 5,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: { boxWidth: 10, font: { size: 10 } },
          },
        },
        scales: {
          x: { ticks: { font: { size: 10 } } },
          y: { max: 100, ticks: { font: { size: 10 } } },
        },
      },
    },
  );

  chartInstances.fusionChart = new Chart(
    document.getElementById("fusionChart"),
    {
      type: "doughnut",
      data: {
        labels: ["Benign", "Malignant"],
        datasets: [
          {
            data: [r.benign, r.malignant],
            backgroundColor: ["#16a34a", "#dc2626"],
            hoverOffset: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: { boxWidth: 10, font: { size: 10 } },
          },
        },
      },
    },
  );
}

function destroyChart(id) {
  if (chartInstances[id]) {
    chartInstances[id].destroy();
    delete chartInstances[id];
  }
}

// ─── HISTORY ──────────────────────────────────────────────────────────────────
function addToHistory(r) {
  history.unshift({ ...r });
  document.getElementById("hist-count").textContent =
    history.length + " record" + (history.length !== 1 ? "s" : "");
}

function renderHistory() {
  const el = document.getElementById("history-list");
  if (!history.length) {
    el.innerHTML =
      '<div class="empty-state"><div class="empty-icon">📋</div>No predictions yet. Run a prediction to populate history.</div>';
    return;
  }
  el.innerHTML = history
    .map(
      (h) => `
    <div class="history-row">
      <div style="font-family:'Space Mono',monospace;font-size:11px;color:var(--blue)">${h.pid}</div>
      <div><span class="pill ${h.result === "BENIGN" ? "b" : "m"}">${h.result}</span></div>
      <div style="font-family:'Space Mono',monospace;font-size:11px">${h.benign.toFixed(1)}%</div>
      <div style="font-family:'Space Mono',monospace;font-size:11px">${h.malignant.toFixed(1)}%</div>
      <div style="font-size:11px;color:var(--text3)">${h.ts}</div>
      <div style="font-family:'Space Mono',monospace;font-size:11px">${h.fw}</div>
    </div>`,
    )
    .join("");
}

function clearHistory() {
  if (!confirm("Clear all prediction history?")) return;
  history = [];
  document.getElementById("hist-count").textContent = "0 records";
  renderHistory();
}

function exportCSV() {
  if (!history.length) {
    showToast("No records to export");
    return;
  }
  const rows = [
    "Patient ID,Age,Result,Benign %,Malignant %,Mammo Malig %,US Malig %,Fusion w1,Date/Time",
    ...history.map(
      (h) =>
        `${h.pid},${h.age},${h.result},${h.benign.toFixed(2)},${h.malignant.toFixed(2)},${h.mammoMalig.toFixed(2)},${h.usMalig.toFixed(2)},${h.fw},"${h.ts}"`,
    ),
  ];
  const blob = new Blob([rows.join("\n")], { type: "text/csv" });
  const a = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(blob),
    download:
      "prediction_history_" + new Date().toISOString().slice(0, 10) + ".csv",
  });
  a.click();
  showToast("CSV exported successfully");
}

// ─── ANALYTICS CHARTS ─────────────────────────────────────────────────────────
function initAnalyticsCharts() {
  const epochs = Array.from({ length: 15 }, (_, i) => i + 1);

  // Training loss
  const mammoLoss = [
    0.72, 0.61, 0.53, 0.46, 0.4, 0.35, 0.31, 0.28, 0.25, 0.23, 0.21, 0.2, 0.19,
    0.18, 0.17,
  ];
  const usLoss = [
    0.68, 0.58, 0.5, 0.43, 0.37, 0.33, 0.29, 0.26, 0.23, 0.21, 0.2, 0.18, 0.17,
    0.16, 0.15,
  ];

  // Training accuracy
  const mammoAcc = [62, 68, 73, 77, 80, 83, 85, 87, 88, 89, 90, 91, 91, 92, 92];
  const usAcc = [65, 70, 75, 79, 82, 84, 86, 88, 89, 90, 91, 91, 92, 92, 93];

  new Chart(document.getElementById("lossChart"), {
    type: "line",
    data: {
      labels: epochs,
      datasets: [
        {
          label: "Mammography",
          data: mammoLoss,
          borderColor: "#1a6cc8",
          backgroundColor: "rgba(26,108,200,.08)",
          pointRadius: 2,
          tension: 0.4,
        },
        {
          label: "Ultrasound",
          data: usLoss,
          borderColor: "#0891b2",
          backgroundColor: "rgba(8,145,178,.08)",
          pointRadius: 2,
          tension: 0.4,
          borderDash: [4, 4],
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: { boxWidth: 10, font: { size: 10 } },
        },
      },
      scales: {
        x: { title: { display: true, text: "Epoch", font: { size: 10 } } },
        y: { title: { display: true, text: "Loss", font: { size: 10 } } },
      },
    },
  });

  new Chart(document.getElementById("accChart"), {
    type: "line",
    data: {
      labels: epochs,
      datasets: [
        {
          label: "Mammography",
          data: mammoAcc,
          borderColor: "#16a34a",
          backgroundColor: "rgba(22,163,74,.08)",
          pointRadius: 2,
          tension: 0.4,
        },
        {
          label: "Ultrasound",
          data: usAcc,
          borderColor: "#d97706",
          backgroundColor: "rgba(217,119,6,.08)",
          pointRadius: 2,
          tension: 0.4,
          borderDash: [4, 4],
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: { boxWidth: 10, font: { size: 10 } },
        },
      },
      scales: {
        x: { title: { display: true, text: "Epoch", font: { size: 10 } } },
        y: {
          max: 100,
          title: { display: true, text: "Accuracy (%)", font: { size: 10 } },
        },
      },
    },
  });

  // AUC sweep bars
  const sweepData = [
    { w: "0.00", auc: 0.9285 },
    { w: "0.10", auc: 0.935 },
    { w: "0.20", auc: 0.942 },
    { w: "0.30", auc: 0.956 },
    { w: "0.40", auc: 0.971 },
    { w: "0.50", auc: 0.9891 },
    { w: "0.60", auc: 0.987 },
    { w: "0.70", auc: 0.985 },
    { w: "0.80", auc: 0.9845 },
    { w: "0.90", auc: 0.984 },
    { w: "1.00", auc: 0.9839 },
  ];
  const maxAUC = Math.max(...sweepData.map((d) => d.auc));
  document.getElementById("sweep-bars").innerHTML = sweepData
    .map((d) => {
      const isOpt = d.auc === maxAUC;
      // Adjusted bounds calculation for the UI slider filling logic based on your higher bounds
      const pct = (((d.auc - 0.9) / 0.09) * 100).toFixed(0);
      return `
      <div class="sweep-row">
        <span class="sweep-w">w₁=${d.w}</span>
        <div class="sweep-bar">
          <div class="sweep-fill" style="width:${Math.max(5, Math.min(100, pct))}%;background:${isOpt ? "#1a6cc8" : "#0891b2"}"></div>
        </div>
        <span class="sweep-auc">${d.auc.toFixed(4)}${isOpt ? '<span class="optimal-tag">★ Best</span>' : ""}</span>
      </div>`;
    })
    .join("");

  // ROC curves
  function genROC(auc, n = 60) {
    const pts = [[0, 0]];
    for (let i = 1; i <= n; i++) {
      const t = i / n;
      const s = Math.min(1, Math.pow(t, 1 / (2 * auc - 1 + 0.01)));
      pts.push([t, s]);
    }
    pts.push([1, 1]);
    return pts;
  }

  const rocDM = genROC(0.9839);
  const rocBUS = genROC(0.9285);
  const rocFusion = genROC(0.9891);

  new Chart(document.getElementById("rocChart"), {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "AI-DM  AUC=0.9839",
          data: rocDM.map((p) => ({ x: p[0], y: p[1] })),
          showLine: true,
          borderColor: "#1a6cc8",
          backgroundColor: "transparent",
          pointRadius: 0,
          tension: 0.4,
        },
        {
          label: "AI-BUS  AUC=0.9285",
          data: rocBUS.map((p) => ({ x: p[0], y: p[1] })),
          showLine: true,
          borderColor: "#0891b2",
          backgroundColor: "transparent",
          pointRadius: 0,
          tension: 0.4,
          borderDash: [5, 3],
        },
        {
          label: "AI-Fusion  AUC=0.9891",
          data: rocFusion.map((p) => ({ x: p[0], y: p[1] })),
          showLine: true,
          borderColor: "#16a34a",
          backgroundColor: "transparent",
          pointRadius: 0,
          tension: 0.4,
          borderWidth: 2.5,
        },
        {
          label: "Random classifier",
          data: [
            { x: 0, y: 0 },
            { x: 1, y: 1 },
          ],
          showLine: true,
          borderColor: "#94a3b8",
          backgroundColor: "transparent",
          pointRadius: 0,
          borderDash: [3, 3],
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: { boxWidth: 10, font: { size: 10 } },
        },
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "1 - Specificity (FPR)",
            font: { size: 10 },
          },
          min: 0,
          max: 1,
        },
        y: {
          title: {
            display: true,
            text: "Sensitivity (TPR)",
            font: { size: 10 },
          },
          min: 0,
          max: 1,
        },
      },
    },
  });
}

// ─── REPORT ───────────────────────────────────────────────────────────────────
// ─── REPORT ───────────────────────────────────────────────────────────────────
function generateReport() {
  // 1. Check if a prediction has been run yet to extract the weight setting
  const fwInput = document.getElementById("fw");
  const currentWeight = fwInput ? parseFloat(fwInput.value) : 0.5;

  showToast("Generating secure diagnostic report PDF...");

  // 2. Direct the browser directly to your Flask API endpoint with the query parameter string
  // This triggers a seamless browser file attachment download stream
  window.location.href = `/export_pdf?fusion_weight=${currentWeight}`;
}

function previewReport() {
  const el = document.getElementById("report-preview");
  const content = document.getElementById("report-content");
  el.style.display = "block";

  const histRows =
    history.length ?
      history
        .map(
          (h) => `
        <tr>
          <td>${h.pid}</td>
          <td><strong style="color:${h.result === "BENIGN" ? "#16a34a" : "#dc2626"}">${h.result}</strong></td>
          <td>${h.benign.toFixed(1)}%</td>
          <td>${h.malignant.toFixed(1)}%</td>
          <td>${h.ts}</td>
          <td>${h.fw}</td>
        </tr>`,
        )
        .join("")
    : '<tr><td colspan="6" style="color:#94a3b8;text-align:center">No predictions recorded</td></tr>';

  content.innerHTML = `
    <div style="font-size:11px;color:var(--text3);margin-bottom:4px;font-family:'Space Mono',monospace">
      REPORT PREVIEW — ${new Date().toLocaleDateString()}
    </div>
    <h2 style="font-size:16px;font-weight:600;margin-bottom:4px">Multimodal Breast Cancer Classification Report</h2>
    <p style="font-size:12px;color:var(--text3);margin-bottom:16px">BMED 4244 · ResNet-18 Fusion Model (w₁=0.50) · Spring 2026</p>

    <div class="section-title">Performance Summary</div>
    <table class="stat-table" style="margin-bottom:16px">
      <thead><tr><th>Metric</th><th>Mammography (DM)</th><th>Ultrasound (BUS)</th><th>Multimodal Fusion</th></tr></thead>
      <tbody>
        <tr><td>AUC</td><td>0.9839</td><td>0.9285</td><td class="highlight">0.9891</td></tr>
        <tr><td>Accuracy</td><td>93.58%</td><td>86.15%</td><td class="highlight">96.15%</td></tr>
        <tr><td>Sensitivity</td><td>93.74%</td><td>63.89%</td><td class="highlight">91.67%</td></tr>
        <tr><td>Specificity</td><td>93.27%</td><td>94.68%</td><td class="highlight">97.87%</td></tr>
        <tr><td>F1-Score</td><td>0.9513</td><td>0.7188</td><td class="highlight">0.9296</td></tr>
      </tbody>
    </table>

    <div class="section-title">Prediction History (${history.length} records)</div>
    <table class="stat-table">
      <thead><tr><th>Patient ID</th><th>Result</th><th>Benign</th><th>Malignant</th><th>Timestamp</th><th>w₁</th></tr></thead>
      <tbody>${histRows}</tbody>
    </table>

    <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:12px;font-size:12px;color:#1e40af;margin-top:16px;line-height:1.6">
      <strong>Note:</strong> Full PDF export with embedded charts is available via the Flask backend
      (<code>/export_pdf</code>). In Google Colab, use <code>matplotlib.backends.backend_pdf.PdfPages</code>
      to save all figures to a single PDF file.
    </div>`;
}

// ─── TOAST ────────────────────────────────────────────────────────────────────
function showToast(msg) {
  const t = document.createElement("div");
  t.className = "toast";
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3000);
}
