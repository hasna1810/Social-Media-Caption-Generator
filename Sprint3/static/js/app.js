/**
 * CaptionAI — app.js
 * Handles: image upload/preview, platform selection,
 *          API call, rendering results, Chart.js charts,
 *          copy/download, toast notifications.
 */

"use strict";

// ── DOM refs ──────────────────────────────────────────────────────────────
const dropZone     = document.getElementById("dropZone");
const imageInput   = document.getElementById("imageInput");
const previewBox   = document.getElementById("previewBox");
const previewImg   = document.getElementById("previewImg");
const removeImg    = document.getElementById("removeImg");
const imgLabel     = document.getElementById("imgLabel");
const generateBtn  = document.getElementById("generateBtn");
const btnText      = generateBtn.querySelector(".btn-text");
const btnLoader    = generateBtn.querySelector(".btn-loader");
const emptyState   = document.getElementById("emptyState");
const results      = document.getElementById("results");
const captionBox   = document.getElementById("captionBox");
const platformInd  = document.getElementById("platformIndicator");
const copyBtn      = document.getElementById("copyBtn");
const downloadBtn  = document.getElementById("downloadBtn");
const toastMsg     = document.getElementById("toastMsg");

let radarChartInst   = null;
let platformCompInst = null;
let currentFile      = null;

// ── Platform selection ────────────────────────────────────────────────────
document.querySelectorAll(".platform-card").forEach(card => {
  card.addEventListener("click", () => {
    document.querySelectorAll(".platform-card").forEach(c => c.classList.remove("active"));
    card.classList.add("active");
    card.querySelector("input[type=radio]").checked = true;
  });
});
// Mark default active
document.querySelector('.platform-card[data-platform="instagram"]').classList.add("active");

function getSelectedPlatform() {
  const checked = document.querySelector('input[name="platform"]:checked');
  return checked ? checked.value : "instagram";
}

// ── Drop Zone ─────────────────────────────────────────────────────────────
dropZone.addEventListener("click", () => imageInput.click());

dropZone.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) loadPreview(file);
});

imageInput.addEventListener("change", () => {
  if (imageInput.files[0]) loadPreview(imageInput.files[0]);
});

removeImg.addEventListener("click", e => {
  e.stopPropagation();
  clearPreview();
});

function loadPreview(file) {
  currentFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  imgLabel.textContent = `${file.name}  (${(file.size / 1024).toFixed(1)} KB)`;
  dropZone.style.display = "none";
  previewBox.style.display = "block";
}

function clearPreview() {
  currentFile = null;
  previewImg.src = "";
  imageInput.value = "";
  previewBox.style.display = "none";
  dropZone.style.display = "block";
}

// ── Generate ──────────────────────────────────────────────────────────────
generateBtn.addEventListener("click", async () => {
  if (!currentFile) { showToast("⚠️ Please upload an image first.", "warn"); return; }

  setLoading(true);

  const formData = new FormData();
  formData.append("image", currentFile);
  formData.append("platform", getSelectedPlatform());

  try {
    const res = await fetch("/generate", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok || data.error) {
      throw new Error(data.error || "Server error");
    }

    renderResults(data);
  } catch (err) {
    showToast("❌ " + err.message, "error");
  } finally {
    setLoading(false);
  }
});

// ── Render Results ────────────────────────────────────────────────────────
function renderResults(data) {
  const { caption, platform, analytics, wordcloud, heatmap, platform_chart, styling_engine } = data;
  const a = analytics;

  // Base caption (Step 1)
  document.getElementById("baseCaptionBox").textContent = data.base_caption;

  // Arrow label — show engine used
  const arrowLbl = document.getElementById("arrowLabel");
  arrowLbl.textContent = data.styling_engine === "flan-t5-base (HuggingFace API)"
    ? "↓  Styled by Flan-T5-Base (HuggingFace)"
    : "↓  Styled by Rule-Based Engine";

  // Styled caption (Step 2)
  captionBox.textContent = caption;

  // Platform indicator
  const pConf = {
    instagram: { label: "Instagram", color: "#e1306c", icon: "fa-brands fa-instagram" },
    facebook:  { label: "Facebook",  color: "#1877f2", icon: "fa-brands fa-facebook" },
    linkedin:  { label: "LinkedIn",  color: "#0a66c2", icon: "fa-brands fa-linkedin" },
  };
  const pc = pConf[platform] || pConf.instagram;
  platformInd.innerHTML = `<i class="${pc.icon}" style="color:${pc.color}"></i> &nbsp;${pc.label} style`;
  platformInd.style.color = pc.color;

  // Styling engine badge
  const badge = document.getElementById("engineBadge");
  if (badge) {
    const isFlan = styling_engine === "flan-t5-large";
    badge.innerHTML = `<i class="fa-solid fa-${isFlan ? 'robot' : 'code'}"></i> Styled by: <strong>${isFlan ? 'Flan-T5-Large' : 'Rule-Based'}</strong>`;
    badge.className = `engine-badge ${isFlan ? 'flan' : 'rule'}`;
  }

  // Stat chips
  setStatChip("wordCount",    a.word_count);
  setStatChip("charCount",    a.char_count);
  setStatChip("hashCount",    a.hashtag_count);
  setStatChip("sentimentChip", a.sentiment, a.sentiment_color);

  // Quality
  document.getElementById("qualityNum").textContent = a.quality_score;
  animateBar("readBar", a.readability);
  animateBar("lenBar",  a.length_score);
  animateBar("engBar",  a.engagement_score);
  document.getElementById("readVal").textContent = a.readability;
  document.getElementById("lenVal").textContent  = a.length_score;
  document.getElementById("engVal").textContent  = a.engagement_score;
  drawGauge(a.quality_score);

  // Radar
  drawRadar(a.radar);

  // Sentiment
  const pct = a.sentiment_score;
  document.getElementById("sentimentNeedle").style.left = pct + "%";
  document.getElementById("sentimentLabel").textContent = a.sentiment;
  document.getElementById("sentimentLabel").style.color = a.sentiment_color;

  // ── NEW: Platform Comparison Chart ──────────────────────────────────────
  if (platform_chart) drawPlatformComparison(platform_chart);

  // ── NEW: Word Cloud ──────────────────────────────────────────────────────
  const wcCard = document.getElementById("wordCloudCard");
  const wcImg  = document.getElementById("wordCloudImg");
  if (wordcloud) {
    wcImg.src = `data:image/png;base64,${wordcloud}`;
    wcCard.style.display = "flex";
  } else {
    wcCard.style.display = "none";
  }

  // ── NEW: Attention Heatmap ───────────────────────────────────────────────
  const hmCard = document.getElementById("heatmapCard");
  const hmImg  = document.getElementById("heatmapImg");
  if (heatmap) {
    hmImg.src = `data:image/png;base64,${heatmap}`;
    hmCard.style.display = "flex";
  } else {
    hmCard.style.display = "none";
  }

  // Show results
  emptyState.style.display = "none";
  results.style.display = "flex";
  results.style.animation = "fadeIn 0.5s ease";

  showToast("✅ Caption generated successfully!");
}

function setStatChip(id, val, color) {
  const chip = document.getElementById(id);
  chip.querySelector(".stat-val").textContent = val;
  if (color) chip.querySelector(".stat-val").style.color = color;
}

function animateBar(id, pct) {
  const el = document.getElementById(id);
  setTimeout(() => { el.style.width = Math.min(100, pct) + "%"; }, 100);
}

// ── Gauge (half-donut) canvas ─────────────────────────────────────────────
function drawGauge(score) {
  const canvas = document.getElementById("gaugeCanvas");
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const cx = w / 2, cy = h - 8;
  const r = 72;
  const startAngle = Math.PI;
  const totalAngle = Math.PI;

  // Track
  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, startAngle + totalAngle);
  ctx.lineWidth = 12;
  ctx.strokeStyle = "rgba(255,255,255,0.07)";
  ctx.lineCap = "round";
  ctx.stroke();

  // Fill
  const pct = Math.min(score, 100) / 100;
  const color = pct > 0.7 ? "#10b981" : pct > 0.45 ? "#f59e0b" : "#ef4444";
  const grad = ctx.createLinearGradient(cx - r, cy, cx + r, cy);
  grad.addColorStop(0, color);
  grad.addColorStop(1, "#00d4ff");

  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, startAngle + totalAngle * pct);
  ctx.lineWidth = 12;
  ctx.strokeStyle = grad;
  ctx.lineCap = "round";
  ctx.stroke();
}

// ── Radar Chart ───────────────────────────────────────────────────────────
function drawRadar(radar) {
  const canvas = document.getElementById("radarChart");

  if (radarChartInst) {
    radarChartInst.destroy();
    radarChartInst = null;
  }

  radarChartInst = new Chart(canvas, {
    type: "radar",
    data: {
      labels: ["Professionalism", "Engagement", "Hashtags", "Length"],
      datasets: [{
        label: "Platform Style",
        data: [radar.professionalism, radar.engagement, radar.hashtags, radar.length],
        backgroundColor: "rgba(0, 212, 255, 0.15)",
        borderColor: "rgba(0, 212, 255, 0.8)",
        pointBackgroundColor: "#00d4ff",
        pointRadius: 4,
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        r: {
          min: 0, max: 100,
          ticks: { stepSize: 25, color: "rgba(255,255,255,0.3)", font: { size: 9 }, backdropColor: "transparent" },
          grid: { color: "rgba(255,255,255,0.07)" },
          angleLines: { color: "rgba(255,255,255,0.07)" },
          pointLabels: {
            color: "#94a3b8",
            font: { size: 10, family: "'DM Sans', sans-serif" },
          },
        }
      }
    }
  });
}

// ── Platform Comparison Chart (grouped bar) ───────────────────────────────
function drawPlatformComparison(pc) {
  const canvas = document.getElementById("platformCompChart");
  if (!canvas) return;
  if (platformCompInst) { platformCompInst.destroy(); platformCompInst = null; }

  const metrics = ["readability", "length", "engagement", "hashtags"];
  const labels  = ["Readability", "Length", "Engagement", "Hashtags"];

  platformCompInst = new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Instagram",
          data: metrics.map(m => pc.instagram[m]),
          backgroundColor: "rgba(225,48,108,0.75)",
          borderColor: "#e1306c",
          borderWidth: 1,
          borderRadius: 5,
        },
        {
          label: "Facebook",
          data: metrics.map(m => pc.facebook[m]),
          backgroundColor: "rgba(24,119,242,0.75)",
          borderColor: "#1877f2",
          borderWidth: 1,
          borderRadius: 5,
        },
        {
          label: "LinkedIn",
          data: metrics.map(m => pc.linkedin[m]),
          backgroundColor: "rgba(10,102,194,0.75)",
          borderColor: "#0a66c2",
          borderWidth: 1,
          borderRadius: 5,
        },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "#94a3b8", font: { size: 11 }, boxWidth: 12 }
        }
      },
      scales: {
        x: {
          ticks: { color: "#64748b", font: { size: 10 } },
          grid:  { color: "rgba(255,255,255,0.04)" },
        },
        y: {
          min: 0, max: 100,
          ticks: { color: "#64748b", font: { size: 10 }, stepSize: 25 },
          grid:  { color: "rgba(255,255,255,0.06)" },
        }
      }
    }
  });
}


copyBtn.addEventListener("click", async () => {
  const text = captionBox.textContent;
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    showToast("📋 Caption copied to clipboard!");
  } catch {
    // fallback
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
    showToast("📋 Caption copied!");
  }
});

downloadBtn.addEventListener("click", () => {
  const text = captionBox.textContent;
  if (!text) return;
  const platform = getSelectedPlatform();
  const blob = new Blob([text], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `caption_${platform}_${Date.now()}.txt`;
  a.click();
  showToast("⬇️ Caption downloaded!");
});

// ── Loading State ─────────────────────────────────────────────────────────
function setLoading(on) {
  generateBtn.disabled = on;
  btnText.style.display  = on ? "none"   : "flex";
  btnLoader.style.display = on ? "flex"  : "none";
}

// ── Toast ─────────────────────────────────────────────────────────────────
let toastTimer;
function showToast(msg) {
  toastMsg.textContent = msg;
  toastMsg.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastMsg.classList.remove("show"), 3000);
}

// ── Fade-in keyframes (inject once) ──────────────────────────────────────
const style = document.createElement("style");
style.textContent = `@keyframes fadeIn { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:none; } }`;
document.head.appendChild(style);
