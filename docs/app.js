const state = { data: null, classValue: null, bins: 10, feature: null };
const $ = (id) => document.getElementById(id);
const numericFeatures = () => state.data.features.filter((feature) => feature.kind === "numerical");
const categoricalFeatures = () => state.data.features.filter((feature) => feature.kind === "categorical");
const records = () => state.data.bins[String(state.bins)].records;
const rules = () => state.data.bins[String(state.bins)].rules;
const mean = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;

function selectedRecords(method, feature, classValue) {
  return records().filter((row) => row.method === method && row.feature === feature && row.class === classValue);
}
function importanceLimit() {
  return Math.max(...records().map((row) => Math.abs(row.importance)), 1);
}
function chartLayout(title, height = 560) {
  return { title: { text: title, x: 0.01, font: { size: 18 } }, height, autosize: true,
    margin: { l: 175, r: 30, t: 62, b: 62 }, paper_bgcolor: "transparent", plot_bgcolor: "transparent",
    font: { family: "system-ui, sans-serif", color: "#19222d" }, hoverlabel: { namelength: -1 } };
}
function matrixFor(method, feature, classValue) {
  const meta = state.data.features.find((item) => item.name === feature);
  const labels = meta.kind === "categorical" ? meta.categories : [feature];
  const width = meta.kind === "categorical" ? 2 : state.bins;
  const matrix = labels.map(() => Array(width).fill(0));
  selectedRecords(method, feature, classValue).forEach((row) => {
    const y = meta.kind === "categorical" ? labels.indexOf(row.category) : 0;
    if (y >= 0 && row.bin < width) matrix[y][row.bin] = row.importance;
  });
  return { labels, matrix };
}
function heatmapTrace(matrix, labels, method, limit, axes = {}) {
  return { type: "heatmap", z: matrix, y: labels, x: matrix[0].map((_, index) => index), name: method,
    colorscale: "RdBu", zmid: 0, zmin: -limit, zmax: limit, colorbar: { title: "Mean signed<br>importance" },
    xaxis: axes.xaxis, yaxis: axes.yaxis, hovertemplate: `${method}<br>%{y}<br>Bin %{x}: %{z:.3f}<extra></extra>` };
}
function ruleTotal(feature, category, classValue) {
  return rules().filter((row) => row.feature === feature && row.category === category && row.class === classValue)
    .reduce((sum, row) => sum + row.count, 0);
}
function drawOverview() {
  const features = state.data.features.map((item) => item.name);
  const traces = state.data.methods.map((method) => ({ type: "bar", name: method,
    x: features.map((name) => name.replaceAll("_", " ")),
    y: features.map((feature) => mean(selectedRecords(method, feature, state.classValue).map((row) => Math.abs(row.importance)))) }));
  Plotly.react($("overview-chart"), traces, { ...chartLayout("Mean absolute importance by feature", Math.max(520, features.length * 27)),
    barmode: "group", margin: { l: 190, r: 30, t: 62, b: 110 }, xaxis: { tickangle: -35 }, yaxis: { title: "Mean absolute importance" } },
    { responsive: true, displaylogo: false });
}
function drawCategorical() {
  const feature = state.feature;
  const meta = state.data.features.find((item) => item.name === feature);
  const limit = importanceLimit();
  const traces = [{ type: "bar", orientation: "h", name: "Rules", x: meta.categories.map((category) => ruleTotal(feature, category, state.classValue)), y: meta.categories,
    marker: { color: "#9aa7b4" }, xaxis: "x", yaxis: "y", hovertemplate: "Rules<br>%{y}: %{x}<extra></extra>" }];
  state.data.methods.forEach((method, index) => {
    const { labels, matrix } = matrixFor(method, feature, state.classValue);
    traces.push(heatmapTrace(matrix, labels, method, limit, { xaxis: `x${index + 2}`, yaxis: `y${index + 2}` }));
  });
  const layout = { ...chartLayout(`${feature.replaceAll("_", " ")} · categorical comparison`, Math.max(500, meta.categories.length * 36)),
    grid: { rows: 1, columns: 4, pattern: "independent" }, xaxis: { title: "Rule count" }, yaxis: { automargin: true },
    xaxis2: { title: "Category state" }, xaxis3: { title: "Category state" }, xaxis4: { title: "Category state" },
    yaxis2: { automargin: true }, yaxis3: { automargin: true }, yaxis4: { automargin: true }, showlegend: false };
  Plotly.react($("categorical-chart"), traces, layout, { responsive: true, displaylogo: false });
}
function drawNumeric() {
  const features = numericFeatures().map((item) => item.name);
  const limit = importanceLimit();
  const traces = [{ type: "bar", orientation: "h", name: "Rules", x: features.map((feature) => ruleTotal(feature, null, state.classValue)),
    y: features.map((name) => name.replaceAll("_", " ")), marker: { color: "#9aa7b4" }, xaxis: "x", yaxis: "y" }];
  state.data.methods.forEach((method, index) => {
    const matrix = features.map((feature) => matrixFor(method, feature, state.classValue).matrix[0]);
    traces.push(heatmapTrace(matrix, features.map((name) => name.replaceAll("_", " ")), method, limit, { xaxis: `x${index + 2}`, yaxis: `y${index + 2}` }));
  });
  const layout = { ...chartLayout("Numerical features · comparison", Math.max(560, features.length * 36)),
    grid: { rows: 1, columns: 4, pattern: "independent" }, xaxis: { title: "Rule count" }, yaxis: { automargin: true },
    xaxis2: { title: "Value bin" }, xaxis3: { title: "Value bin" }, xaxis4: { title: "Value bin" },
    yaxis2: { automargin: true }, yaxis3: { automargin: true }, yaxis4: { automargin: true }, showlegend: false };
  Plotly.react($("numeric-chart"), traces, layout, { responsive: true, displaylogo: false });
}
function drawAll() {
  drawOverview(); drawCategorical(); drawNumeric();
  $("status").textContent = `Showing predicted class ${state.classValue} with ${state.bins} numeric bins.`;
}
function setupControls() {
  state.data.classes.forEach((value) => $("class-select").add(new Option(String(value), value)));
  state.classValue = state.data.classes[0];
  categoricalFeatures().forEach((feature) => $("feature-select").add(new Option(feature.name.replaceAll("_", " "), feature.name)));
  state.feature = categoricalFeatures()[0].name;
  $("class-select").value = state.classValue; $("feature-select").value = state.feature;
  $("class-select").addEventListener("change", (event) => { state.classValue = Number(event.target.value); drawAll(); });
  $("feature-select").addEventListener("change", (event) => { state.feature = event.target.value; drawCategorical(); });
  $("bin-slider").addEventListener("input", (event) => { state.bins = Number(event.target.value); $("bin-value").textContent = state.bins; drawAll(); });
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((item) => { item.classList.remove("is-active"); item.setAttribute("aria-selected", "false"); });
    document.querySelectorAll(".panel").forEach((panel) => { panel.hidden = panel.id !== tab.dataset.panel; });
    tab.classList.add("is-active"); tab.setAttribute("aria-selected", "true"); window.dispatchEvent(new Event("resize"));
  }));
}
async function start() {
  try {
    const response = await fetch("data/dashboard.json");
    if (!response.ok) throw new Error(`Could not load dashboard data (${response.status})`);
    state.data = await response.json(); setupControls();
    while (!window.Plotly) await new Promise((resolve) => setTimeout(resolve, 25));
    drawAll();
  } catch (error) { $("status").textContent = error.message; $("status").classList.add("error"); }
}
start();
