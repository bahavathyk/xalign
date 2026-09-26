const state = { data: null, classValue: null, bins: 10, feature: null, showRules: false };
const $ = (id) => document.getElementById(id);
const numericFeatures = () => state.data.features.filter((feature) => feature.kind === "numerical");
const categoricalFeatures = () => state.data.features.filter((feature) => feature.kind === "categorical");
const records = () => state.data.bins[String(state.bins)].records;
const rules = () => state.data.bins[String(state.bins)].rules;
const mean = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;

function selectedRecords(method, feature, classValue) {
  return records().filter((row) => row.method === method && row.feature === feature && row.class === classValue);
}
function importanceLimit(features) {
  const selected = records().filter((row) => features.includes(row.feature) && row.class === state.classValue);
  return Math.max(...selected.map((row) => Math.abs(row.importance)), 1e-6);
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
  const rowIds = labels.map((_, index) => index);
  return { type: "heatmap", z: matrix, y: rowIds, x: matrix[0].map((_, index) => index),
    customdata: matrix.map((row, index) => row.map(() => labels[index])), name: method,
    colorscale: [[0, "#2166ac"], [0.5, "#f7f7f7"], [1, "#b2182b"]], zmid: 0, zmin: -limit, zmax: limit,
    colorbar: { title: "Mean signed<br>importance" },
    xaxis: axes.xaxis, yaxis: axes.yaxis, hovertemplate: `${method}<br>%{customdata}<br>Bin %{x}: %{z:.3f}<extra></extra>` };
}
function ruleTotal(feature, category, classValue) {
  return rules().filter((row) => row.feature === feature && row.category === category && row.class === classValue)
    .reduce((sum, row) => sum + row.count, 0);
}
function ruleCount(feature, category, bin, classValue) {
  return rules().filter((row) => row.feature === feature && row.category === category && row.bin === bin && row.class === classValue)
    .reduce((sum, row) => sum + row.count, 0);
}
function heatmapAxis(labels, showticklabels) {
  return { automargin: true, showticklabels, tickmode: "array", tickvals: labels.map((_, index) => index), ticktext: labels };
}
function overlayShapes(features, labelsByFeature, rowByFeature, axisIds, binCount) {
  if (!state.showRules) return [];
  const shapes = [];
  axisIds.forEach((axisId) => features.forEach((feature) => {
    const labels = labelsByFeature[feature];
    const categorical = labels.length > 1;
    const categories = categorical ? labels : [null];
    const totalBins = categorical ? 2 : binCount;
    const counts = categories.flatMap((category) => Array.from({ length: totalBins }, (_, bin) => ruleCount(feature, category, bin, state.classValue)));
    const maximum = Math.max(...counts, 0);
    if (!maximum) return;
    categories.forEach((category, row) => {
      const yCenter = rowByFeature[feature] + row;
      for (let bin = 0; bin < totalBins; bin += 1) {
        const count = ruleCount(feature, category, bin, state.classValue);
        if (!count) continue;
        const side = 0.55 * count / maximum;
        shapes.push({ type: "rect", xref: `x${axisId}`, yref: `y${axisId}`,
          x0: bin - side / 2, x1: bin + side / 2, y0: yCenter - side / 2, y1: yCenter + side / 2,
          line: { color: "#111827", width: 1 }, fillcolor: "rgba(0,0,0,0)" });
      }
    });
  }));
  return shapes;
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
  const limit = importanceLimit([feature]);
  const traces = [{ type: "bar", orientation: "h", name: "Rules", x: meta.categories.map((category) => ruleTotal(feature, category, state.classValue)), y: meta.categories.map((_, index) => index), customdata: meta.categories,
    marker: { color: "#9aa7b4" }, xaxis: "x", yaxis: "y", hovertemplate: "Rules<br>%{customdata}: %{x}<extra></extra>" }];
  state.data.methods.forEach((method, index) => {
    const { labels, matrix } = matrixFor(method, feature, state.classValue);
    traces.push(heatmapTrace(matrix, labels, method, limit, { xaxis: `x${index + 2}`, yaxis: `y${index + 2}` }));
  });
  const labelsByFeature = { [feature]: meta.categories };
  const layout = { ...chartLayout(`${feature.replaceAll("_", " ")} · categorical comparison`, Math.max(500, meta.categories.length * 36)),
    grid: { rows: 1, columns: 4, pattern: "independent" }, xaxis: { title: "Rule count" }, yaxis: heatmapAxis(meta.categories, true),
    xaxis2: { title: "Category state" }, xaxis3: { title: "Category state" }, xaxis4: { title: "Category state" },
    yaxis2: heatmapAxis(meta.categories, false), yaxis3: heatmapAxis(meta.categories, false), yaxis4: heatmapAxis(meta.categories, false),
    shapes: overlayShapes([feature], labelsByFeature, { [feature]: 0 }, [2, 3, 4], 2),
    annotations: ["Rules", ...state.data.methods].map((text, index) => ({ text, x: (index + 0.5) / 4, y: 1.08, xref: "paper", yref: "paper", showarrow: false, font: { size: 15, color: "#19222d" } })),
    showlegend: false };
  Plotly.react($("categorical-chart"), traces, layout, { responsive: true, displaylogo: false });
}
function drawNumeric() {
  const features = numericFeatures().map((item) => item.name);
  const limit = importanceLimit(features);
  const displayFeatures = features.map((name) => name.replaceAll("_", " "));
  const traces = [{ type: "bar", orientation: "h", name: "Rules", x: features.map((feature) => ruleTotal(feature, null, state.classValue)),
    y: features.map((_, index) => index), customdata: displayFeatures, marker: { color: "#9aa7b4" }, xaxis: "x", yaxis: "y",
    hovertemplate: "Rules<br>%{customdata}: %{x}<extra></extra>" }];
  state.data.methods.forEach((method, index) => {
    const matrix = features.map((feature) => matrixFor(method, feature, state.classValue).matrix[0]);
    traces.push(heatmapTrace(matrix, features.map((name) => name.replaceAll("_", " ")), method, limit, { xaxis: `x${index + 2}`, yaxis: `y${index + 2}` }));
  });
  const labelsByFeature = Object.fromEntries(features.map((feature) => [feature, [feature]]));
  const rowByFeature = Object.fromEntries(features.map((feature, index) => [feature, index]));
  const layout = { ...chartLayout("Numerical features", Math.max(560, features.length * 36)),
    grid: { rows: 1, columns: 4, pattern: "independent" }, xaxis: { title: "Rule count" }, yaxis: heatmapAxis(displayFeatures, true),
    xaxis2: { title: "Value bin" }, xaxis3: { title: "Value bin" }, xaxis4: { title: "Value bin" },
    yaxis2: heatmapAxis(displayFeatures, false), yaxis3: heatmapAxis(displayFeatures, false), yaxis4: heatmapAxis(displayFeatures, false),
    shapes: overlayShapes(features, labelsByFeature, rowByFeature, [2, 3, 4], state.bins),
    annotations: ["Rules", ...state.data.methods].map((text, index) => ({ text, x: (index + 0.5) / 4, y: 1.08, xref: "paper", yref: "paper", showarrow: false, font: { size: 15, color: "#19222d" } })),
    showlegend: false };
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
  $("rules-toggle").addEventListener("change", (event) => { state.showRules = event.target.checked; drawAll(); });
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
