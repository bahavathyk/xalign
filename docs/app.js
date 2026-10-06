const state = { data: null, defaultData: null, classValue: null, bins: 10, feature: null, showRules: false, pendingRows: null };
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
  const width = meta.kind === "categorical" ? (meta.states || 2) : state.bins;
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
function formatFeatureValue(value) {
  return new Intl.NumberFormat(undefined, { maximumSignificantDigits: 5 }).format(value);
}
function numericRangeAnnotations(features, axes, binCount) {
  const ranges = state.data.bins[String(state.bins)].ranges || {};
  const annotations = [];
  if (!axes.length) return annotations;
  features.forEach((feature, row) => {
    const range = ranges[feature];
    if (!range) return;
    const [low, high] = range.map(formatFeatureValue);
    axes.forEach((axis) => {
      for (const [x, text, xanchor] of [[-0.5, low, "right"], [binCount - 0.5, high, "left"]]) {
        annotations.push({ x, y: row, xref: `x${axis}`, yref: `y${axis}`, text, showarrow: false,
          xanchor, yanchor: "middle", font: { size: 10, color: "#19222d" },
          bgcolor: "rgba(255,255,255,0.82)", borderpad: 1 });
      }
    });
  });
  return annotations;
}
function uniqueValues(rows, column) {
  return [...new Set(rows.map((row) => String(row[column] ?? "").trim()).filter(Boolean))];
}
function isNumericColumn(rows, column) {
  const values = rows.map((row) => String(row[column] ?? "").trim()).filter(Boolean);
  return values.length > 0 && values.every((value) => Number.isFinite(Number(value)));
}
function fastDataset(rows, target, columns) {
  const sampled = rows.filter((row) => String(row[target] ?? "").trim()).slice(0, 3000);
  const classes = uniqueValues(sampled, target);
  if (classes.length < 2) throw new Error("The target column must contain at least two values.");
  if (classes.length > 20) throw new Error("The target column has too many unique values for a fast browser analysis.");
  const features = columns.map((name) => {
    const numerical = isNumericColumn(sampled, name);
    return { name, kind: numerical ? "numerical" : "categorical", categories: numerical ? [] : uniqueValues(sampled, name).slice(0, 20), states: numerical ? 0 : 1 };
  });
  const byBins = {};
  for (let binCount = 3; binCount <= 30; binCount += 1) {
    const records = [];
    const ruleRows = [];
    features.forEach((feature) => {
      const numericValues = feature.kind === "numerical" ? sampled.map((row) => Number(row[feature.name])).filter(Number.isFinite) : [];
      let low = Math.min(...numericValues); let high = Math.max(...numericValues);
      if (low === high) { low -= 0.5; high += 0.5; }
      const totalByGroup = new Map();
      const classByGroup = new Map();
      sampled.forEach((row) => {
        const raw = String(row[feature.name] ?? "").trim();
        let group = raw;
        if (feature.kind === "numerical") {
          const value = Number(raw);
          if (!Number.isFinite(value)) return;
          group = Math.min(binCount - 1, Math.floor(((value - low) / (high - low)) * binCount));
        } else if (!feature.categories.includes(group)) return;
        const key = String(group);
        totalByGroup.set(key, (totalByGroup.get(key) || 0) + 1);
        if (!classByGroup.has(key)) classByGroup.set(key, new Map());
        const counts = classByGroup.get(key);
        const targetValue = String(row[target]).trim();
        counts.set(targetValue, (counts.get(targetValue) || 0) + 1);
      });
      const groups = feature.kind === "categorical" ? feature.categories : Array.from({ length: binCount }, (_, index) => index);
      classes.forEach((classValue) => {
        const globalRate = sampled.filter((row) => String(row[target]).trim() === classValue).length / sampled.length;
        groups.forEach((group) => {
          const key = String(group); const total = totalByGroup.get(key) || 0;
          if (!total) return;
          const classCount = classByGroup.get(key)?.get(classValue) || 0;
          records.push({ method: "Fast attribution", feature: feature.name, category: feature.kind === "categorical" ? group : null,
            class: classValue, bin: feature.kind === "categorical" ? 0 : group, importance: classCount / total - globalRate });
          ruleRows.push({ feature: feature.name, category: feature.kind === "categorical" ? group : null,
            class: classValue, bin: feature.kind === "categorical" ? 0 : group, count: total });
        });
      });
    });
    byBins[String(binCount)] = { records, rules: ruleRows };
  }
  return { schemaVersion: 1, classes, methods: ["Fast attribution"], features, bins: byBins, ruleLabel: "Support count" };
}
function selectedFeatureNames() {
  return [...document.querySelectorAll("#feature-picker input:checked")].map((input) => input.value);
}
function populateFeaturePicker(columns, target) {
  const picker = $("feature-picker"); picker.replaceChildren();
  columns.filter((column) => column !== target).forEach((column) => {
    const label = document.createElement("label"); const input = document.createElement("input");
    input.type = "checkbox"; input.value = column; input.checked = true; label.append(input, document.createTextNode(column)); picker.append(label);
  });
  picker.hidden = false; $("analyze-upload").disabled = false;
}
function showCustomFile(file) {
  if (!window.Papa) throw new Error("The CSV parser is still loading; please try again.");
  Papa.parse(file, { header: true, skipEmptyLines: true, complete: (result) => {
    const columns = result.meta.fields || [];
    if (!columns.length || !result.data.length) { $("upload-status").textContent = "The CSV has no readable rows or columns."; return; }
    state.pendingRows = result.data;
    $("target-select").replaceChildren(...columns.map((column) => new Option(column, column)));
    $("target-field").hidden = false; populateFeaturePicker(columns, columns[0]);
    $("upload-status").textContent = `${result.data.length.toLocaleString()} rows loaded. Choose a target and included columns.`;
  }, error: (error) => { $("upload-status").textContent = `CSV could not be read: ${error.message}`; } });
}
function overlayShapes(features, labelsByFeature, rowByFeature, axisIds, binCount) {
  if (!state.showRules) return [];
  const shapes = [];
  axisIds.forEach((axisId) => features.forEach((feature) => {
    const labels = labelsByFeature[feature];
    const meta = state.data.features.find((item) => item.name === feature);
    const categorical = meta.kind === "categorical";
    const categories = categorical ? labels : [null];
    const totalBins = categorical ? (meta.states || 2) : binCount;
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
  const categories = meta.kind === "categorical" ? meta.categories : [feature];
  const limit = importanceLimit([feature]);
  const columnCount = state.data.methods.length + 1;
  const methodAxes = state.data.methods.map((_, index) => index + 2);
  const traces = [{ type: "bar", orientation: "h", name: state.data.ruleLabel || "Rules", x: categories.map((category) => ruleTotal(feature, meta.kind === "categorical" ? category : null, state.classValue)), y: categories.map((_, index) => index), customdata: categories,
    marker: { color: "#9aa7b4" }, xaxis: "x", yaxis: "y", hovertemplate: "Rules<br>%{customdata}: %{x}<extra></extra>" }];
  state.data.methods.forEach((method, index) => {
    const { labels, matrix } = matrixFor(method, feature, state.classValue);
    traces.push(heatmapTrace(matrix, labels, method, limit, { xaxis: `x${index + 2}`, yaxis: `y${index + 2}` }));
  });
  const labelsByFeature = { [feature]: categories };
  const layout = { ...chartLayout(`${feature.replaceAll("_", " ")} · comparison`, Math.max(500, categories.length * 36)),
    grid: { rows: 1, columns: columnCount, pattern: "independent" }, xaxis: { title: state.data.ruleLabel || "Rule count" }, yaxis: heatmapAxis(categories, true),
    shapes: overlayShapes([feature], labelsByFeature, { [feature]: 0 }, methodAxes, meta.kind === "categorical" ? (meta.states || 2) : state.bins),
    annotations: [
      ...[state.data.ruleLabel || "Rules", ...state.data.methods].map((text, index) => ({ text, x: (index + 0.5) / columnCount, y: 1.08, xref: "paper", yref: "paper", showarrow: false, font: { size: 15, color: "#19222d" } })),
      ...numericRangeAnnotations([feature], methodAxes, meta.kind === "categorical" ? 2 : state.bins),
    ], showlegend: false };
  methodAxes.forEach((axis, index) => { layout[`xaxis${axis}`] = { title: meta.kind === "categorical" ? "Category state" : "Value bin" }; layout[`yaxis${axis}`] = heatmapAxis(categories, false); });
  Plotly.react($("categorical-chart"), traces, layout, { responsive: true, displaylogo: false });
}
function drawNumeric() {
  const features = numericFeatures().map((item) => item.name);
  const limit = importanceLimit(features);
  const displayFeatures = features.map((name) => name.replaceAll("_", " "));
  const columnCount = state.data.methods.length + 1;
  const methodAxes = state.data.methods.map((_, index) => index + 2);
  const traces = [{ type: "bar", orientation: "h", name: state.data.ruleLabel || "Rules", x: features.map((feature) => ruleTotal(feature, null, state.classValue)),
    y: features.map((_, index) => index), customdata: displayFeatures, marker: { color: "#9aa7b4" }, xaxis: "x", yaxis: "y",
    hovertemplate: "Rules<br>%{customdata}: %{x}<extra></extra>" }];
  state.data.methods.forEach((method, index) => {
    const matrix = features.map((feature) => matrixFor(method, feature, state.classValue).matrix[0]);
    traces.push(heatmapTrace(matrix, features.map((name) => name.replaceAll("_", " ")), method, limit, { xaxis: `x${index + 2}`, yaxis: `y${index + 2}` }));
  });
  const labelsByFeature = Object.fromEntries(features.map((feature) => [feature, [feature]]));
  const rowByFeature = Object.fromEntries(features.map((feature, index) => [feature, index]));
  const heatmapWidth = 0.16;
  const heatmapStart = 0.28;
  const heatmapGap = 0.09;
  const heatmapDomains = methodAxes.map((_, index) => {
    const start = heatmapStart + index * (heatmapWidth + heatmapGap);
    return [start, start + heatmapWidth];
  });
  const layout = { ...chartLayout("Numerical features", Math.max(560, features.length * 36)),
    xaxis: { domain: [0.0, 0.19], title: state.data.ruleLabel || "Rule count" }, yaxis: { domain: [0, 1], ...heatmapAxis(displayFeatures, true) },
    shapes: overlayShapes(features, labelsByFeature, rowByFeature, methodAxes, state.bins),
    annotations: [
      ...[state.data.ruleLabel || "Rules", ...state.data.methods].map((text, index) => ({ text, x: (index + 0.5) / columnCount, y: 1.08, xref: "paper", yref: "paper", showarrow: false, font: { size: 15, color: "#19222d" } })),
      ...numericRangeAnnotations(features, methodAxes, state.bins),
    ], showlegend: false };
  methodAxes.forEach((axis, index) => {
    layout[`xaxis${axis}`] = { domain: heatmapDomains[index], title: "Value bin", range: [-0.5, state.bins - 0.5] };
    layout[`yaxis${axis}`] = { domain: [0, 1], ...heatmapAxis(displayFeatures, false) };
  });
  Plotly.react($("numeric-chart"), traces, layout, { responsive: true, displaylogo: false });
}
function drawAll() {
  drawOverview(); drawCategorical(); drawNumeric();
  $("status").textContent = `Showing predicted class ${state.classValue} with ${state.bins} numeric bins.`;
}
function populateAnalysisControls() {
  const classSelect = $("class-select"); classSelect.replaceChildren();
  const classLabels = state.data.classLabels || state.data.classes;
  state.data.classes.forEach((value, index) => classSelect.add(new Option(String(classLabels[index]), String(index))));
  state.classValue = state.data.classes[0];
  const featureSelect = $("feature-select"); featureSelect.replaceChildren();
  const featureChoices = categoricalFeatures().length ? categoricalFeatures() : state.data.features;
  featureChoices.forEach((feature) => featureSelect.add(new Option(feature.name.replaceAll("_", " "), feature.name)));
  state.feature = featureChoices[0]?.name;
  classSelect.value = "0"; featureSelect.value = state.feature;
}
function setupControls() {
  populateAnalysisControls();
  $("class-select").addEventListener("change", (event) => { state.classValue = state.data.classes[Number(event.target.value)]; drawAll(); });
  $("feature-select").addEventListener("change", (event) => { state.feature = event.target.value; drawCategorical(); });
  $("bin-slider").addEventListener("input", (event) => { state.bins = Number(event.target.value); $("bin-value").textContent = state.bins; drawAll(); });
  $("rules-toggle").addEventListener("change", (event) => { state.showRules = event.target.checked; drawAll(); });
  $("csv-upload").addEventListener("change", (event) => { const file = event.target.files[0]; if (file) showCustomFile(file); });
  $("target-select").addEventListener("change", (event) => { const columns = Object.keys(state.pendingRows?.[0] || {}); populateFeaturePicker(columns, event.target.value); });
  $("analyze-upload").addEventListener("click", () => {
    try {
      const target = $("target-select").value; const columns = selectedFeatureNames();
      if (!columns.length) throw new Error("Select at least one feature column.");
      const file = $("csv-upload").files[0];
      if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
        const form = new FormData(); form.append("file", file); form.append("target", target); columns.forEach((column) => form.append("features", column));
        $("analyze-upload").disabled = true; $("upload-status").textContent = "Training Random Forest and computing XAI explanations…";
        fetch("api/analyze", { method: "POST", body: form }).then(async (response) => {
          const payload = await response.json(); if (!response.ok) throw new Error(payload.error || "Local analysis failed."); return payload;
        }).then((payload) => {
          state.data = payload; state.bins = 10; $("bin-slider").value = "10"; $("bin-value").textContent = "10";
          populateAnalysisControls(); state.showRules = false; $("rules-toggle").checked = false; drawAll();
          $("upload-status").textContent = "Custom dataset active. Results came from the local Python Random Forest + XAI pipeline.";
        }).catch((error) => {
          $("upload-status").textContent = error instanceof TypeError
            ? "The local Python API is not reachable. Stop the static HTTP server and run local_server.py."
            : error.message;
        }).finally(() => { $("analyze-upload").disabled = false; });
      } else {
        state.data = fastDataset(state.pendingRows, target, columns); state.bins = 10; $("bin-slider").value = "10"; $("bin-value").textContent = "10";
        populateAnalysisControls(); state.showRules = false; $("rules-toggle").checked = false; drawAll();
        $("upload-status").textContent = "Custom browser-only attribution active. For exact XAI, run local_server.py.";
      }
    } catch (error) { $("upload-status").textContent = error.message; }
  });
  $("reset-dataset").addEventListener("click", () => {
    state.data = state.defaultData; state.bins = 10; $("bin-slider").value = "10"; $("bin-value").textContent = "10";
    state.showRules = false; $("rules-toggle").checked = false; populateAnalysisControls(); drawAll();
    $("upload-status").textContent = "Using the built-in German Credit dataset.";
  });
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((item) => { item.classList.remove("is-active"); item.setAttribute("aria-selected", "false"); });
    document.querySelectorAll(".panel").forEach((panel) => { panel.hidden = panel.id !== tab.dataset.panel; });
    tab.classList.add("is-active"); tab.setAttribute("aria-selected", "true"); window.dispatchEvent(new Event("resize"));
  }));
}
async function start() {
  try {
    const localMode = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
    if (!localMode) $("upload-panel").hidden = true;
    const response = await fetch("data/dashboard.json");
    if (!response.ok) throw new Error(`Could not load dashboard data (${response.status})`);
    state.data = await response.json(); state.defaultData = state.data; setupControls();
    while (!window.Plotly) await new Promise((resolve) => setTimeout(resolve, 25));
    drawAll();
  } catch (error) { $("status").textContent = error.message; $("status").classList.add("error"); }
}
start();
