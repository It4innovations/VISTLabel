/* Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
 * SPDX-License-Identifier: MIT */
// Project creation with upload progress, cancellation and an unload guard,
// plus the project-list search filter, the create-project modal, the
// per-card status picker and the export dialog trigger.
// Deliberately plain ES modules, no build step.

// Relative module imports do not inherit the entry point's query string.
// Keep dependencies on the same asset version as the HTML, JS and CSS.
const assetQuery = new URL(import.meta.url).search;
const [{ openExportDialog }, { setUpThemeToggle }] = await Promise.all([
  import(`./exports.js${assetQuery}`),
  import(`./theme.js${assetQuery}`),
]);

setUpThemeToggle("theme-toggle");

const form = document.getElementById("create-project-form");
const progressContainer = document.getElementById("upload-progress-container");
const progressBar = document.getElementById("upload-progress");
const progressLabel = document.getElementById("upload-progress-label");
const cancelButton = document.getElementById("cancel-upload-button");
const submitButton = document.getElementById("submit-button");
const errorParagraph = document.getElementById("upload-error");

const modal = document.getElementById("create-project-modal");
const openModalButton = document.getElementById("create-project-button");
const closeModalButton = document.getElementById("close-modal-button");
const cancelModalButton = document.getElementById("cancel-modal-button");
const projectSearch = document.getElementById("project-search");
const projectCount = document.getElementById("project-count");
const noSearchResults = document.getElementById("no-search-results");

let activeXhr = null;
let uploadInProgress = false;

window.addEventListener("beforeunload", (event) => {
  if (uploadInProgress) {
    event.preventDefault();
    event.returnValue = "";
  }
});

function resetUploadUi() {
  uploadInProgress = false;
  activeXhr = null;
  submitButton.disabled = false;
}

function extractErrorDetail(xhr) {
  try {
    const parsed = JSON.parse(xhr.responseText);
    return parsed.detail || xhr.responseText;
  } catch (parseError) {
    return xhr.responseText;
  }
}

// ------------------------------------------------------------------ modal

function openModal() {
  modal.hidden = false;
  document.getElementById("project_name").focus();
}

function closeModal() {
  // Never strand an in-flight upload behind a closed dialog -- the progress
  // bar and its Cancel button live inside it.
  if (uploadInProgress) {
    return;
  }
  modal.hidden = true;
  errorParagraph.textContent = "";
  progressContainer.hidden = true;
}

if (openModalButton) {
  openModalButton.addEventListener("click", openModal);
}
for (const button of [closeModalButton, cancelModalButton]) {
  if (button) {
    button.addEventListener("click", closeModal);
  }
}
if (modal) {
  modal.addEventListener("click", (event) => {
    if (event.target === modal) {
      closeModal();
    }
  });
}
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && modal && !modal.hidden) {
    closeModal();
  }
});

// ----------------------------------------------------------- search filter

function applySearchFilter() {
  const cards = Array.from(document.querySelectorAll(".lv-card[data-project-name]"));
  const query = (projectSearch.value || "").trim().toLowerCase();
  let visible = 0;

  for (const card of cards) {
    const matches = !query || card.dataset.projectName.includes(query);
    card.hidden = !matches;
    if (matches) {
      visible += 1;
    }
  }

  if (noSearchResults) {
    noSearchResults.hidden = visible > 0 || cards.length === 0;
  }
  if (projectCount) {
    const suffix = visible === 1 ? "project" : "projects";
    projectCount.textContent = query
      ? `${visible} of ${cards.length} ${suffix}`
      : `${cards.length} ${cards.length === 1 ? "project" : "projects"}`;
  }
}

if (projectSearch) {
  projectSearch.addEventListener("input", applySearchFilter);
  applySearchFilter();
}

// ------------------------------------------------- status picker + export

// Status is an operator-set label with no automatic link to per-frame review
// (those track different things), so it is simply persisted as chosen. The
// revision is carried on the element and refreshed from each response to keep
// the optimistic-concurrency check working without a page reload.
for (const select of document.querySelectorAll(".lv-status-select")) {
  select.addEventListener("change", async () => {
    const previous = select.dataset.previousValue || select.value;
    select.disabled = true;
    const response = await fetch(`/api/v1/projects/${select.dataset.projectId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        expected_revision: parseInt(select.dataset.revision, 10),
        status: select.value,
      }),
    });
    select.disabled = false;

    if (!response.ok) {
      select.value = previous;
      const detail = await response.json().catch(() => ({}));
      window.alert(`Could not change status: ${detail.detail || response.statusText}`);
      return;
    }
    const updated = await response.json();
    select.dataset.revision = String(updated.revision);
    select.dataset.previousValue = updated.status;
    select.className = `lv-status-select lv-status-${updated.status}`;
  });
  select.dataset.previousValue = select.value;
}

for (const button of document.querySelectorAll(".lv-export-button")) {
  button.addEventListener("click", () => {
    openExportDialog(button.dataset.projectId, button.dataset.projectTitle);
  });
}

// Deleting a project removes its videos, masks and exports irreversibly, so
// require the name to be typed rather than accepting a single click.
for (const button of document.querySelectorAll(".lv-delete-button")) {
  button.addEventListener("click", async () => {
    const title = button.dataset.projectTitle;
    const typed = window.prompt(
      `Deleting "${title}" removes its video, annotations and exports for good.\n\n` +
        "Type the project name to confirm:"
    );
    if (typed === null) return;
    if (typed.trim() !== title) {
      window.alert("That does not match the project name -- nothing was deleted.");
      return;
    }
    button.disabled = true;
    const response = await fetch(`/api/v1/projects/${button.dataset.projectId}`, {
      method: "DELETE",
    });
    if (response.ok) {
      window.location.reload();
    } else {
      const detail = await response.json().catch(() => ({}));
      window.alert(`Could not delete: ${detail.detail || response.statusText}`);
      button.disabled = false;
    }
  });
}

// ---------------------------------------------------------- system status

const metricsModal = document.getElementById("metrics-modal");

function renderKeyValues(container, rows) {
  container.innerHTML = "";
  for (const [key, value, tone] of rows) {
    const term = document.createElement("dt");
    term.textContent = key;
    const detail = document.createElement("dd");
    detail.textContent = value;
    if (tone) detail.className = `lv-tone-${tone}`;
    container.appendChild(term);
    container.appendChild(detail);
  }
}

async function loadMetrics() {
  // Backend probing and the job-file walk make this take a couple of
  // seconds, so say something immediately rather than showing a blank box.
  renderKeyValues(document.getElementById("health-list"), [["Loading", "..."]]);
  try {
    await renderMetrics();
  } catch (error) {
    // Never leave the dialog silently blank -- an operator opening "system
    // status" to diagnose something must see the failure, not an empty box.
    renderKeyValues(document.getElementById("health-list"), [
      ["Error", String(error && error.message ? error.message : error), "warn"],
    ]);
  }
}

async function renderMetrics() {
  const [health, models, metrics] = await Promise.all([
    fetch("/api/v1/health").then((r) => r.json()),
    fetch("/api/v1/models").then((r) => r.json()),
    fetch("/api/v1/metrics").then((r) => r.json()),
  ]);

  renderKeyValues(document.getElementById("health-list"), [
    ["Overall", health.status, health.status === "ok" ? "ok" : "warn"],
    ["Version", health.package_version],
    ["Filesystem", health.filesystem.detail, health.filesystem.status === "ok" ? "ok" : "warn"],
    ["FFmpeg", health.ffmpeg.detail, health.ffmpeg.status === "ok" ? "ok" : "warn"],
    ["Python", health.python.detail],
    ["CUDA", health.cuda.detail, health.cuda.status === "ok" ? "ok" : "warn"],
  ]);

  renderKeyValues(
    document.getElementById("backends-list"),
    models.backends.map((backend) => [
      backend.display_name,
      backend.available ? "available" : backend.unavailable_reason || "unavailable",
      backend.available ? "ok" : "warn",
    ])
  );

  const jobRows = Object.entries(metrics.jobs_by_status).map(([status, count]) => [
    `Jobs ${status}`,
    String(count),
  ]);
  renderKeyValues(document.getElementById("metrics-list"), [
    ["Uptime", `${metrics.uptime_seconds} s`],
    ["Projects", String(metrics.project_count)],
    ["Annotated frames", String(metrics.total_annotated_frame_count)],
    ...jobRows,
  ]);
}

if (metricsModal) {
  const closeMetrics = () => {
    metricsModal.hidden = true;
  };
  document.getElementById("open-metrics-button").addEventListener("click", async () => {
    metricsModal.hidden = false;
    await loadMetrics();
  });
  document.getElementById("metrics-refresh-button").addEventListener("click", loadMetrics);
  for (const id of ["metrics-close-button", "metrics-dismiss-button"]) {
    document.getElementById(id).addEventListener("click", closeMetrics);
  }
  metricsModal.addEventListener("click", (event) => {
    if (event.target === metricsModal) closeMetrics();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !metricsModal.hidden) closeMetrics();
  });
}

// --------------------------------------------------------- import options

// Resolution and frame-rate choices are applied server-side as one re-encode
// before frame indexing. Only the options actually chosen are sent; omitting
// them keeps the video as uploaded.
const fileInput = document.getElementById("video_file");
const fileInfo = document.getElementById("video-file-info");
const resolutionSelect = document.getElementById("import-resolution");
const customSizeField = document.getElementById("import-custom-size-field");
const maxWidthInput = document.getElementById("import-max-width");
const maxHeightInput = document.getElementById("import-max-height");
const resolutionHint = document.getElementById("import-resolution-hint");
const frameModeSelect = document.getElementById("import-frame-mode");
const keepEveryNthField = document.getElementById("import-keep-every-nth-field");
const keepEveryNthInput = document.getElementById("import-keep-every-nth");
const targetFpsField = document.getElementById("import-target-fps-field");
const targetFpsInput = document.getElementById("import-target-fps");

function syncImportFields() {
  const custom = resolutionSelect.value === "custom";
  customSizeField.hidden = !custom;
  maxWidthInput.required = custom;
  maxHeightInput.required = custom;
  resolutionHint.hidden = resolutionSelect.value === "native";

  keepEveryNthField.hidden = frameModeSelect.value !== "nth";
  keepEveryNthInput.required = frameModeSelect.value === "nth";
  targetFpsField.hidden = frameModeSelect.value !== "fps";
  targetFpsInput.required = frameModeSelect.value === "fps";
}

function importOptionEntries() {
  const entries = [];
  if (resolutionSelect.value === "custom") {
    entries.push(["max_width", maxWidthInput.value], ["max_height", maxHeightInput.value]);
  } else if (resolutionSelect.value !== "native") {
    const [width, height] = resolutionSelect.value.split("x");
    entries.push(["max_width", width], ["max_height", height]);
  }
  if (frameModeSelect.value === "nth") {
    entries.push(["keep_every_nth", keepEveryNthInput.value]);
  } else if (frameModeSelect.value === "fps") {
    entries.push(["target_fps", targetFpsInput.value]);
  }
  return entries;
}

// Show the picked file's native size so "fit within" choices are informed.
// Browsers expose neither the frame rate nor, for codecs they cannot play
// (often HEVC), the size -- the hint then simply stays hidden.
function describeSelectedFile() {
  fileInfo.hidden = true;
  const file = fileInput.files && fileInput.files[0];
  if (!file) return;
  const probe = document.createElement("video");
  const url = URL.createObjectURL(file);
  probe.preload = "metadata";
  probe.addEventListener("loadedmetadata", () => {
    if (probe.videoWidth && probe.videoHeight) {
      fileInfo.textContent =
        `${probe.videoWidth}×${probe.videoHeight} · ${probe.duration.toFixed(1)} s`;
      fileInfo.hidden = false;
      if (!maxWidthInput.value && !maxHeightInput.value) {
        maxWidthInput.value = String(probe.videoWidth);
        maxHeightInput.value = String(probe.videoHeight);
      }
    }
    URL.revokeObjectURL(url);
  });
  probe.addEventListener("error", () => URL.revokeObjectURL(url));
  probe.src = url;
}

if (resolutionSelect) {
  resolutionSelect.addEventListener("change", syncImportFields);
  frameModeSelect.addEventListener("change", syncImportFields);
  fileInput.addEventListener("change", describeSelectedFile);
  syncImportFields();
}

// ---------------------------------------------------------------- source

// A project comes from an MP4 or from a COCO/YOLO dataset ZIP (stitched into
// a video server-side). Disabled inputs stay out of the FormData and out of
// required-field validation, so only the chosen source's fields count.
const sourceSelect = document.getElementById("create-source");
const videoSourceFields = document.getElementById("video-source-fields");
const datasetSourceFields = document.getElementById("dataset-source-fields");

const datasetFrameModeSelect = document.getElementById("dataset-frame-mode");
const datasetKeepEveryNthField = document.getElementById("dataset-keep-every-nth-field");
const datasetKeepEveryNthInput = document.getElementById("dataset-keep-every-nth");

function isDatasetSource() {
  return Boolean(sourceSelect) && sourceSelect.value === "dataset";
}

// keep_every_nth is only sent (enabled) when "Keep every nth image" is chosen.
function syncDatasetFrameFields() {
  const nth = isDatasetSource() && datasetFrameModeSelect.value === "nth";
  datasetKeepEveryNthField.hidden = !nth;
  datasetKeepEveryNthInput.disabled = !nth;
  datasetKeepEveryNthInput.required = nth;
}

function syncSourceFields() {
  const dataset = isDatasetSource();
  videoSourceFields.hidden = dataset;
  datasetSourceFields.hidden = !dataset;
  for (const field of videoSourceFields.querySelectorAll("input, select")) {
    field.disabled = dataset;
  }
  for (const field of datasetSourceFields.querySelectorAll("input, select")) {
    field.disabled = !dataset;
  }
  document.getElementById("dataset_file").required = dataset;
  syncDatasetFrameFields();
  if (!dataset) syncImportFields();
}

if (sourceSelect) {
  sourceSelect.addEventListener("change", syncSourceFields);
  datasetFrameModeSelect.addEventListener("change", syncDatasetFrameFields);
  syncSourceFields();
}

function describeImportReport(report) {
  const kept = report.image_count === report.source_image_count
    ? `${report.image_count}`
    : `${report.image_count} of ${report.source_image_count}`;
  const lines = [
    `Imported ${kept} ${report.dataset_format.toUpperCase()} images ` +
      `(${report.width}×${report.height}) with ${report.annotation_count} annotations ` +
      `as ${report.object_count} objects.`,
  ];
  if (report.bbox_annotation_count) {
    lines.push(`${report.bbox_annotation_count} were boxes, imported as rectangle masks.`);
  }
  if (report.warnings.length) {
    lines.push("", "Warnings:", ...report.warnings.map((warning) => `- ${warning}`));
  }
  return lines.join("\n");
}

// ----------------------------------------------------------------- upload

form.addEventListener("submit", (event) => {
  event.preventDefault();
  errorParagraph.textContent = "";
  const dataset = isDatasetSource();

  const xhr = new XMLHttpRequest();
  activeXhr = xhr;
  uploadInProgress = true;
  progressContainer.hidden = false;
  progressBar.value = 0;
  progressLabel.textContent = "0%";
  submitButton.disabled = true;

  xhr.upload.addEventListener("progress", (progressEvent) => {
    if (progressEvent.lengthComputable) {
      const percent = Math.round((progressEvent.loaded / progressEvent.total) * 100);
      progressBar.value = percent;
      progressLabel.textContent = `${percent}%`;
    }
  });

  // The response only arrives after the server has validated, possibly
  // re-encoded and indexed the video -- minutes for a long 4K recording.
  xhr.upload.addEventListener("load", () => {
    progressBar.removeAttribute("value");
    progressLabel.textContent = dataset
      ? "Building the video from the dataset on the server..."
      : "Processing video on the server...";
  });

  xhr.addEventListener("load", () => {
    if (xhr.status >= 200 && xhr.status < 300) {
      resetUploadUi();
      if (dataset) {
        const { report } = JSON.parse(xhr.responseText);
        // Only worth interrupting for when something was skipped or changed.
        if (report.warnings.length) window.alert(describeImportReport(report));
      }
      window.location.reload();
    } else {
      errorParagraph.textContent = `Upload failed (${xhr.status}): ${extractErrorDetail(xhr)}`;
      progressContainer.hidden = true;
      resetUploadUi();
    }
  });

  xhr.addEventListener("error", () => {
    errorParagraph.textContent = "Upload failed: network error";
    progressContainer.hidden = true;
    resetUploadUi();
  });

  xhr.addEventListener("abort", () => {
    errorParagraph.textContent = "Upload cancelled";
    progressContainer.hidden = true;
    resetUploadUi();
  });

  cancelButton.onclick = () => {
    if (activeXhr) {
      activeXhr.abort();
    }
  };

  xhr.open("POST", dataset ? "/api/v1/projects/import-dataset" : "/api/v1/projects");
  const formData = new FormData(form);
  if (!dataset) {
    for (const [name, value] of importOptionEntries()) {
      formData.append(name, value);
    }
  }
  xhr.send(formData);
});
