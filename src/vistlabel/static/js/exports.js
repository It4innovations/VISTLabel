/* Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
 * SPDX-License-Identifier: MIT */
// Dataset export dialog, shared by the project list and the
// annotation workspace. Exported as a module so both pages get one
// implementation: openExportDialog(projectId, projectName).

const { apiFetch } = await import(`./session.js${new URL(import.meta.url).search}`);

const modal = document.getElementById("export-modal");

// Both pages include the dialog partial, but guard anyway so importing this
// module can never break a page that does not.
const els = modal
  ? {
      subject: document.getElementById("export-project-name"),
      policy: document.getElementById("export-policy"),
      everyNthField: document.getElementById("export-every-nth-field"),
      everyNth: document.getElementById("export-every-nth"),
      everyNthLabel: document.getElementById("export-every-nth-label"),
      intervalsField: document.getElementById("export-intervals-field"),
      intervals: document.getElementById("export-intervals"),
      allFramesField: document.getElementById("export-all-frames-field"),
      acknowledgeAll: document.getElementById("export-acknowledge-all"),
      annotationType: document.getElementById("export-annotation-type"),
      layout: document.getElementById("export-layout"),
      imageFormat: document.getElementById("export-image-format"),
      negativeFraction: document.getElementById("export-negative-fraction"),
      multipartField: document.getElementById("export-multipart-field"),
      multipart: document.getElementById("export-multipart"),
      cocoExactField: document.getElementById("export-coco-exact-field"),
      cocoExact: document.getElementById("export-coco-exact"),
      status: document.getElementById("export-status"),
      progressContainer: document.getElementById("export-progress-container"),
      progress: document.getElementById("export-progress"),
      progressLabel: document.getElementById("export-progress-label"),
      history: document.getElementById("export-history"),
      startButton: document.getElementById("export-start-button"),
      closeButton: document.getElementById("export-close-button"),
      cancelButton: document.getElementById("export-cancel-button"),
    }
  : null;

let currentProjectId = null;
let pollTimer = null;

const POLICY_STORAGE_KEY = "vistlabel-export-policy";

// The option text as written in the template, before any count is appended.
function policyLabel(value) {
  const option = els.policy.querySelector(`option[value="${value}"]`);
  if (!option) return value;
  if (!option.dataset.label) option.dataset.label = option.textContent;
  return option.dataset.label;
}

// "Reviewed frames only (371)" / "All annotated frames ... (717)", so the
// difference between the two is visible before exporting. Counts are the
// same ones the project overview shows.
async function showPolicyCounts() {
  const counts = { reviewed_only: null, annotated_only: null };
  try {
    const response = await apiFetch(`/api/v1/projects/${currentProjectId}/progress`);
    if (response.ok) {
      const progress = await response.json();
      counts.reviewed_only = progress.reviewed_frame_count;
      counts.annotated_only = progress.annotated_frame_count;
    }
  } catch (error) {
    // counts are a convenience -- the plain labels still work
  }
  for (const [value, count] of Object.entries(counts)) {
    const option = els.policy.querySelector(`option[value="${value}"]`);
    if (!option) continue;
    const label = policyLabel(value);
    option.textContent = count === null ? label : `${label} (${count})`;
  }
}

function restorePolicy() {
  try {
    const saved = window.localStorage.getItem(POLICY_STORAGE_KEY);
    if (saved && els.policy.querySelector(`option[value="${saved}"]`)) els.policy.value = saved;
  } catch (error) {
    // storage unavailable -- keep the template default
  }
}

function rememberPolicy() {
  try {
    window.localStorage.setItem(POLICY_STORAGE_KEY, els.policy.value);
  } catch (error) {
    // storage unavailable -- nothing to remember
  }
}

function setStatus(message, isError = false) {
  els.status.textContent = message;
  els.status.classList.toggle("error", isError);
}

async function readError(response) {
  try {
    const body = await response.json();
    return body.detail || response.statusText;
  } catch (parseError) {
    return response.statusText;
  }
}

// Only the fields the chosen policy actually uses stay visible -- the API
// rejects an all_frames request without the acknowledgement, so surfacing
// that checkbox exactly when it applies keeps the rejection unsurprising.
function syncPolicyFields() {
  const policy = els.policy.value;
  els.everyNthField.hidden = !policy.startsWith("every_nth_");
  els.everyNthLabel.textContent = policy === "every_nth_annotated"
    ? "n (take every nth annotated frame)"
    : "n (take every nth reviewed frame)";
  els.intervalsField.hidden = policy !== "selected_intervals";
  els.allFramesField.hidden = policy !== "all_frames";
}

// The multipart policy only shapes YOLO polygons, and the extra COCO files
// only exist next to the Ultralytics layout -- the COCO layout always has
// them. "Project default" could be either, so it keeps both visible.
function syncFormatFields() {
  const layout = els.layout.value;
  els.multipartField.hidden = layout === "coco" || els.annotationType.value === "bbox";
  els.cocoExactField.hidden = layout === "coco";
}

function parseIntervals(raw) {
  const trimmed = (raw || "").trim();
  if (!trimmed) return [];
  return trimmed.split(",").map((chunk) => {
    const match = chunk.trim().match(/^(\d+)\s*-\s*(\d+)$/);
    if (!match) {
      throw new Error(`Cannot read interval "${chunk.trim()}" -- expected e.g. 0-99`);
    }
    const start = parseInt(match[1], 10);
    const end = parseInt(match[2], 10);
    if (end < start) {
      throw new Error(`Interval "${chunk.trim()}" ends before it starts`);
    }
    return [start, end];
  });
}

function buildRequestBody() {
  const policy = els.policy.value;
  const body = {
    frame_selection_policy: policy,
    every_nth: Math.max(1, parseInt(els.everyNth.value, 10) || 1),
    selected_intervals: policy === "selected_intervals" ? parseIntervals(els.intervals.value) : [],
    negative_frame_fraction: parseFloat(els.negativeFraction.value) || 0,
    always_include_coco_exact_annotations: els.cocoExact.checked,
    acknowledge_all_frames_warning: policy === "all_frames" && els.acknowledgeAll.checked,
  };
  // Empty string means "leave it to the project configuration"; sending null
  // would override the configured default with the schema default instead.
  if (els.imageFormat.value) body.image_format = els.imageFormat.value;
  if (els.annotationType.value) body.annotation_type = els.annotationType.value;
  if (els.layout.value) body.dataset_layout = els.layout.value;
  if (els.multipart.value) body.yolo_multipart_policy = els.multipart.value;
  return body;
}

function formatDate(iso) {
  return String(iso).replace("T", " ").replace(/\.\d+/, "").replace("+00:00", " UTC");
}

async function renderHistory() {
  const response = await apiFetch(`/api/v1/projects/${currentProjectId}/exports`);
  if (!response.ok) return;
  const manifests = await response.json();
  els.history.innerHTML = "";

  if (manifests.length === 0) {
    const empty = document.createElement("li");
    empty.className = "lv-export-empty";
    empty.textContent = "No exports yet.";
    els.history.appendChild(empty);
    return;
  }

  for (const manifest of manifests.slice().reverse()) {
    const item = document.createElement("li");

    const title = document.createElement("div");
    title.className = "lv-export-title";
    const kind = manifest.annotation_type === "bbox" ? "boxes" : "masks";
    const layout = manifest.dataset_layout === "coco" ? "COCO" : "YOLO";
    const policy = policyLabel(manifest.frame_selection_policy);
    title.textContent = `${layout} ${kind}, ${policy} — ${manifest.status}`;
    item.appendChild(title);

    const meta = document.createElement("div");
    meta.className = "lv-export-meta";
    const counts =
      manifest.status === "succeeded"
        ? ` · ${manifest.frame_count} frames (${manifest.train_frame_count} train / ${manifest.val_frame_count} val)`
        : "";
    meta.textContent = formatDate(manifest.created_at) + counts;
    item.appendChild(meta);

    if (manifest.error) {
      const error = document.createElement("div");
      error.className = "lv-export-error";
      error.textContent = manifest.error;
      item.appendChild(error);
    }

    const actions = document.createElement("div");
    actions.className = "lv-export-actions";

    if (manifest.status === "succeeded") {
      const download = document.createElement("a");
      download.className = "lv-btn";
      download.href = `/api/v1/projects/${currentProjectId}/exports/${manifest.export_id}/download`;
      download.textContent = "Download ZIP";
      actions.appendChild(download);

      const report = document.createElement("a");
      report.className = "lv-btn";
      report.href = `/api/v1/projects/${currentProjectId}/exports/${manifest.export_id}/validation-report`;
      report.target = "_blank";
      report.rel = "noopener";
      report.textContent = "Validation report";
      actions.appendChild(report);
    }

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "lv-btn lv-btn-danger";
    remove.textContent = "Delete";
    remove.addEventListener("click", async () => {
      remove.disabled = true;
      const deleted = await apiFetch(
        `/api/v1/projects/${currentProjectId}/exports/${manifest.export_id}`,
        { method: "DELETE" }
      );
      if (deleted.ok) {
        await renderHistory();
      } else {
        setStatus(`Could not delete export: ${await readError(deleted)}`, true);
        remove.disabled = false;
      }
    });
    actions.appendChild(remove);

    item.appendChild(actions);
    els.history.appendChild(item);
  }
}

function stopPolling() {
  if (pollTimer !== null) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function pollJob(jobId) {
  stopPolling();
  els.progressContainer.hidden = false;
  els.progress.value = 0;
  els.progressLabel.textContent = "0%";

  pollTimer = setInterval(async () => {
    const response = await apiFetch(`/api/v1/projects/${currentProjectId}/jobs/${jobId}`);
    if (!response.ok) {
      stopPolling();
      setStatus(`Lost track of the export job: ${await readError(response)}`, true);
      els.startButton.disabled = false;
      return;
    }
    const record = await response.json();
    const percent = Math.round((record.progress_fraction || 0) * 100);
    els.progress.value = percent;
    els.progressLabel.textContent = `${percent}%`;

    if (record.status === "succeeded") {
      stopPolling();
      setStatus("Export finished — download it below.");
      els.progressContainer.hidden = true;
      els.startButton.disabled = false;
      await renderHistory();
    } else if (record.status === "failed" || record.status === "cancelled") {
      stopPolling();
      setStatus(`Export ${record.status}: ${record.error || "no detail given"}`, true);
      els.progressContainer.hidden = true;
      els.startButton.disabled = false;
      await renderHistory();
    } else {
      setStatus(`Export running (${record.status})...`);
    }
  }, 700);
}

async function startExport() {
  let body;
  try {
    body = buildRequestBody();
  } catch (parseError) {
    setStatus(parseError.message, true);
    return;
  }

  els.startButton.disabled = true;
  setStatus("Starting export...");

  const response = await apiFetch(`/api/v1/projects/${currentProjectId}/exports`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    setStatus(`Export rejected: ${await readError(response)}`, true);
    els.startButton.disabled = false;
    return;
  }

  const { job_id: jobId } = await response.json();
  pollJob(jobId);
}

function closeDialog() {
  stopPolling();
  modal.hidden = true;
  els.startButton.disabled = false;
  els.progressContainer.hidden = true;
  setStatus("");
}

export async function openExportDialog(projectId, projectName) {
  if (!modal) return;
  currentProjectId = projectId;
  els.subject.textContent = projectName || projectId;
  setStatus("");
  els.progressContainer.hidden = true;
  modal.hidden = false;
  restorePolicy();
  syncPolicyFields();
  syncFormatFields();
  await Promise.all([showPolicyCounts(), renderHistory()]);
}

if (modal) {
  els.policy.addEventListener("change", () => {
    rememberPolicy();
    syncPolicyFields();
  });
  els.annotationType.addEventListener("change", syncFormatFields);
  els.layout.addEventListener("change", syncFormatFields);
  els.startButton.addEventListener("click", startExport);
  for (const button of [els.closeButton, els.cancelButton]) {
    button.addEventListener("click", closeDialog);
  }
  modal.addEventListener("click", (event) => {
    if (event.target === modal) closeDialog();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !modal.hidden) closeDialog();
  });
}
