/* Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
 * SPDX-License-Identifier: MIT */
// Review queue, train/val/excluded splits, automatic pre-annotation and the
// propagation-run history -- the parts of the annotation workspace that talk
// to project-scoped APIs rather than to the canvas. Kept out of
// annotation-canvas.js so that file stays about pixels.
//
// setUp() is called by annotation-canvas.js with the hooks it owns
// (frame navigation, the status line, a reload of the current frame).

let projectId = null;
let hooks = {
  goToFrame: () => {},
  selectObject: async () => {},
  markFrameReviewed: async () => {},
  setStatus: () => {},
  reloadFrame: async () => {},
};

const el = (id) => document.getElementById(id);

async function apiGet(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(await detailOf(response));
  return response.json();
}

async function apiSend(method, path, body) {
  const response = await fetch(path, {
    method,
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await detailOf(response));
  return response.status === 204 ? null : response.json();
}

async function detailOf(response) {
  try {
    const body = await response.json();
    return typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch (parseError) {
    return response.statusText;
  }
}

// Shared "0-9, 12" parser for the frame-range inputs.
function parseFrameList(raw) {
  const trimmed = (raw || "").trim();
  if (!trimmed) return [];
  const frames = new Set();
  for (const chunk of trimmed.split(",")) {
    const piece = chunk.trim();
    if (!piece) continue;
    const range = piece.match(/^(\d+)\s*-\s*(\d+)$/);
    if (range) {
      const start = parseInt(range[1], 10);
      const end = parseInt(range[2], 10);
      if (end < start) throw new Error(`Range "${piece}" ends before it starts`);
      for (let i = start; i <= end; i += 1) frames.add(i);
      continue;
    }
    if (!/^\d+$/.test(piece)) throw new Error(`Cannot read "${piece}" as a frame or range`);
    frames.add(parseInt(piece, 10));
  }
  return [...frames].sort((a, b) => a - b);
}

// ---------------------------------------------------------- review queue

const REASON_LABELS = {
  low_model_score: "low score",
  large_area_change: "area jump",
  abrupt_centroid_displacement: "centroid jump",
  component_count_change: "split/merged",
  object_overlap_conflict: "overlap",
  identity_ambiguity: "identity unclear",
  disappearance: "disappeared",
};

async function refreshReviewQueue() {
  const list = el("review-queue-list");
  const badge = el("review-queue-count");
  let entries;
  try {
    entries = await apiGet(`/api/v1/projects/${projectId}/review-queue`);
  } catch (error) {
    list.innerHTML = "";
    const failed = document.createElement("li");
    failed.className = "lv-queue-empty";
    failed.textContent = `Could not load the queue: ${error.message}`;
    list.appendChild(failed);
    return;
  }

  badge.hidden = entries.length === 0;
  badge.textContent = String(entries.length);
  list.innerHTML = "";

  if (entries.length === 0) {
    const empty = document.createElement("li");
    empty.className = "lv-queue-empty";
    empty.textContent = "Nothing flagged for review.";
    list.appendChild(empty);
    return;
  }

  for (const entry of entries) {
    const item = document.createElement("li");
    item.className = "lv-queue-item";

    const head = document.createElement("div");
    head.className = "lv-queue-frame";
    const title = document.createElement("span");
    title.textContent = `Frame ${entry.frame_index}`;
    head.appendChild(title);
    const solved = document.createElement("button");
    solved.type = "button";
    solved.className = "lv-queue-solved";
    solved.textContent = "Solved";
    solved.title = "Mark this frame reviewed and remove it from the queue";
    solved.addEventListener("click", async (event) => {
      event.stopPropagation();
      solved.disabled = true;
      try {
        await hooks.markFrameReviewed(entry.frame_index);
      } catch (error) {
        hooks.setStatus(`Could not mark frame ${entry.frame_index} reviewed: ${error.message}`);
      }
      await refreshReviewQueue();
    });
    head.appendChild(solved);
    item.appendChild(head);

    if (entry.review_reasons.length > 0) {
      const reasons = document.createElement("div");
      reasons.className = "lv-queue-reasons";
      for (const reason of entry.review_reasons) {
        const chip = document.createElement("span");
        chip.className = "lv-chip";
        chip.textContent = REASON_LABELS[reason] || reason;
        reasons.appendChild(chip);
      }
      item.appendChild(reasons);
    }

    if (entry.review_comment) {
      const comment = document.createElement("div");
      comment.className = "lv-queue-comment";
      comment.textContent = entry.review_comment;
      item.appendChild(comment);
    }

    item.addEventListener("click", async () => {
      await hooks.goToFrame(entry.frame_index);
      if (entry.object_id) await hooks.selectObject(entry.object_id);
    });
    list.appendChild(item);
  }
}

// Review tab's "Mark all annotated frames reviewed": a dry run first, so the
// confirmation can say how many frames change and how many of those are
// still flagged in the queue.
async function markAllAnnotatedReviewed(currentFrame) {
  const status = el("mark-all-reviewed-status");
  const button = el("mark-all-reviewed-button");
  const url = `/api/v1/projects/${projectId}/frames/mark-annotated-reviewed`;
  button.disabled = true;
  try {
    const preview = await apiSend("POST", url, { dry_run: true });
    if (preview.frame_count === 0) {
      status.textContent = "Every annotated frame is already reviewed.";
      return;
    }
    const flagged = preview.needs_review_count
      ? `\n\n${preview.needs_review_count} of them are flagged in the review queue and will leave it.`
      : "";
    if (!window.confirm(`Mark ${preview.frame_count} annotated frame(s) as reviewed?${flagged}`)) {
      return;
    }
    const done = await apiSend("POST", url, { dry_run: false });
    status.textContent = `Marked ${done.frame_count} frame(s) reviewed.`;
    // The open frame's revision changed underneath the canvas.
    await hooks.reloadFrame(currentFrame);
    await refreshReviewQueue();
  } catch (error) {
    status.textContent = `Could not mark frames reviewed: ${error.message}`;
  } finally {
    button.disabled = false;
  }
}

// ----------------------------------------------------------- split ranges

let splits = [];

function renderSplits() {
  const container = el("split-list");
  container.innerHTML = "";

  if (splits.length === 0) {
    const empty = document.createElement("p");
    empty.className = "lv-hint";
    empty.textContent = "No ranges assigned yet.";
    container.appendChild(empty);
    return;
  }

  splits.forEach((split, index) => {
    const row = document.createElement("div");
    row.className = "lv-split-row";

    const text = document.createElement("span");
    text.textContent = `${split.start_frame}–${split.end_frame}`;
    row.appendChild(text);

    const tag = document.createElement("span");
    tag.className = `lv-chip lv-split-${split.label}`;
    tag.textContent = split.label;
    row.appendChild(tag);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "Remove";
    remove.addEventListener("click", () => {
      splits.splice(index, 1);
      renderSplits();
    });
    row.appendChild(remove);

    container.appendChild(row);
  });
}

async function loadSplits() {
  splits = await apiGet(`/api/v1/projects/${projectId}/split-assignments`);
  renderSplits();
}

function addSplit() {
  const start = parseInt(el("split-start").value, 10);
  const end = parseInt(el("split-end").value, 10);
  const status = el("split-status");
  if (Number.isNaN(start) || Number.isNaN(end)) {
    status.textContent = "Enter both a start and an end frame.";
    return;
  }
  if (end < start) {
    status.textContent = "The range ends before it starts.";
    return;
  }
  // Overlapping ranges would make a frame's split ambiguous at export time,
  // and nothing downstream resolves that -- refuse it here instead.
  const clash = splits.find((s) => start <= s.end_frame && end >= s.start_frame);
  if (clash) {
    status.textContent =
      `Overlaps the existing ${clash.start_frame}–${clash.end_frame} (${clash.label}) range.`;
    return;
  }
  splits.push({ start_frame: start, end_frame: end, label: el("split-label").value });
  splits.sort((a, b) => a.start_frame - b.start_frame);
  el("split-start").value = "";
  el("split-end").value = "";
  status.textContent = "Not saved yet.";
  renderSplits();
}

async function saveSplits(getRevision, setRevision) {
  const status = el("split-status");
  try {
    const updated = await apiSend("PUT", `/api/v1/projects/${projectId}/split-assignments`, {
      expected_revision: getRevision(),
      split_assignments: splits,
    });
    setRevision(updated.revision);
    status.textContent = `Saved ${splits.length} range(s).`;
  } catch (error) {
    status.textContent = `Could not save: ${error.message}`;
  }
}

// ------------------------------------------------- automatic pre-annotation

let autoPollTimer = null;
let autoJobId = null;

async function populateAutoBackends() {
  const models = await apiGet("/api/v1/models");
  const select = el("auto-backend");
  select.innerHTML = "";
  // Only backends that declare the capability can pre-annotate; the rest
  // would be rejected by the API, so show why rather than offering them.
  for (const backend of models.backends) {
    const capable = backend.capabilities.includes("class_aware_automatic_preannotation");
    if (!capable) continue;
    const option = document.createElement("option");
    option.value = backend.backend_id;
    option.textContent = backend.available
      ? backend.display_name
      : `${backend.display_name} (unavailable)`;
    option.disabled = !backend.available;
    option.title = backend.unavailable_reason || "";
    select.appendChild(option);
  }
  const hint = el("auto-backend-hint");
  const capable = models.backends.filter((b) =>
    b.capabilities.includes("class_aware_automatic_preannotation")
  );
  if (capable.length === 0) {
    hint.textContent = "No backend in this build can do class-aware pre-annotation.";
    el("auto-start-button").disabled = true;
    return;
  }
  // A disabled <option> leaves select.value empty, so read the reason off the
  // backend list instead -- otherwise an unavailable backend shows no
  // explanation at all, which is exactly when one is needed.
  const chosen = capable.find((b) => b.backend_id === select.value) || capable[0];
  hint.textContent = chosen.available ? "" : chosen.unavailable_reason || "Backend unavailable.";
  el("auto-start-button").disabled = !capable.some((b) => b.available);
}

function stopAutoPolling() {
  if (autoPollTimer !== null) {
    clearInterval(autoPollTimer);
    autoPollTimer = null;
  }
}

function pollAutoJob(jobId, currentFrame) {
  stopAutoPolling();
  autoJobId = jobId;
  el("auto-progress-container").hidden = false;
  autoPollTimer = setInterval(async () => {
    let record;
    try {
      record = await apiGet(`/api/v1/projects/${projectId}/jobs/${jobId}`);
    } catch (error) {
      stopAutoPolling();
      el("auto-status").textContent = `Lost track of the job: ${error.message}`;
      el("auto-start-button").disabled = false;
      return;
    }
    const percent = Math.round((record.progress_fraction || 0) * 100);
    el("auto-progress").value = percent;
    el("auto-progress-label").textContent = `${percent}%`;

    if (["succeeded", "failed", "cancelled"].includes(record.status)) {
      stopAutoPolling();
      autoJobId = null;
      el("auto-progress-container").hidden = true;
      el("auto-start-button").disabled = false;
      if (record.status === "succeeded") {
        el("auto-status").textContent =
          "Finished. Objects whose label was not in this project were skipped.";
        await hooks.reloadFrame(currentFrame);
      } else {
        el("auto-status").textContent = `Job ${record.status}: ${record.error || "no detail"}`;
      }
    }
  }, 700);
}

async function startAutoPreannotate(currentFrame) {
  const status = el("auto-status");
  let frames;
  let classIds;
  try {
    frames = parseFrameList(el("auto-frames").value);
    classIds = parseFrameList(el("auto-class-ids").value);
  } catch (error) {
    status.textContent = error.message;
    return;
  }
  if (frames.length === 0) frames = [currentFrame];

  el("auto-start-button").disabled = true;
  status.textContent = `Running on ${frames.length} frame(s)...`;
  try {
    const { job_id: jobId } = await apiSend(
      "POST",
      `/api/v1/projects/${projectId}/auto-preannotate`,
      {
        backend_id: el("auto-backend").value,
        frame_indices: frames,
        requested_class_ids: classIds,
      }
    );
    pollAutoJob(jobId, currentFrame);
  } catch (error) {
    status.textContent = `Rejected: ${error.message}`;
    el("auto-start-button").disabled = false;
  }
}

// ------------------------------------------------------- tracking run list

async function refreshTrackingRuns() {
  const list = el("tracking-runs-list");
  const status = el("runs-status");
  let runs;
  try {
    runs = await apiGet(`/api/v1/projects/${projectId}/tracking-runs`);
  } catch (error) {
    status.textContent = `Could not load runs: ${error.message}`;
    return;
  }
  status.textContent = "";
  list.innerHTML = "";

  if (runs.length === 0) {
    const empty = document.createElement("li");
    empty.className = "lv-export-empty";
    empty.textContent = "No propagation has been run yet.";
    list.appendChild(empty);
    return;
  }

  for (const run of runs.slice().reverse()) {
    const item = document.createElement("li");

    const title = document.createElement("div");
    title.className = "lv-export-title";
    title.textContent = `${run.direction} ${run.start_frame}–${run.end_frame} — ${run.status}`;
    item.appendChild(title);

    const meta = document.createElement("div");
    meta.className = "lv-export-meta";
    const done =
      run.last_completed_frame === null
        ? "no chunk committed yet"
        : `committed up to frame ${run.last_completed_frame}`;
    meta.textContent = `${run.backend_id} · ${run.object_ids.length} object(s) · ${done}`;
    item.appendChild(meta);

    if (run.manual_masks_protected_count > 0) {
      const protectedNote = document.createElement("div");
      protectedNote.className = "lv-export-meta";
      protectedNote.textContent =
        `${run.manual_masks_protected_count} manually edited mask(s) left untouched`;
      item.appendChild(protectedNote);
    }

    const actions = document.createElement("div");
    actions.className = "lv-export-actions";
    const add = (label, method, path, danger = false) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = danger ? "lv-btn lv-btn-danger" : "lv-btn";
      button.textContent = label;
      button.addEventListener("click", async () => {
        button.disabled = true;
        try {
          await apiSend(method, path);
          await refreshTrackingRuns();
        } catch (error) {
          status.textContent = error.message;
          button.disabled = false;
        }
      });
      actions.appendChild(button);
    };

    const base = `/api/v1/projects/${projectId}/tracking-runs/${run.run_id}`;
    if (run.status === "running") {
      add("Pause", "POST", `${base}/pause`);
      add("Cancel", "POST", `${base}/cancel`, true);
    } else if (run.status === "paused" || run.status === "interrupted") {
      add("Resume", "POST", `${base}/resume`);
      add("Cancel", "POST", `${base}/cancel`, true);
    }
    item.appendChild(actions);
    list.appendChild(item);
  }
}

// ------------------------------------------------------------------ wiring

function wireDialog(modalId, openId, closeIds, onOpen) {
  const modal = el(modalId);
  if (!modal) return;
  const close = () => {
    modal.hidden = true;
  };
  el(openId).addEventListener("click", async () => {
    modal.hidden = false;
    await onOpen();
  });
  for (const id of closeIds) el(id).addEventListener("click", close);
  modal.addEventListener("click", (event) => {
    if (event.target === modal) close();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !modal.hidden) close();
  });
}

export function setUp(options) {
  projectId = options.projectId;
  hooks = { ...hooks, ...options.hooks };

  el("refresh-review-queue").addEventListener("click", refreshReviewQueue);
  el("mark-all-reviewed-button").addEventListener("click", () =>
    markAllAnnotatedReviewed(options.getCurrentFrame())
  );
  el("tab-queue").addEventListener("click", refreshReviewQueue);

  el("add-split-button").addEventListener("click", addSplit);
  el("save-splits-button").addEventListener("click", () =>
    saveSplits(options.getRevision, options.setRevision)
  );
  el("tab-splits").addEventListener("click", () => {
    loadSplits().catch((error) => {
      el("split-status").textContent = `Could not load splits: ${error.message}`;
    });
  });

  wireDialog(
    "auto-preannotate-modal",
    "open-auto-preannotate-button",
    ["auto-close-button", "auto-dismiss-button"],
    async () => {
      el("auto-status").textContent = "";
      await populateAutoBackends();
    }
  );
  el("auto-backend").addEventListener("change", populateAutoBackends);
  el("auto-start-button").addEventListener("click", () =>
    startAutoPreannotate(options.getCurrentFrame())
  );
  el("auto-cancel-job-button").addEventListener("click", async () => {
    if (!autoJobId) return;
    await apiSend("POST", `/api/v1/projects/${projectId}/jobs/${autoJobId}/cancel`);
  });

  wireDialog(
    "tracking-runs-modal",
    "open-tracking-runs-button",
    ["runs-close-button", "runs-dismiss-button"],
    refreshTrackingRuns
  );
  el("runs-refresh-button").addEventListener("click", refreshTrackingRuns);

  // Populate the queue badge once on load so a project with outstanding
  // review work says so without the operator opening the tab first.
  refreshReviewQueue().catch(() => {});

  // The canvas refreshes the queue after edits that change it (hide/show,
  // review status, deleting a frame) so the list and badge stay current.
  return { refreshReviewQueue: () => refreshReviewQueue().catch(() => {}) };
}
