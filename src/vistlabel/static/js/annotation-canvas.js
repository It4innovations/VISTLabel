/* Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
 * SPDX-License-Identifier: MIT */
// Annotation canvas: frame display, mask overlays, point/box prompts,
// brush/eraser on a full-resolution offscreen mask, undo/redo, and autosave
// through the revision-checked frame-annotation API. Plain ES module, no
// build step.

// Relative module imports do not inherit the entry point's query string.
// Keep dependencies on the same asset version as the HTML, JS and CSS.
const assetQuery = new URL(import.meta.url).search;
const [
  { openExportDialog }, { setUpConceptSearch },
  { setUpThemeToggle }, { setUp: setUpWorkspacePanels },
  { apiFetch, setUpLogout },
] = await Promise.all([
  import(`./exports.js${assetQuery}`),
  import(`./concept-search.js${assetQuery}`),
  import(`./theme.js${assetQuery}`),
  import(`./workspace-panels.js${assetQuery}`),
  import(`./session.js${assetQuery}`),
]);

setUpThemeToggle("theme-toggle");
setUpLogout();

const body = document.getElementById("annotation-body");
const projectId = body.dataset.projectId;
const frameWidth = parseInt(body.dataset.frameWidth, 10);
const frameHeight = parseInt(body.dataset.frameHeight, 10);
const frameCount = parseInt(body.dataset.frameCount, 10);
const videoFps = parseFloat(body.dataset.videoFps) || 15;

const OBJECT_COLORS = [
  [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
  [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
];
const UNDO_DEPTH = 20;
// Per mouse-wheel notch, in slider units (percent, frame pixels).
const WHEEL_ZOOM_STEP = 10;
const WHEEL_BRUSH_STEP = 2;
const SHOW_BOXES_STORAGE_KEY = "vistlabel-show-boxes";

function readShowBoxesPreference() {
  try {
    return window.localStorage.getItem(SHOW_BOXES_STORAGE_KEY) === "1";
  } catch (error) {
    return false; // storage unavailable -- boxes simply start hidden
  }
}

const state = {
  projectRevision: parseInt(body.dataset.projectRevision, 10),
  currentFrame: clampFrame(parseInt(body.dataset.lastOpenedFrame, 10) || 0),
  selectedObjectId: body.dataset.selectedObjectId || null,
  selectedBackendId: body.dataset.selectedBackendId || "fake",
  backends: [], // descriptors from /api/v1/models, incl. their capabilities
  objects: [],
  objectsRevision: 0,
  labels: [], // free-form project ontology, grown on demand from typed names
  frameAnnotation: null,
  overlayCache: new Map(), // object_id -> offscreen canvas (colored overlay)
  boxCache: new Map(), // object_id -> [x, y, width, height] of the saved mask
  showBoxes: readShowBoxesPreference(),
  pendingPoints: [],
  candidate: null, // { objectId, rawCanvas, overlayCanvas }
  conceptOverlay: null,
  editMaskCanvas: null,
  editMaskObjectId: null,
  tool: "point_positive",
  brushRadius: 10,
  maskOpacity: 0.55,
  zoom: 1.0,
  showOffFrameObjects: false,
  // Created in this tab and not saved on any frame yet: listed everywhere,
  // or creating a second object would hide the first.
  unplacedObjectIds: new Set(),
  isPainting: false,
  strokeErases: false, // the stroke in progress removes mask, whatever the tool
  hoverPoint: null, // pointer over the frame, in frame pixels; null when off it
  propagationRunning: false,
  // Undo/redo history, newest last: { kind: "mask", canvas } for a brush or
  // eraser stroke on the edit mask, { kind: "deleted", objectIds, name } for
  // a deleted object or class. Strokes belong to one object on one frame and
  // are dropped when either changes; deletes stay until the page closes.
  undoStack: [],
  redoStack: [],
  historyBusy: false, // an undo/redo of a delete is waiting on the server
  frameImage: null,
  pendingAxisPoints: [],
  propagationJobId: null,
  // "Hide" (H): view-only, per frame -- `${frame}:${objectId}` keys of
  // overlays not drawn. Never saved; the data and the export are unchanged.
  viewHidden: new Set(),
};

const els = {
  backendSelect: document.getElementById("backend-select"),
  canvas: document.getElementById("frame-canvas"),
  canvasScroll: document.getElementById("canvas-scroll"),
  boxLayer: document.getElementById("box-layer"),
  pointLayer: document.getElementById("point-layer"),
  brushCursor: document.getElementById("brush-cursor"),
  objectList: document.getElementById("object-list"),
  toggleOffFrameObjects: document.getElementById("toggle-offframe-objects"),
  classSection: document.getElementById("class-section"),
  classCount: document.getElementById("class-count"),
  classList: document.getElementById("class-list"),
  newObjectLabel: document.getElementById("new-object-label"),
  addObjectButton: document.getElementById("add-object-button"),
  brushRadius: document.getElementById("brush-radius"),
  maskOpacity: document.getElementById("mask-opacity"),
  showBoxes: document.getElementById("show-boxes"),
  zoomSlider: document.getElementById("zoom-slider"),
  canvasScroll: document.getElementById("canvas-scroll"),
  undoButton: document.getElementById("undo-button"),
  redoButton: document.getElementById("redo-button"),
  acceptButton: document.getElementById("accept-candidate-button"),
  clearPointsButton: document.getElementById("clear-points-button"),
  prevFrameButton: document.getElementById("prev-frame-button"),
  nextFrameButton: document.getElementById("next-frame-button"),
  frameSlider: document.getElementById("frame-slider"),
  frameNumberInput: document.getElementById("frame-number-input"),
  frameCountLabel: document.getElementById("frame-count-label"),
  statusLine: document.getElementById("status-line"),
  reviewStatus: document.getElementById("review-status"),
  barReviewStatus: document.getElementById("bar-review-status"),
  reviewComment: document.getElementById("review-comment"),
  saveReviewButton: document.getElementById("save-review-button"),
  saveAxisHintButton: document.getElementById("save-axis-hint-button"),
  deleteAxisHintButton: document.getElementById("delete-axis-hint-button"),
  occluderSelect: document.getElementById("occluder-select"),
  markOccluderButton: document.getElementById("mark-occluder-button"),
  occlusionEdgesSummary: document.getElementById("occlusion-edges-summary"),
  propagationDirection: document.getElementById("propagation-direction"),
  propagationEndFrame: document.getElementById("propagation-end-frame"),
  propagateButton: document.getElementById("propagate-button"),
  cancelPropagationButton: document.getElementById("cancel-propagation-button"),
  propagationStatus: document.getElementById("propagation-status"),
  propagationNotify: document.getElementById("propagation-notify"),
  playButton: document.getElementById("play-button"),
  playbackSpeed: document.getElementById("playback-speed"),
  deleteFrameButton: document.getElementById("delete-frame-button"),
  frameExcludedBadge: document.getElementById("frame-excluded-badge"),
};

const ctx = els.canvas.getContext("2d");
// Replaced by setUpWorkspacePanels()'s return value once init() runs.
let panels = { refreshReviewQueue: async () => {} };
let conceptPreviewGeneration = 0;
const conceptSearch = setUpConceptSearch({
  reportStatus: setStatus,
  getContext: () => ({ frameIndex: state.currentFrame, backend: selectedBackend() }),
  request: (context, text) => apiJson(
    "POST", `/api/v1/projects/${projectId}/segment-concept`,
    { backend_id: context.backend.backend_id, frame_index: context.frameIndex, text }
  ),
  preview: async (proposal) => {
    const token = ++conceptPreviewGeneration;
    state.conceptOverlay = null;
    render();
    if (!proposal) return;
    const overlay = await coloredOverlayFromMaskBase64(
      proposal.visible_mask_png_base64, [0, 180, 255], 0.8
    );
    if (token !== conceptPreviewGeneration) return;
    state.conceptOverlay = overlay;
    render();
  },
  accept: async (proposal, labelName, provenance) => {
    const frameIndex = provenance.frame_index;
    if (frameIndex !== state.currentFrame) throw new Error("Return to the search frame first");
    els.backendSelect.disabled = true;
    try {
      // Keep the created object on this proposal if saving fails, so retrying
      // does not create duplicate identities.
      if (!proposal.createdObjectId) {
        const labelId = await resolveLabelId(labelName);
        const created = await apiJson("POST", `/api/v1/projects/${projectId}/objects`, {
          label_id: labelId, label_name: labelName,
        });
        proposal.createdObjectId = created.object_id;
        state.objects.push(created);
        state.unplacedObjectIds.add(created.object_id);
        state.objectsRevision += 1;
      }
      const object = state.objects.find((entry) => entry.object_id === proposal.createdObjectId);
      if (object.label_name !== labelName) {
        const labelId = await resolveLabelId(labelName);
        const updated = await apiJson(
          "PATCH", `/api/v1/projects/${projectId}/objects/${object.object_id}`,
          { expected_revision: state.objectsRevision, label_id: labelId, label_name: labelName }
        );
        Object.assign(object, updated);
        state.objectsRevision += 1;
        await refreshLabels();
      }
      await upsertObjectMask(
        proposal.createdObjectId, proposal.visible_mask_png_base64,
        `${provenance.backend_id}_text`,
        { backendId: provenance.backend_id, manuallyModified: false,
          confidence: proposal.score, provenance }
      );
      await selectObject(proposal.createdObjectId);
      setStatus(`Saved text segmentation as ${labelName}`);
    } finally {
      els.backendSelect.disabled = false;
    }
  },
});

function clampFrame(index) {
  return Math.max(0, Math.min(frameCount - 1, index));
}

function objectColor(objectId) {
  const index = state.objects.findIndex((o) => o.object_id === objectId);
  const [r, g, b] = OBJECT_COLORS[(index < 0 ? 0 : index) % OBJECT_COLORS.length];
  return `rgb(${r}, ${g}, ${b})`;
}

function objectColorTuple(objectId) {
  const index = state.objects.findIndex((o) => o.object_id === objectId);
  return OBJECT_COLORS[(index < 0 ? 0 : index) % OBJECT_COLORS.length];
}

function setStatus(text) {
  els.statusLine.textContent = text;
}

// ---------------------------------------------------------------- API ----

async function apiGet(path) {
  const response = await apiFetch(path);
  if (!response.ok) throw new Error(`GET ${path} failed: ${response.status}`);
  return response.json();
}

async function apiJson(method, path, body) {
  const response = await apiFetch(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const raw = await response.text();
    const error = new Error(`${method} ${path} failed: ${response.status}`);
    error.status = response.status;
    // FastAPI wraps HTTPException messages as {"detail": "..."}; unwrap it so
    // callers can put the backend's own wording in front of the operator.
    error.detail = raw;
    try {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed.detail === "string") error.detail = parsed.detail;
    } catch {
      // non-JSON body (proxy error page, etc.) -- keep the raw text
    }
    throw error;
  }
  return response.status === 204 ? null : response.json();
}

// -------------------------------------------------------- mask helpers ----

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

function maskBase64ToImage(base64Png) {
  return loadImage(`data:image/png;base64,${base64Png}`);
}

async function coloredOverlayFromMaskBase64(base64Png, [r, g, b], opacity) {
  const img = await maskBase64ToImage(base64Png);
  const canvas = document.createElement("canvas");
  canvas.width = img.width;
  canvas.height = img.height;
  const octx = canvas.getContext("2d");
  octx.drawImage(img, 0, 0);
  const imageData = octx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;
  for (let i = 0; i < data.length; i += 4) {
    const on = data[i] > 127;
    data[i] = r;
    data[i + 1] = g;
    data[i + 2] = b;
    data[i + 3] = on ? Math.round(opacity * 255) : 0;
  }
  octx.putImageData(imageData, 0, 0);
  return canvas;
}

/** Binary (black/white, no anti-aliasing) offscreen canvas, used for the
 * brush/eraser edit surface and decoded from an existing accepted mask. */
async function binaryCanvasFromMaskBase64(base64Png) {
  const img = await maskBase64ToImage(base64Png);
  const canvas = document.createElement("canvas");
  canvas.width = frameWidth;
  canvas.height = frameHeight;
  const octx = canvas.getContext("2d");
  octx.drawImage(img, 0, 0);
  return canvas;
}

function blankBinaryCanvas() {
  const canvas = document.createElement("canvas");
  canvas.width = frameWidth;
  canvas.height = frameHeight;
  return canvas;
}

/** Threshold a possibly anti-aliased canvas to strict 0/255 grayscale and
 * return it as base64 PNG -- the backend rejects non-binary masks. */
function canvasToBinaryMaskPngBase64(canvas) {
  const octx = canvas.getContext("2d");
  const imageData = octx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;
  for (let i = 0; i < data.length; i += 4) {
    const on = data[i + 3] > 127 && data[i] > 127;
    const value = on ? 255 : 0;
    data[i] = value;
    data[i + 1] = value;
    data[i + 2] = value;
    data[i + 3] = 255;
  }
  const out = document.createElement("canvas");
  out.width = canvas.width;
  out.height = canvas.height;
  out.getContext("2d").putImageData(imageData, 0, 0);
  return out.toDataURL("image/png").split(",")[1];
}

// ------------------------------------------------------------ rendering ----

function applyZoom() {
  els.canvas.style.width = `${Math.round(frameWidth * state.zoom)}px`;
  els.canvas.style.height = `${Math.round(frameHeight * state.zoom)}px`;
}

// Tinting is a full-frame pixel pass; redoing it on every render was the
// biggest cost of playback. Reuse it until the edit mask is replaced (new
// canvas) or painted on (paintAt bumps editMaskVersion).
let editMaskVersion = 0;
let editTintCache = null;

function tintedEditMask(r, g, b) {
  const source = state.editMaskCanvas;
  const c = editTintCache;
  if (!c || c.source !== source || c.version !== editMaskVersion || c.rgb !== `${r},${g},${b}`) {
    editTintCache = {
      source,
      version: editMaskVersion,
      rgb: `${r},${g},${b}`,
      tinted: tintBinaryCanvas(source, r, g, b, 1),
    };
  }
  return editTintCache.tinted;
}

function render() {
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  if (state.frameImage) {
    ctx.drawImage(state.frameImage, 0, 0);
  }
  for (const object of state.objects) {
    if (isViewHidden(object.object_id)) continue;
    const overlay = state.overlayCache.get(object.object_id);
    if (overlay) {
      ctx.drawImage(overlay, 0, 0);
    }
  }
  if (state.editMaskCanvas && state.editMaskObjectId && !isViewHidden(state.editMaskObjectId)) {
    const [r, g, b] = objectColorTuple(state.editMaskObjectId);
    ctx.save();
    ctx.globalAlpha = Math.min(1, state.maskOpacity + 0.25);
    ctx.globalCompositeOperation = "source-over";
    ctx.filter = "none";
    ctx.drawImage(tintedEditMask(r, g, b), 0, 0);
    ctx.restore();
  }
  renderObjectBoxes();
  if (state.candidate) {
    ctx.drawImage(state.candidate.overlayCanvas, 0, 0);
  }
  if (state.conceptOverlay) ctx.drawImage(state.conceptOverlay, 0, 0);
  renderPromptPoints();
  const savedAxisHint = savedGeometryHintPolyline();
  els.deleteAxisHintButton.disabled = savedAxisHint.length === 0;
  drawAxisHint(savedAxisHint, "rgba(255,255,0,0.6)");
  drawAxisHint(state.pendingAxisPoints.map((p) => [p.x, p.y]), "rgba(255,140,0,0.9)");
}

// The same tight box the bbox export derives from each mask, so what is
// drawn here is exactly what a detection dataset will contain. Drawn as
// elements over the canvas (see #box-layer in annotation.css) so the tag
// text stays sharp at any zoom. render() runs on every brush stroke, so the
// elements are only rebuilt when something they show has changed.
let boxLayerKey = "";

function renderObjectBoxes() {
  const boxes = [];
  if (state.showBoxes) {
    for (const object of state.objects) {
      if (isViewHidden(object.object_id)) continue;
      const box = state.boxCache.get(object.object_id);
      if (box) boxes.push([box, objectColor(object.object_id), object.label_name]);
    }
  }
  const key = JSON.stringify([state.zoom, boxes]);
  if (key === boxLayerKey) return;
  boxLayerKey = key;

  const percent = (value, total) => `${(value / total) * 100}%`;
  els.boxLayer.replaceChildren();
  for (const [[x, y, width, height], color, text] of boxes) {
    const outline = document.createElement("div");
    outline.className = "box-outline";
    Object.assign(outline.style, {
      left: percent(x, frameWidth),
      top: percent(y, frameHeight),
      // Plus the 2px the line sticks out past the box (margin: -1px).
      width: `calc(${percent(width, frameWidth)} + 2px)`,
      height: `calc(${percent(height, frameHeight)} + 2px)`,
      borderColor: color,
    });
    const tag = document.createElement("div");
    tag.className = "box-tag";
    // Tag sits above the box, or just inside it when the box touches the top edge.
    if (y * state.zoom >= 16) tag.classList.add("above");
    tag.textContent = text;
    Object.assign(tag.style, {
      left: percent(x, frameWidth),
      top: percent(y, frameHeight),
      background: color,
    });
    els.boxLayer.append(outline, tag);
  }
}

// Positive/negative click points, over the canvas like the boxes so they
// stay the same size on screen at any zoom.
let pointLayerKey = "";

function renderPromptPoints() {
  const key = JSON.stringify(state.pendingPoints);
  if (key === pointLayerKey) return;
  pointLayerKey = key;
  els.pointLayer.replaceChildren(...state.pendingPoints.map((point) => {
    const dot = document.createElement("div");
    dot.className = `prompt-point ${point.positive ? "positive" : "negative"}`;
    dot.style.left = `${(point.x / frameWidth) * 100}%`;
    dot.style.top = `${(point.y / frameHeight) * 100}%`;
    return dot;
  }));
}

function maskBoxFromBinaryCanvas(canvas) {
  const { data, width, height } = canvas
    .getContext("2d")
    .getImageData(0, 0, canvas.width, canvas.height);
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < height; y += 1) {
    const row = y * width * 4;
    for (let x = 0; x < width; x += 1) {
      if (data[row + x * 4] > 127) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  return maxX < 0 ? null : [minX, minY, maxX - minX + 1, maxY - minY + 1];
}

function savedGeometryHintPolyline() {
  const object = state.objects.find((o) => o.object_id === state.selectedObjectId);
  return object && object.geometry_hint ? object.geometry_hint.polyline : [];
}

function drawAxisHint(polyline, color) {
  if (!polyline || polyline.length === 0) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  polyline.forEach(([x, y], index) => {
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  for (const [x, y] of polyline) {
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function tintBinaryCanvas(canvas, r, g, b, opacity) {
  const octx = canvas.getContext("2d");
  const src = octx.getImageData(0, 0, canvas.width, canvas.height);
  const out = document.createElement("canvas");
  out.width = canvas.width;
  out.height = canvas.height;
  const outData = out.getContext("2d").createImageData(canvas.width, canvas.height);
  for (let i = 0; i < src.data.length; i += 4) {
    const on = src.data[i] > 127 && src.data[i + 3] > 127;
    outData.data[i] = r;
    outData.data[i + 1] = g;
    outData.data[i + 2] = b;
    outData.data[i + 3] = on ? Math.round(opacity * 255) : 0;
  }
  out.getContext("2d").putImageData(outData, 0, 0);
  return out;
}

/** Colored overlay and tight box of a binary mask canvas in one pixel pass
 * -- no PNG encode/decode round trip, which made each frame load (and so
 * playback) noticeably slower at 1080p. */
function overlayAndBoxFromBinary(binary, [r, g, b], opacity) {
  const { width, height } = binary;
  const src = binary.getContext("2d").getImageData(0, 0, width, height).data;
  const overlay = document.createElement("canvas");
  overlay.width = width;
  overlay.height = height;
  const octx = overlay.getContext("2d");
  const out = octx.createImageData(width, height);
  const dst = out.data;
  const alpha = Math.round(opacity * 255);
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < height; y += 1) {
    let i = y * width * 4;
    for (let x = 0; x < width; x += 1, i += 4) {
      if (src[i] > 127 && src[i + 3] > 127) {
        dst[i] = r;
        dst[i + 1] = g;
        dst[i + 2] = b;
        dst[i + 3] = alpha;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        maxY = y;
      }
    }
  }
  octx.putImageData(out, 0, 0);
  const box = maxX < 0 ? null : [minX, minY, maxX - minX + 1, maxY - minY + 1];
  return { overlay, box };
}

// -------------------------------------------------------- frame loading ----

/** Everything a frame needs from the server: its image, its annotation and
 * each object's mask as a binary canvas. Separate from showing it so
 * playback can fetch the next frame while the current one is on screen. */
async function fetchFrameData(frameIndex) {
  const [image, annotation] = await Promise.all([
    loadImage(`/api/v1/projects/${projectId}/frames/${frameIndex}`),
    apiGet(`/api/v1/projects/${projectId}/frames/${frameIndex}/annotations`),
  ]);
  const masks = new Map();
  await Promise.all(
    annotation.objects.map(async (entry) => {
      // The RLE stored server-side isn't browser-decodable directly, so
      // fetch the dedicated PNG re-encoding (see get_object_mask_png) --
      // this is what makes a reopened project redraw masks saved in an
      // earlier session, not only ones created in the current one.
      let binary = objectMaskCache.get(`${frameIndex}:${entry.object_id}`);
      if (!binary) {
        const maskImage = await loadImage(
          `/api/v1/projects/${projectId}/frames/${frameIndex}/objects/${entry.object_id}/mask.png`
        );
        binary = blankBinaryCanvas();
        binary.getContext("2d").drawImage(maskImage, 0, 0);
      }
      masks.set(entry.object_id, binary);
    })
  );
  return { image, annotation, masks };
}

// Bumped by every loadFrame. Quick navigation (e.g. → then ←) overlaps
// loads, and the older one can finish last; without this it would install
// its frame's annotation and masks over the newer frame's.
let frameLoadGeneration = 0;

async function loadFrame(frameIndex, prefetched = null) {
  const generation = ++frameLoadGeneration;
  conceptSearch.invalidate();
  state.currentFrame = frameIndex;
  state.pendingPoints = [];
  state.pendingAxisPoints = [];
  state.candidate = null;
  els.acceptButton.disabled = true;
  els.saveAxisHintButton.disabled = true;
  els.frameSlider.value = String(frameIndex);
  els.frameNumberInput.value = String(frameIndex);

  const { image, annotation, masks } = await (prefetched || fetchFrameData(frameIndex));
  if (generation !== frameLoadGeneration) return;
  state.frameImage = image;
  state.frameAnnotation = annotation;
  renderReviewStatus();
  els.reviewComment.value = annotation.review_comment || "";

  // From here on everything is synchronous, so no newer load can interleave.
  const overlays = new Map();
  const boxes = new Map();
  for (const [objectId, binary] of masks) {
    cacheMask(`${frameIndex}:${objectId}`, binary);
    const { overlay, box } = overlayAndBoxFromBinary(
      binary, objectColorTuple(objectId), state.maskOpacity
    );
    overlays.set(objectId, overlay);
    if (box) boxes.set(objectId, box);
  }
  state.overlayCache = overlays;
  state.boxCache = boxes;
  await rebuildEditMaskForSelectedObject();
  applyZoom();
  renderObjectList();
  renderOcclusionEdgesSummary();
  renderFrameExcluded();
  render();
  persistUiState({ last_opened_frame: frameIndex });
}

// ------------------------------------------------------------- backends ----

// Which capability each canvas tool needs from the *selected* backend. A
// backend can be available while supporting only a subset of prompt types,
// so availability alone is not enough to decide what the rail may offer.
const TOOL_CAPABILITY = {
  point_positive: "point_prompts",
  point_negative: "point_prompts",
};

const CAPABILITY_LABEL = {
  point_prompts: "point prompts",
  video_propagation: "mask propagation",
};

function selectedBackend() {
  return state.backends.find((b) => b.backend_id === state.selectedBackendId) || null;
}

/** Null when the selected backend supports `capability` (or is not known
 *  yet -- then let the server be the authority); otherwise the sentence to
 *  show the operator. */
function unsupportedReason(capability) {
  const backend = selectedBackend();
  if (!backend || backend.capabilities.includes(capability)) return null;
  const what = CAPABILITY_LABEL[capability] || capability.replace(/_/g, " ");
  return `${backend.display_name} does not support ${what}`;
}

// Grey out what the selected backend cannot do, with the reason as a
// tooltip -- never hide the control, and never let a click turn into a 422
// the operator has to read out of the server log.
function applyBackendCapabilities() {
  conceptSearch.updateCapabilities();
  for (const button of document.querySelectorAll(".tool-button")) {
    const capability = TOOL_CAPABILITY[button.dataset.tool];
    if (!capability) continue;
    if (button.dataset.baseTitle === undefined) button.dataset.baseTitle = button.title;
    const reason = unsupportedReason(capability);
    button.disabled = reason !== null;
    button.title = reason || button.dataset.baseTitle;
  }
  els.propagateButton.disabled =
    state.propagationRunning || unsupportedReason("video_propagation") !== null;
  els.propagateButton.title = unsupportedReason("video_propagation") || "";

  const currentCapability = TOOL_CAPABILITY[state.tool];
  const currentReason = currentCapability ? unsupportedReason(currentCapability) : null;
  if (currentReason) {
    setTool("brush");
    setStatus(`${currentReason} -- switched to the brush; use SAM 2.1 for point prompts`);
  }
}

// Capability-driven control: every registered backend is listed.
// Unavailable ones render disabled with their exact unavailable_reason as
// a tooltip -- never hidden, and never silently swapped for a different
// model.
async function loadBackends() {
  const models = await apiGet("/api/v1/models");
  state.backends = models.backends;
  els.backendSelect.innerHTML = "";
  for (const backend of models.backends) {
    const option = document.createElement("option");
    option.value = backend.backend_id;
    option.textContent = backend.available
      ? backend.display_name
      : `${backend.display_name} (unavailable)`;
    option.disabled = !backend.available;
    if (!backend.available) {
      option.title = backend.unavailable_reason || "unavailable";
    }
    els.backendSelect.appendChild(option);
  }
  if ([...els.backendSelect.options].some((o) => o.value === state.selectedBackendId)) {
    els.backendSelect.value = state.selectedBackendId;
  }
  applyBackendCapabilities();
}

els.backendSelect.addEventListener("change", () => {
  state.selectedBackendId = els.backendSelect.value;
  conceptSearch.invalidate();
  persistUiState({ selected_backend_id: state.selectedBackendId });
  applyBackendCapabilities();
});

// ------------------------------------------------------------- objects ----

async function loadLabelsAndObjects() {
  const [labels, objects] = await Promise.all([
    apiGet(`/api/v1/projects/${projectId}/labels`),
    apiGet(`/api/v1/projects/${projectId}/objects`),
  ]);
  // The ontology is not a fixed enum any more: the operator types any name
  // and it is registered on first use. Existing names are offered as
  // autocomplete suggestions so repeat annotations stay consistent.
  state.labels = labels.labels;
  renderLabelSuggestions();
  renderClassList();
  state.objects = objects.objects;
  state.objectsRevision = objects.revision;
  if (!state.selectedObjectId && state.objects.length > 0) {
    state.selectedObjectId = state.objects[0].object_id;
  }
  renderObjectList();
}

// Lists the objects on this frame (active or deactivated), the selected one
// (so it can be added to this frame) and new ones with no mask yet; the
// rest sit behind the "not on this frame" toggle under the list.
function renderObjectList() {
  els.objectList.innerHTML = "";
  const frameObjects = (state.frameAnnotation && state.frameAnnotation.objects) || [];
  let offFrameCount = 0;
  for (const object of state.objects) {
    const frameEntry = frameObjects.find((o) => o.object_id === object.object_id);
    const inactiveHere = isInactiveOnFrame(object.object_id);
    if (
      !frameEntry && !inactiveHere && object.object_id !== state.selectedObjectId
      && !state.unplacedObjectIds.has(object.object_id)
    ) {
      offFrameCount += 1;
      if (!state.showOffFrameObjects) continue;
    }
    const hiddenHere = Boolean(frameEntry) && isViewHidden(object.object_id);

    const item = document.createElement("li");
    item.className = [
      object.object_id === state.selectedObjectId ? "selected" : "",
      inactiveHere ? "inactive" : "",
      object.locked ? "locked" : "",
    ].join(" ").trim();

    const swatch = document.createElement("span");
    swatch.className = "object-swatch";
    swatch.style.background = objectColor(object.object_id);
    item.appendChild(swatch);

    let stateSuffix = "";
    if (inactiveHere) stateSuffix = " [inactive]";
    else if (frameEntry) stateSuffix = ` [${frameEntry.occlusion_state}${hiddenHere ? ", hidden" : ""}]`;
    const label = document.createElement("span");
    label.textContent = `${object.label_name} (${object.object_id.slice(0, 10)})${stateSuffix}`;
    if (frameEntry && frameEntry.review_reasons && frameEntry.review_reasons.length > 0) {
      label.title = `Review: ${frameEntry.review_reasons.join(", ")}`;
    }
    item.appendChild(label);

    item.addEventListener("click", () => selectObject(object.object_id));

    const button = (text, title, onClick) => {
      const element = document.createElement("button");
      element.type = "button";
      element.textContent = text;
      element.title = title;
      element.addEventListener("click", (event) => {
        event.stopPropagation();
        onClick();
      });
      return element;
    };
    const actions = document.createElement("div");
    actions.className = "object-actions";
    const classButton = button(
      "Class", "Change this object's class on every frame",
      () => editObjectClass(object, label)
    );
    classButton.classList.add("object-class");
    actions.appendChild(classButton);
    const lockButton = button(
      object.locked ? "Unlock" : "Lock",
      object.locked
        ? "Allow deleting this object again"
        : "Keep this object from being deleted, on every frame, until unlocked",
      () => toggleObjectLocked(object.object_id)
    );
    lockButton.classList.add("object-lock");
    lockButton.setAttribute("aria-pressed", String(Boolean(object.locked)));
    actions.appendChild(lockButton);
    const deleteButton = button(
      "Delete",
      object.locked
        ? "Locked -- unlock this object to delete it"
        : "Delete this object from the whole project",
      () => deleteObject(object.object_id)
    );
    deleteButton.classList.add("object-delete");
    deleteButton.disabled = Boolean(object.locked);
    actions.appendChild(deleteButton);
    if (frameEntry || inactiveHere) {
      const activation = button(
        inactiveHere ? "Activate" : "Deactivate",
        inactiveHere
          ? "Restore this object's mask on this frame, so it's exported again (D)"
          : "Remove this object from this frame and its export, e.g. when it's out of view (D)",
        () => toggleObjectActiveOnFrame(object.object_id).catch(
          (error) => setStatus(`Could not (de)activate: ${error.detail || error.message}`)
        )
      );
      activation.classList.add("object-activation");
      actions.appendChild(activation);
    }
    if (frameEntry) {
      const visibility = button(
        hiddenHere ? "Show" : "Hide",
        hiddenHere
          ? "Draw this object's mask on this frame again (H)"
          : "Stop drawing this object's mask on this frame -- view only, still exported (H)",
        () => toggleObjectViewHidden(object.object_id)
      );
      visibility.classList.add("object-visibility");
      actions.appendChild(visibility);
    }
    item.appendChild(actions);

    els.objectList.appendChild(item);
  }
  els.toggleOffFrameObjects.hidden = offFrameCount === 0;
  els.toggleOffFrameObjects.textContent = state.showOffFrameObjects
    ? `Hide ${offFrameCount} not on this frame`
    : `Show ${offFrameCount} not on this frame`;
  populateOccluderSelect();
}

// "Class": swap the object's name for a text box (with the project's classes
// as suggestions) in place. Enter saves, Escape or clicking away cancels.
function editObjectClass(object, labelSpan) {
  const input = document.createElement("input");
  input.type = "text";
  input.className = "object-class-input";
  input.value = object.label_name;
  input.setAttribute("list", "object-label-suggestions");
  input.setAttribute("aria-label", `New class for ${object.label_name}`);
  input.title = "Enter saves the new class on every frame, Escape cancels";
  input.addEventListener("click", (event) => event.stopPropagation());
  let finished = false;
  const finish = (save) => {
    if (finished) return;
    finished = true;
    const labelName = input.value.trim();
    if (save && labelName && labelName !== object.label_name) {
      changeObjectClass(object.object_id, labelName);
    } else {
      renderObjectList();
    }
  };
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      finish(true);
    } else if (event.key === "Escape") {
      event.preventDefault();
      finish(false);
    }
  });
  input.addEventListener("blur", () => finish(false));
  labelSpan.replaceWith(input);
  input.focus();
  input.select();
}

// The server writes the new class onto the object's masks in every frame
// (exports read it from there), and drops the old class if nothing else
// uses it.
async function changeObjectClass(objectId, labelName) {
  const object = state.objects.find((o) => o.object_id === objectId);
  if (!object) return;
  const before = `${object.label_name} (${objectId.slice(0, 10)})`;
  setStatus(`Changing ${before} to ${labelName} on every frame...`);
  try {
    const labelId = await resolveLabelId(labelName);
    const updated = await apiJson(
      "PATCH", `/api/v1/projects/${projectId}/objects/${objectId}`,
      { expected_revision: state.objectsRevision, label_id: labelId, label_name: labelName }
    );
    Object.assign(object, updated);
    state.objectsRevision += 1;
  } catch (error) {
    setStatus(`Could not change the class of ${before}: ${error.detail || error.message}`);
    renderObjectList();
    return;
  }
  // This frame was rewritten too, so reload it rather than keep a stale
  // revision.
  await loadFrame(state.currentFrame);
  panels.refreshReviewQueue();
  await refreshLabels();
  setStatus(`${before} is now ${labelName} on every frame`);
}

async function selectObject(objectId) {
  state.selectedObjectId = objectId;
  state.pendingPoints = [];
  state.candidate = null;
  els.acceptButton.disabled = true;
  renderObjectList();
  await rebuildEditMaskForSelectedObject();
  render();
  persistUiState({ selected_object_id: objectId });
}

async function rebuildEditMaskForSelectedObject() {
  state.undoStack = state.undoStack.filter((entry) => entry.kind !== "mask");
  state.redoStack = state.redoStack.filter((entry) => entry.kind !== "mask");
  updateUndoRedoButtons();
  if (!state.selectedObjectId || !state.frameAnnotation) {
    state.editMaskCanvas = null;
    state.editMaskObjectId = null;
    return;
  }
  const cached = objectMaskCache.get(`${state.currentFrame}:${state.selectedObjectId}`);
  state.editMaskObjectId = state.selectedObjectId;
  state.editMaskCanvas = cached ? cloneCanvas(cached) : blankBinaryCanvas();
}

function cloneCanvas(canvas) {
  const clone = document.createElement("canvas");
  clone.width = canvas.width;
  clone.height = canvas.height;
  clone.getContext("2d").drawImage(canvas, 0, 0);
  return clone;
}

// Per (frame, object) binary mask cache populated whenever we receive a
// mask PNG from the server (segment response or a save echoing it back),
// since the frame-annotation GET response only carries canonical RLE, not
// a browser-renderable PNG (RLE is canonical, PNG is a temporary exchange
// format).
const objectMaskCache = new Map();

// A 1080p mask canvas is ~8 MB, and every frame visited used to stay cached
// forever -- playing a whole video would hold gigabytes. Keep the most
// recently used entries (Map order) plus whatever the current frame needs;
// an evicted mask is simply fetched again when its frame is next shown.
const MASK_CACHE_LIMIT = 48;

function cacheMask(key, canvas) {
  objectMaskCache.delete(key);
  objectMaskCache.set(key, canvas);
  const currentPrefix = `${state.currentFrame}:`;
  for (const oldKey of objectMaskCache.keys()) {
    if (objectMaskCache.size <= MASK_CACHE_LIMIT) break;
    if (!oldKey.startsWith(currentPrefix)) objectMaskCache.delete(oldKey);
  }
}

async function cacheAndOverlayMask(objectId, base64Png) {
  const binary = await binaryCanvasFromMaskBase64(base64Png);
  cacheMask(`${state.currentFrame}:${objectId}`, binary);
  const box = maskBoxFromBinaryCanvas(binary);
  if (box) state.boxCache.set(objectId, box);
  else state.boxCache.delete(objectId);
  const overlay = await coloredOverlayFromMaskBase64(
    base64Png, objectColorTuple(objectId), state.maskOpacity
  );
  state.overlayCache.set(objectId, overlay);
}

function renderLabelSuggestions() {
  const datalist = document.getElementById("object-label-suggestions");
  if (!datalist) return;
  datalist.innerHTML = "";
  for (const label of state.labels) {
    const option = document.createElement("option");
    option.value = label.name;
    datalist.appendChild(option);
  }
}

// The class list: labels.json order is the exported class number (the
// label_id itself never changes, since objects and frames store it), so
// moving a class here renumbers it in every later export. Each row also
// renames the class (its name is a text box), merges it into another class
// or deletes it -- all for every object of the class, on every frame.
function renderClassList() {
  els.classList.innerHTML = "";
  els.classSection.hidden = state.labels.length === 0;
  els.classCount.textContent = `(${state.labels.length})`;
  state.labels.forEach((label, index) => {
    const item = document.createElement("li");
    const number = document.createElement("span");
    number.className = "class-number";
    number.textContent = String(index);
    const name = document.createElement("input");
    name.type = "text";
    name.className = "class-name";
    name.value = label.name;
    name.maxLength = 200;
    name.title = "Rename this class for all its objects (Enter saves, Escape cancels)";
    name.setAttribute("aria-label", `Class name: ${label.name}`);
    name.setAttribute("autocomplete", "off");
    name.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        name.value = label.name;
        name.blur();
      }
    });
    // "change" fires on Enter and on leaving the box with an edited value.
    name.addEventListener("change", () => renameClass(label, name.value));
    item.append(number, name);

    const button = (text, title, onClick) => {
      const element = document.createElement("button");
      element.type = "button";
      element.textContent = text;
      element.title = title;
      element.setAttribute("aria-label", `${title}: ${label.name}`);
      element.addEventListener("click", onClick);
      item.appendChild(element);
      return element;
    };
    for (const [text, offset, title] of [
      ["↑", -1, "Move up (lower class number)"],
      ["↓", 1, "Move down (higher class number)"],
    ]) {
      const target = index + offset;
      button(text, title, () => moveClass(label.label_id, offset)).disabled =
        target < 0 || target >= state.labels.length;
    }
    const merge = button(
      "Merge", "Merge into another class", () => chooseMergeTarget(label, merge)
    );
    merge.classList.add("class-merge");
    merge.disabled = state.labels.length < 2;
    const lockedCount = state.objects.filter(
      (object) => object.label_id === label.label_id && object.locked
    ).length;
    const remove = button(
      "✕",
      lockedCount
        ? `${objectCountText(lockedCount)} of this class locked -- unlock to delete the class`
        : "Delete this class and all its objects",
      () => deleteClass(label)
    );
    remove.classList.add("class-delete");
    remove.disabled = lockedCount > 0;
    els.classList.appendChild(item);
  });
}

function classObjectCount(labelId) {
  return state.objects.filter((object) => object.label_id === labelId).length;
}

function objectCountText(count) {
  return count === 1 ? "1 object" : `${count} objects`;
}

// Runs one whole-class change: the list is locked meanwhile (it can take a
// while on a long video), then objects, classes and this frame are reloaded,
// since the change may have rewritten all of them.
async function runClassChange(busyText, failedText, request, doneText) {
  els.classList.inert = true;
  setStatus(busyText);
  let result = null;
  try {
    result = await request();
    setStatus(doneText(result));
  } catch (error) {
    setStatus(`${failedText}: ${error.detail || error.message}`);
  } finally {
    els.classList.inert = false;
  }
  await reloadObjectsAndFrame();
  return result;
}

// After a change that may have rewritten objects, classes and every frame
// (a class change, a delete or its undo): reload them all, keeping the
// selection if its object is still there.
async function reloadObjectsAndFrame() {
  const objects = await apiGet(`/api/v1/projects/${projectId}/objects`);
  state.objects = objects.objects;
  state.objectsRevision = objects.revision;
  const ids = new Set(state.objects.map((object) => object.object_id));
  if (state.selectedObjectId && !ids.has(state.selectedObjectId)) {
    state.selectedObjectId = state.objects[0]?.object_id || null;
  }
  for (const key of [...objectMaskCache.keys()]) {
    if (!ids.has(key.slice(key.indexOf(":") + 1))) objectMaskCache.delete(key);
  }
  await refreshLabels();
  await loadFrame(state.currentFrame);
  panels.refreshReviewQueue();
}

async function renameClass(label, typedName) {
  const name = typedName.trim();
  if (!name || name === label.name) {
    renderClassList();
    return;
  }
  // Two classes can't share a name, so renaming to an existing one merges.
  const existing = state.labels.find((other) => other.name === name);
  if (existing) {
    await mergeClass(label, existing);
    return;
  }
  await runClassChange(
    `Renaming "${label.name}" to "${name}"...`,
    `Could not rename "${label.name}"`,
    () => apiJson("PATCH", `/api/v1/projects/${projectId}/labels/${label.label_id}`, { name }),
    (result) => `Renamed "${label.name}" to "${name}" (${objectCountText(result.object_count)})`
  );
}

// "Merge" turns into a list of the other classes; picking one merges,
// clicking away or Escape cancels.
function chooseMergeTarget(label, mergeButton) {
  const select = document.createElement("select");
  select.className = "class-merge-select";
  select.setAttribute("aria-label", `Merge ${label.name} into`);
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Merge into...";
  select.appendChild(placeholder);
  for (const other of state.labels) {
    if (other.label_id === label.label_id) continue;
    const option = document.createElement("option");
    option.value = String(other.label_id);
    option.textContent = other.name;
    select.appendChild(option);
  }
  let finished = false;
  const finish = (into) => {
    if (finished) return;
    finished = true;
    if (into) mergeClass(label, into);
    else renderClassList();
  };
  select.addEventListener("change", () => {
    finish(state.labels.find((other) => String(other.label_id) === select.value));
  });
  select.addEventListener("keydown", (event) => {
    if (event.key === "Escape") finish(null);
  });
  select.addEventListener("blur", () => finish(null));
  mergeButton.replaceWith(select);
  select.focus();
}

async function mergeClass(label, into) {
  const count = classObjectCount(label.label_id);
  if (!window.confirm(
    `Merge "${label.name}" into "${into.name}"?\n\n` +
    `Its ${objectCountText(count)} become "${into.name}" on every frame, ` +
    `and "${label.name}" leaves the class list.`
  )) {
    renderClassList();
    return;
  }
  await runClassChange(
    `Merging "${label.name}" into "${into.name}"...`,
    `Could not merge "${label.name}"`,
    () => apiJson("POST", `/api/v1/projects/${projectId}/labels/${label.label_id}/merge`, {
      into_label_id: into.label_id,
    }),
    (result) => `Merged ${objectCountText(result.object_count)} into "${into.name}"`
  );
}

async function deleteClass(label) {
  const count = classObjectCount(label.label_id);
  if (!window.confirm(
    `Delete the class "${label.name}" and its ${objectCountText(count)}?\n\n` +
    "Their masks on every frame are deleted too. Undo (Ctrl+Z) brings them back."
  )) return;
  const result = await runClassChange(
    `Deleting "${label.name}"...`,
    `Could not delete "${label.name}"`,
    () => apiJson("DELETE", `/api/v1/projects/${projectId}/labels/${label.label_id}`),
    (result) => `Deleted "${label.name}" and ${objectCountText(result.object_count)} ` +
      "-- Undo (Ctrl+Z) brings them back"
  );
  if (result?.deleted_object_ids.length) {
    pushDeletion(result.deleted_object_ids, `the class "${label.name}"`);
  }
}

// The server drops a class once its last object is deleted or relabeled.
async function refreshLabels() {
  try {
    state.labels = (await apiGet(`/api/v1/projects/${projectId}/labels`)).labels;
  } catch (error) {
    return; // the list is only stale until the next reload
  }
  renderLabelSuggestions();
  renderClassList();
}

// PUT replaces the whole list, so start from the server's copy: a label
// another tab registered meanwhile must not be dropped by a stale reorder.
async function moveClass(labelId, offset) {
  try {
    const latest = (await apiGet(`/api/v1/projects/${projectId}/labels`)).labels;
    const from = latest.findIndex((label) => label.label_id === labelId);
    const to = from + offset;
    if (from >= 0 && to >= 0 && to < latest.length) {
      const [moved] = latest.splice(from, 1);
      latest.splice(to, 0, moved);
      const updated = await apiJson("PUT", `/api/v1/projects/${projectId}/labels`, {
        labels: latest,
      });
      state.labels = updated.labels;
      setStatus(`"${moved.name}" is now class ${to}.`);
    } else {
      state.labels = latest;
    }
  } catch (error) {
    setStatus(`Could not reorder classes: ${error.detail || error.message}`);
  }
  renderLabelSuggestions();
  renderClassList();
}

// Resolve a free-text object name to a label_id, registering a brand new
// label in the project ontology when the name has not been used before.
// label_id is the label's stable identity (objects and frames store it), so
// it must be unique and never reassigned; the exported class number is its
// position in the list instead (see renderClassList).
async function resolveLabelId(labelName) {
  const existing = state.labels.find((label) => label.name === labelName);
  if (existing) {
    return existing.label_id;
  }
  // Append to the server's list, not ours, so a reorder made in another
  // tab is kept.
  const latest = (await apiGet(`/api/v1/projects/${projectId}/labels`)).labels;
  const already = latest.find((label) => label.name === labelName);
  let labelId;
  if (already) {
    labelId = already.label_id;
    state.labels = latest;
  } else {
    labelId = latest.reduce((max, label) => Math.max(max, label.label_id), -1) + 1;
    const updated = await apiJson("PUT", `/api/v1/projects/${projectId}/labels`, {
      labels: [...latest, { label_id: labelId, name: labelName }],
    });
    state.labels = updated.labels;
  }
  renderLabelSuggestions();
  renderClassList();
  return labelId;
}

async function createObject() {
  const labelName = els.newObjectLabel.value.trim();
  if (!labelName) {
    setStatus("Enter an object name first");
    els.newObjectLabel.focus();
    return;
  }
  const labelId = await resolveLabelId(labelName);
  const created = await apiJson("POST", `/api/v1/projects/${projectId}/objects`, {
    label_id: labelId,
    label_name: labelName,
  });
  state.objects.push(created);
  state.unplacedObjectIds.add(created.object_id);
  state.objectsRevision += 1;
  els.newObjectLabel.value = "";
  await selectObject(created.object_id);
  setStatus(`Created object ${labelName} (${created.object_id})`);
}

// Lock/Unlock: a locked object can't be deleted, alone or with its class.
// It's a property of the object, so it holds on every frame.
async function toggleObjectLocked(objectId) {
  const object = state.objects.find((o) => o.object_id === objectId);
  if (!object) return;
  const name = `${object.label_name} (${objectId.slice(0, 10)})`;
  const locked = !object.locked;
  try {
    const updated = await apiJson(
      "PATCH", `/api/v1/projects/${projectId}/objects/${objectId}`,
      { expected_revision: state.objectsRevision, locked }
    );
    Object.assign(object, updated);
    state.objectsRevision += 1;
  } catch (error) {
    setStatus(`Could not ${locked ? "lock" : "unlock"} ${name}: ${error.detail || error.message}`);
    return;
  }
  renderObjectList();
  renderClassList();
  setStatus(locked ? `Locked ${name} -- it can't be deleted until unlocked` : `Unlocked ${name}`);
}

async function deleteObject(objectId) {
  const object = state.objects.find((o) => o.object_id === objectId);
  const name = object ? `${object.label_name} (${objectId.slice(0, 10)})` : objectId;
  if (object?.locked) {
    setStatus(`${name} is locked -- unlock it to delete it`);
    return;
  }
  if (!window.confirm(
    `Delete ${name} from the whole project?\n\n` +
    "Its masks on every frame are deleted too. Undo (Ctrl+Z) brings it back."
  )) return;
  if (await deleteObjects([objectId], name)) {
    pushDeletion([objectId], name);
    setStatus(`Deleted ${name} -- Undo (Ctrl+Z) brings it back`);
  }
}

// Deletes without asking: Delete has asked already, and Redo repeats a
// delete that was undone. The server keeps what it removes, for Undo.
async function deleteObjects(objectIds, name) {
  setStatus(`Deleting ${name}...`);
  let failure = null;
  try {
    for (const objectId of objectIds) {
      await apiJson(
        "DELETE",
        `/api/v1/projects/${projectId}/objects/${objectId}?expected_revision=${state.objectsRevision}`
      );
      state.objectsRevision += 1;
    }
  } catch (error) {
    failure = error;
  }
  // The delete rewrote every frame the object was on, this one included,
  // so reload it rather than keep a stale revision.
  await reloadObjectsAndFrame();
  if (failure) setStatus(`Could not delete ${name}: ${failure.detail || failure.message}`);
  return !failure;
}

function pushDeletion(objectIds, name) {
  state.undoStack.push({ kind: "deleted", objectIds, name });
  state.redoStack = [];
  updateUndoRedoButtons();
}

// Undo of a delete: the server puts the objects back with their masks on
// every frame and their class. The first is selected, so it's listed here.
async function restoreDeleted(entry) {
  setStatus(`Restoring ${entry.name}...`);
  try {
    await apiJson("POST", `/api/v1/projects/${projectId}/objects/restore`, {
      object_ids: entry.objectIds,
    });
  } catch (error) {
    setStatus(`Could not restore ${entry.name}: ${error.detail || error.message}`);
    return false;
  }
  state.selectedObjectId = entry.objectIds[0];
  persistUiState({ selected_object_id: state.selectedObjectId });
  await reloadObjectsAndFrame();
  setStatus(`Restored ${entry.name} (Redo deletes it again)`);
  return true;
}

function isInactiveOnFrame(objectId) {
  return (state.frameAnnotation?.inactive_objects || []).some((o) => o.object_id === objectId);
}

function isViewHidden(objectId) {
  return state.viewHidden.has(`${state.currentFrame}:${objectId}`);
}

// Hide/Show (H): only stops drawing the object's mask on this frame, e.g. to
// see what's underneath. Nothing is saved; the export is unchanged.
function toggleObjectViewHidden(objectId) {
  if (!state.frameAnnotation?.objects.some((o) => o.object_id === objectId)) return;
  const key = `${state.currentFrame}:${objectId}`;
  const hide = !state.viewHidden.has(key);
  if (hide) state.viewHidden.add(key);
  else state.viewHidden.delete(key);
  renderObjectList();
  render();
  setStatus(`${hide ? "Hidden" : "Shown"} on frame ${state.currentFrame} (view only)`);
}

// Deactivate/Activate (D) an object on this frame only (e.g. out of view):
// an inactive object exports no box here, unlike erasing it, which leaves an
// empty mask behind. The server keeps its mask, so Activate restores it.
async function toggleObjectActiveOnFrame(objectId) {
  const inactive = isInactiveOnFrame(objectId);
  const onFrame = state.frameAnnotation?.objects.some((o) => o.object_id === objectId);
  if (!inactive && !onFrame) return;
  const frameIndex = state.currentFrame;
  const updated = await apiJson(
    "PATCH", `/api/v1/projects/${projectId}/frames/${frameIndex}/annotations`,
    {
      expected_revision: state.frameAnnotation.revision,
      [inactive ? "activate_object_id" : "deactivate_object_id"]: objectId,
    }
  );
  state.frameAnnotation = updated;
  renderReviewStatus();
  state.viewHidden.delete(`${frameIndex}:${objectId}`);
  const cacheKey = `${frameIndex}:${objectId}`;
  if (inactive) {
    const image = await loadImage(
      `/api/v1/projects/${projectId}/frames/${frameIndex}/objects/${objectId}/mask.png`
    );
    const binary = blankBinaryCanvas();
    binary.getContext("2d").drawImage(image, 0, 0);
    cacheMask(cacheKey, binary);
    await cacheAndOverlayMask(objectId, canvasToBinaryMaskPngBase64(cloneCanvas(binary)));
  } else {
    objectMaskCache.delete(cacheKey);
    state.overlayCache.delete(objectId);
    state.boxCache.delete(objectId);
  }
  if (state.selectedObjectId === objectId) await rebuildEditMaskForSelectedObject();
  renderObjectList();
  render();
  panels.refreshReviewQueue();
  setStatus(
    `${inactive ? "Activated" : "Deactivated"} on frame ${frameIndex} (revision ${updated.revision})`
  );
}

// "Delete frame": the frame drops out of exports and the review queue; its
// masks stay, so Restore is lossless (see FrameAnnotation.excluded).
async function toggleFrameExcluded() {
  const excluded = !state.frameAnnotation.excluded;
  const updated = await apiJson(
    "PATCH", `/api/v1/projects/${projectId}/frames/${state.currentFrame}/annotations`,
    { expected_revision: state.frameAnnotation.revision, excluded }
  );
  state.frameAnnotation = updated;
  renderFrameExcluded();
  panels.refreshReviewQueue();
  setStatus(excluded
    ? `Frame ${state.currentFrame} deleted from the dataset (Restore to undo)`
    : `Frame ${state.currentFrame} restored`);
}

function renderFrameExcluded() {
  const excluded = Boolean(state.frameAnnotation?.excluded);
  els.deleteFrameButton.textContent = excluded ? "Restore frame" : "Delete frame";
  els.deleteFrameButton.classList.toggle("lv-frame-restore", excluded);
  els.frameExcludedBadge.hidden = !excluded;
  els.canvasScroll.classList.toggle("frame-excluded", excluded);
}

async function saveReviewStatus(reviewStatus, comment = undefined) {
  const updated = await apiJson(
    "PATCH", `/api/v1/projects/${projectId}/frames/${state.currentFrame}/annotations`,
    {
      expected_revision: state.frameAnnotation.revision,
      review_status: reviewStatus,
      ...(comment === undefined ? {} : { review_comment: comment }),
    }
  );
  state.frameAnnotation = updated;
  renderReviewStatus();
  panels.refreshReviewQueue();
  return updated;
}

// The frame's review status, in the Review tab and at the right end of the
// bottom bar. Called whenever the frame's annotation comes back from the
// server, since saving a mask can mark the frame "needs review".
function renderReviewStatus() {
  const status = state.frameAnnotation?.review_status || "unreviewed";
  els.reviewStatus.value = status;
  els.barReviewStatus.value = status;
  els.barReviewStatus.dataset.status = status;
}

// ---------------------------------------------------------- persistence ----

async function upsertObjectMask(objectId, base64Png, source, {
  backendId, manuallyModified, confidence = null, provenance = null,
} = {}) {
  const payload = {
    expected_revision: state.frameAnnotation.revision,
    upsert_object: {
      object_id: objectId,
      label_id: state.objects.find((o) => o.object_id === objectId)?.label_id ?? 0,
      visible_mask_png_base64: base64Png,
      source,
      manually_modified: Boolean(manuallyModified),
      backend_id: backendId || null,
      confidence,
      model_name: provenance?.model_name || null,
      checkpoint_sha256: provenance?.checkpoint_sha256 || null,
      code_revision: provenance?.code_revision || null,
    },
  };
  const updated = await apiJson(
    "PATCH", `/api/v1/projects/${projectId}/frames/${state.currentFrame}/annotations`, payload
  );
  state.frameAnnotation = updated;
  renderReviewStatus();
  state.unplacedObjectIds.delete(objectId);
  await cacheAndOverlayMask(objectId, base64Png);
  // A freshly saved mask should be seen, not stay view-hidden.
  state.viewHidden.delete(`${state.currentFrame}:${objectId}`);
  // The list shows per-frame state (occlusion, Deactivate/Hide), now changed.
  renderObjectList();
  render();
  setStatus(`Saved (revision ${updated.revision})`);
}

let uiStateSaveTimer = null;
function persistUiState(fields) {
  clearTimeout(uiStateSaveTimer);
  uiStateSaveTimer = setTimeout(async () => {
    try {
      const updated = await apiJson("PATCH", `/api/v1/projects/${projectId}`, {
        expected_revision: state.projectRevision,
        ...fields,
      });
      state.projectRevision = updated.revision;
    } catch (error) {
      if (error.status === 409) {
        const current = await apiGet(`/api/v1/projects/${projectId}`);
        state.projectRevision = current.revision;
      }
    }
  }, 300);
}

// --------------------------------------------------------------- tools ----

function setTool(tool) {
  // Keyboard shortcuts bypass the disabled buttons, so the capability gate
  // has to live here too rather than only on the rail.
  const capability = TOOL_CAPABILITY[tool];
  const reason = capability ? unsupportedReason(capability) : null;
  if (reason) {
    setStatus(`${reason} -- use SAM 2.1 for point prompts`);
    return;
  }
  state.tool = tool;
  for (const button of document.querySelectorAll(".tool-button")) {
    button.classList.toggle("active", button.dataset.tool === tool);
  }
  renderBrushCursor();
}

// Brush and eraser show their footprint instead of the crosshair: a ring
// over the canvas, sized in frame pixels like the stroke, so it follows zoom.
function renderBrushCursor() {
  const painting = state.tool === "brush" || state.tool === "eraser";
  els.canvas.classList.toggle("brush-cursor-active", painting);
  const point = state.hoverPoint;
  els.brushCursor.hidden = !painting || !point;
  if (els.brushCursor.hidden) return;
  const radius = state.brushRadius;
  els.brushCursor.classList.toggle(
    "eraser", state.isPainting ? state.strokeErases : state.tool === "eraser"
  );
  els.brushCursor.style.left = `${((point.x - radius) / frameWidth) * 100}%`;
  els.brushCursor.style.top = `${((point.y - radius) / frameHeight) * 100}%`;
  els.brushCursor.style.width = `${((2 * radius) / frameWidth) * 100}%`;
  els.brushCursor.style.height = `${((2 * radius) / frameHeight) * 100}%`;
}

function imageCoordsFromEvent(event) {
  const rect = els.canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * frameWidth,
    y: ((event.clientY - rect.top) / rect.height) * frameHeight,
  };
}

async function handlePointClick(x, y, positive, addComponent) {
  if (!state.selectedObjectId) {
    setStatus("Select or create an object first");
    return;
  }
  const unsupported = unsupportedReason("point_prompts");
  if (unsupported) {
    setStatus(`${unsupported} -- use SAM 2.1 for point prompts`);
    return;
  }
  state.pendingPoints.push({ x, y, positive });
  render();

  let previousMaskBase64 = null;
  if (addComponent) {
    const cached = objectMaskCache.get(`${state.currentFrame}:${state.selectedObjectId}`);
    if (cached) previousMaskBase64 = canvasToBinaryMaskPngBase64(cloneCanvas(cached));
  }

  let response;
  try {
    response = await apiJson("POST", `/api/v1/projects/${projectId}/segment`, {
      backend_id: state.selectedBackendId,
      frame_index: state.currentFrame,
      object_id: state.selectedObjectId,
      points: state.pendingPoints,
      previous_mask_png_base64: previousMaskBase64,
    });
  } catch (error) {
    // Drop the point again: it produced no candidate, so leaving it on the
    // canvas would make the next click send a prompt the operator never saw
    // rejected.
    state.pendingPoints.pop();
    render();
    setStatus(`Segmentation failed: ${error.detail || error.message}`);
    return;
  }

  const overlayCanvas = await coloredOverlayFromMaskBase64(
    response.visible_mask_png_base64, objectColorTuple(state.selectedObjectId), 0.8
  );
  state.candidate = {
    objectId: state.selectedObjectId,
    base64: response.visible_mask_png_base64,
    overlayCanvas,
    componentCount: response.component_count,
  };
  els.acceptButton.disabled = false;
  setStatus(`Candidate mask: ${response.component_count} component(s), score ${response.score ?? "n/a"}`);
  render();
}

async function acceptCandidate() {
  if (!state.candidate) return;
  const backendId = state.selectedBackendId;
  await upsertObjectMask(
    state.candidate.objectId, state.candidate.base64, `${backendId}_interactive`,
    { backendId, manuallyModified: false }
  );
  state.pendingPoints = [];
  state.candidate = null;
  els.acceptButton.disabled = true;
  await rebuildEditMaskForSelectedObject();
  render();
}

function clearPendingPoints() {
  state.pendingPoints = [];
  state.candidate = null;
  els.acceptButton.disabled = true;
  render();
}

function paintAt(x, y, erase) {
  const canvas = state.editMaskCanvas;
  const octx = canvas.getContext("2d");
  octx.beginPath();
  octx.arc(x, y, state.brushRadius, 0, Math.PI * 2);
  if (erase) {
    octx.globalCompositeOperation = "destination-out";
    octx.fillStyle = "rgba(255,255,255,1)";
  } else {
    octx.globalCompositeOperation = "source-over";
    octx.fillStyle = "white";
  }
  octx.fill();
  octx.globalCompositeOperation = "source-over";
  editMaskVersion += 1;
}

function pushUndoSnapshot() {
  if (!state.editMaskCanvas) return;
  state.undoStack.push({ kind: "mask", canvas: cloneCanvas(state.editMaskCanvas) });
  // The depth caps strokes (each a full-frame canvas), never deletes.
  const strokes = state.undoStack.filter((entry) => entry.kind === "mask");
  if (strokes.length > UNDO_DEPTH) {
    state.undoStack.splice(state.undoStack.indexOf(strokes[0]), 1);
  }
  state.redoStack = [];
  updateUndoRedoButtons();
}

function updateUndoRedoButtons() {
  els.undoButton.disabled = state.undoStack.length === 0;
  els.redoButton.disabled = state.redoStack.length === 0;
}

// Undo/redo themselves are local, but each also re-commits the restored
// raster so the server-side "current mask" never silently diverges from
// what the user sees on screen.
// A delete is undone by the server restoring the objects; redone by
// deleting them again.
function undo() {
  stepHistory("undoStack", "redoStack", (entry) => restoreDeleted(entry));
}

function redo() {
  stepHistory("redoStack", "undoStack", async (entry) => {
    const done = await deleteObjects(entry.objectIds, entry.name);
    if (done) setStatus(`Deleted ${entry.name} again -- Undo (Ctrl+Z) brings it back`);
    return done;
  });
}

// Stacks are named, not held: reloading a frame replaces them (it drops
// the strokes), and a delete's undo/redo reloads the frame.
async function stepHistory(fromKey, toKey, applyDeletion) {
  const entry = state[fromKey][state[fromKey].length - 1];
  if (!entry || state.historyBusy) return;
  if (entry.kind === "deleted") {
    state.historyBusy = true;
    try {
      if (await applyDeletion(entry)) {
        state[fromKey] = state[fromKey].filter((other) => other !== entry);
        state[toKey].push(entry);
      }
    } finally {
      state.historyBusy = false;
      updateUndoRedoButtons();
    }
    return;
  }
  if (!state.editMaskCanvas) return;
  state[fromKey].pop();
  state[toKey].push({ kind: "mask", canvas: cloneCanvas(state.editMaskCanvas) });
  state.editMaskCanvas = entry.canvas;
  updateUndoRedoButtons();
  render();
  commitBrushStroke();
}

async function commitBrushStroke(erased = state.tool === "eraser") {
  if (!state.selectedObjectId || !state.editMaskCanvas) return;
  const base64 = canvasToBinaryMaskPngBase64(cloneCanvas(state.editMaskCanvas));
  await upsertObjectMask(
    state.selectedObjectId, base64,
    erased ? "manual_eraser" : "manual_brush",
    { manuallyModified: true }
  );
}

// ------------------------------------------------------------- pointer ----

els.canvas.addEventListener("pointerdown", (event) => {
  // Editing applies to the frame on screen, so stop on it first.
  stopPlayback();
  const { x, y } = imageCoordsFromEvent(event);
  if (state.tool === "point_positive" || state.tool === "point_negative") {
    handlePointClick(x, y, state.tool === "point_positive", event.shiftKey);
    return;
  }
  if (state.tool === "axis_hint") {
    if (!state.selectedObjectId) {
      setStatus("Select or create an object first");
      return;
    }
    state.pendingAxisPoints.push({ x, y });
    els.saveAxisHintButton.disabled = state.pendingAxisPoints.length < 2;
    render();
    return;
  }
  if (state.tool === "brush" || state.tool === "eraser") {
    if (!state.selectedObjectId) {
      setStatus("Select or create an object first");
      return;
    }
    pushUndoSnapshot();
    state.isPainting = true;
    state.strokeErases = state.tool === "eraser" || event.button === 2;
    paintAt(x, y, state.strokeErases);
    render();
    renderBrushCursor();
  }
});

// Right-drag erases with the brush, so the menu must not open over it.
els.canvas.addEventListener("contextmenu", (event) => {
  if (state.tool === "brush" || state.tool === "eraser") event.preventDefault();
});

els.canvas.addEventListener("pointermove", (event) => {
  const { x, y } = imageCoordsFromEvent(event);
  state.hoverPoint = { x, y };
  renderBrushCursor();
  if (!state.isPainting) return;
  paintAt(x, y, state.strokeErases);
  render();
});

els.canvas.addEventListener("pointerleave", () => {
  state.hoverPoint = null;
  renderBrushCursor();
});

window.addEventListener("pointerup", () => {
  if (state.isPainting) {
    state.isPainting = false;
    commitBrushStroke(state.strokeErases);
    renderBrushCursor();
  }
});

// -------------------------------------------------------- frame nav UI ----

// ------------------------------------------------------------- playback ----

// Plays forward from the current frame through the normal loadFrame, so
// masks/boxes look exactly as when stepping. Each frame waits for its own
// load, so a slow load lowers the frame rate rather than queueing frames
// up; the achieved rate is shown next to the speed picker.
let playbackRun = 0;

function isPlaying() {
  return els.playButton.getAttribute("aria-pressed") === "true";
}

function setPlayingUi(playing) {
  els.playButton.setAttribute("aria-pressed", String(playing));
  els.playButton.title = playing ? "Pause (Space)" : "Play (Space)";
  els.playButton.setAttribute("aria-label", playing ? "Pause" : "Play");
}

function stopPlayback() {
  if (!isPlaying()) return;
  playbackRun += 1;
  setPlayingUi(false);
  setStatus(`Paused at frame ${state.currentFrame}`);
}

async function startPlayback() {
  if (isPlaying() || conceptSearch.isAccepting()) return;
  // From the last frame, Play starts over rather than doing nothing.
  if (state.currentFrame >= frameCount - 1) await loadFrame(0);
  const run = ++playbackRun;
  setPlayingUi(true);
  setStatus("Playing");
  let shown = 0;
  const startedAt = performance.now();
  // The next frame is always already being fetched while the current one
  // is drawn and held on screen.
  const prefetch = (index) => {
    if (index >= frameCount) return null;
    const pending = fetchFrameData(index);
    pending.catch(() => {}); // surfaced by loadFrame's await, if it's used
    return pending;
  };
  let next = prefetch(state.currentFrame + 1);
  while (run === playbackRun) {
    if (state.currentFrame >= frameCount - 1) {
      playbackRun += 1;
      setPlayingUi(false);
      setStatus(`End of video (frame ${state.currentFrame})`);
      return;
    }
    const frameStarted = performance.now();
    const target = state.currentFrame + 1;
    const data = next;
    next = prefetch(target + 1);
    await loadFrame(target, data);
    if (run !== playbackRun) return;
    shown += 1;
    const elapsedSeconds = (performance.now() - startedAt) / 1000;
    if (shown % 5 === 0) setStatus(`Playing · ${(shown / elapsedSeconds).toFixed(1)} fps`);
    const interval = 1000 / (videoFps * parseFloat(els.playbackSpeed.value));
    const wait = interval - (performance.now() - frameStarted);
    if (wait > 0) await sleep(wait);
  }
}

function togglePlayback() {
  if (isPlaying()) stopPlayback();
  else startPlayback().catch((error) => {
    stopPlayback();
    setStatus(`Playback stopped: ${error.message}`);
  });
}

els.playButton.addEventListener("click", togglePlayback);
// Hand focus back after picking a speed, so Space toggles playback instead of
// reopening the dropdown.
els.playbackSpeed.addEventListener("change", () => els.playbackSpeed.blur());

async function goToFrame(index) {
  // Any manual navigation takes over from playback.
  stopPlayback();
  if (conceptSearch.isAccepting()) {
    setStatus("Wait for the text segmentation to finish saving");
    return;
  }
  await loadFrame(clampFrame(index));
}

els.prevFrameButton.addEventListener("click", () => goToFrame(state.currentFrame - 1));
els.nextFrameButton.addEventListener("click", () => goToFrame(state.currentFrame + 1));
els.frameSlider.addEventListener("change", () => goToFrame(parseInt(els.frameSlider.value, 10)));
els.frameNumberInput.addEventListener("change", () =>
  goToFrame(parseInt(els.frameNumberInput.value, 10))
);

// ------------------------------------------------------------- toolbar ----

for (const button of document.querySelectorAll(".tool-button")) {
  button.addEventListener("click", () => setTool(button.dataset.tool));
}
els.brushRadius.addEventListener("input", () => {
  state.brushRadius = parseInt(els.brushRadius.value, 10);
  renderBrushCursor();
});
els.maskOpacity.addEventListener("input", async () => {
  state.maskOpacity = parseInt(els.maskOpacity.value, 10) / 100;
  for (const [objectId] of state.overlayCache) {
    const cached = objectMaskCache.get(`${state.currentFrame}:${objectId}`);
    if (cached) {
      const base64 = canvasToBinaryMaskPngBase64(cloneCanvas(cached));
      await cacheAndOverlayMask(objectId, base64);
    }
  }
  render();
});
els.toggleOffFrameObjects.addEventListener("click", () => {
  state.showOffFrameObjects = !state.showOffFrameObjects;
  renderObjectList();
});
els.showBoxes.checked = state.showBoxes;
els.showBoxes.addEventListener("change", () => {
  state.showBoxes = els.showBoxes.checked;
  try {
    window.localStorage.setItem(SHOW_BOXES_STORAGE_KEY, state.showBoxes ? "1" : "0");
  } catch (error) {
    // Storage unavailable: the toggle still works, it is just not remembered.
  }
  render();
});
els.zoomSlider.addEventListener("input", () => {
  state.zoom = parseInt(els.zoomSlider.value, 10) / 100;
  applyZoom();
  if (state.showBoxes) renderObjectBoxes(); // tag placement depends on zoom
});

// Goes through the slider so both wheel actions share its limits and its
// input handler.
function stepSlider(slider, amount) {
  slider.value = parseInt(slider.value, 10) + amount;
  slider.dispatchEvent(new Event("input"));
}

// Keeps the frame point under the pointer in place while the canvas resizes.
function zoomAtPointer(event, amount) {
  const { x, y } = imageCoordsFromEvent(event);
  stepSlider(els.zoomSlider, amount);
  const rect = els.canvas.getBoundingClientRect();
  els.canvasScroll.scrollLeft += rect.left + (x / frameWidth) * rect.width - event.clientX;
  els.canvasScroll.scrollTop += rect.top + (y / frameHeight) * rect.height - event.clientY;
}

// Not passive: Ctrl+wheel would otherwise zoom the browser page.
els.canvasScroll.addEventListener("wheel", (event) => {
  if (!event.ctrlKey && !event.shiftKey) return;
  event.preventDefault();
  // Browsers turn Shift+wheel into horizontal scrolling, so the delta can
  // arrive on either axis.
  const direction = -Math.sign(event.deltaY || event.deltaX);
  if (event.ctrlKey) {
    zoomAtPointer(event, direction * WHEEL_ZOOM_STEP);
  } else {
    stepSlider(els.brushRadius, direction * WHEEL_BRUSH_STEP);
  }
}, { passive: false });

els.undoButton.addEventListener("click", undo);
els.redoButton.addEventListener("click", redo);
els.acceptButton.addEventListener("click", acceptCandidate);
els.clearPointsButton.addEventListener("click", clearPendingPoints);
els.addObjectButton.addEventListener("click", createObject);

// Enter in the name field creates the object -- the field is a plain text
// input, so without this it would just do nothing (there is no form).
els.newObjectLabel.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    createObject();
  }
});

// Project status: an operator-set workflow label, deliberately independent
// of per-frame review. Rides the same revision-checked PATCH as ui_state.
const statusSelect = document.getElementById("project-status-select");
if (statusSelect) {
  statusSelect.addEventListener("change", () => {
    statusSelect.className = `lv-status-select lv-status-${statusSelect.value}`;
    persistUiState({ status: statusSelect.value });
    setStatus(`Project marked "${statusSelect.selectedOptions[0].textContent}"`);
  });
}

const exportButton = document.getElementById("open-export-button");
if (exportButton) {
  exportButton.addEventListener("click", () => {
    openExportDialog(projectId, exportButton.dataset.projectTitle);
  });
}

// Sidebar tabs (Objects / Review).
const tabButtons = Array.from(document.querySelectorAll(".lv-tab"));
for (const tab of tabButtons) {
  tab.addEventListener("click", () => {
    for (const other of tabButtons) {
      const selected = other === tab;
      other.setAttribute("aria-selected", String(selected));
      document.getElementById(other.getAttribute("aria-controls")).hidden = !selected;
    }
  });
}

els.saveReviewButton.addEventListener("click", async () => {
  const updated = await saveReviewStatus(els.reviewStatus.value, els.reviewComment.value);
  setStatus(`Review saved (revision ${updated.revision})`);
});

els.deleteFrameButton.addEventListener("click", toggleFrameExcluded);

// Unlike the Review tab (status and comment, then Save review), the bar's
// status is saved as soon as it is picked; the comment stays as it is.
els.barReviewStatus.addEventListener("change", async () => {
  const status = els.barReviewStatus.value;
  const name = els.barReviewStatus.selectedOptions[0].textContent;
  try {
    await saveReviewStatus(status);
    setStatus(`Frame ${state.currentFrame}: ${name}`);
  } catch (error) {
    renderReviewStatus();
    setStatus(`Could not save the review status: ${error.detail || error.message}`);
  }
});

// The queue's "Solved" button. The current frame goes through the canvas's
// own state so its revision stays in step; any other frame is read fresh.
async function markFrameReviewed(frameIndex) {
  if (frameIndex === state.currentFrame) {
    await saveReviewStatus("reviewed");
  } else {
    const frame = await apiGet(`/api/v1/projects/${projectId}/frames/${frameIndex}/annotations`);
    await apiJson(
      "PATCH", `/api/v1/projects/${projectId}/frames/${frameIndex}/annotations`,
      { expected_revision: frame.revision, review_status: "reviewed" }
    );
  }
  setStatus(`Frame ${frameIndex} marked reviewed`);
}

// ---------------------------------------------- axis hint / occluder ----

els.saveAxisHintButton.addEventListener("click", async () => {
  if (!state.selectedObjectId || state.pendingAxisPoints.length < 2) return;
  const object = state.objects.find((o) => o.object_id === state.selectedObjectId);
  const updated = await apiJson(
    "PATCH", `/api/v1/projects/${projectId}/objects/${state.selectedObjectId}`,
    {
      expected_revision: state.objectsRevision,
      geometry_hint: {
        type: "elongated_axis",
        polyline: state.pendingAxisPoints.map((p) => [p.x, p.y]),
      },
    }
  );
  state.objectsRevision += 1;
  const index = state.objects.findIndex((o) => o.object_id === state.selectedObjectId);
  if (index >= 0) state.objects[index] = updated;
  state.pendingAxisPoints = [];
  els.saveAxisHintButton.disabled = true;
  render();
  setStatus(`Saved axis hint for ${object ? object.label_name : updated.object_id}`);
});

els.deleteAxisHintButton.addEventListener("click", async () => {
  if (!state.selectedObjectId) return;
  const object = state.objects.find((o) => o.object_id === state.selectedObjectId);
  const updated = await apiJson(
    "PATCH", `/api/v1/projects/${projectId}/objects/${state.selectedObjectId}`,
    { expected_revision: state.objectsRevision, clear_geometry_hint: true }
  );
  state.objectsRevision += 1;
  const index = state.objects.findIndex((o) => o.object_id === updated.object_id);
  if (index >= 0) state.objects[index] = updated;
  render();
  setStatus(`Deleted axis hint for ${object ? object.label_name : updated.object_id}`);
});

function populateOccluderSelect() {
  els.occluderSelect.innerHTML = "";
  for (const object of state.objects) {
    if (object.object_id === state.selectedObjectId) continue;
    const option = document.createElement("option");
    option.value = object.object_id;
    option.textContent = object.label_name;
    els.occluderSelect.appendChild(option);
  }
}

function renderOcclusionEdgesSummary() {
  const edges = (state.frameAnnotation && state.frameAnnotation.occlusion_edges) || [];
  if (edges.length === 0) {
    els.occlusionEdgesSummary.textContent = "";
    return;
  }
  const labelFor = (objectId) =>
    (state.objects.find((o) => o.object_id === objectId) || {}).label_name || objectId;
  els.occlusionEdgesSummary.textContent = edges
    .map((edge) => `${labelFor(edge.occluder_object_id)} in front of ${labelFor(edge.occluded_object_id)}`)
    .join("; ");
}

els.markOccluderButton.addEventListener("click", async () => {
  const occluderId = els.occluderSelect.value;
  if (!state.selectedObjectId || !occluderId) {
    setStatus("Select an object and an occluder first");
    return;
  }
  const existingEdges = (state.frameAnnotation.occlusion_edges || []).filter(
    (edge) => edge.occluded_object_id !== state.selectedObjectId
  );
  const newEdges = [
    ...existingEdges,
    {
      occluder_object_id: occluderId,
      occluded_object_id: state.selectedObjectId,
      relation: "in_front_of",
    },
  ];
  const updated = await apiJson(
    "PATCH", `/api/v1/projects/${projectId}/frames/${state.currentFrame}/annotations`,
    { expected_revision: state.frameAnnotation.revision, occlusion_edges: newEdges }
  );
  state.frameAnnotation = updated;
  renderOcclusionEdgesSummary();
  setStatus(`Marked occluder (revision ${updated.revision})`);
});

// -------------------------------------------------------- propagation ----

// Track every *active* object that has a saved mask on the current frame
// (the server re-seeds its backend session from those masks) forward or
// backward using the selected backend, polling the background job
// (chunked/checkpointed server-side) until it reaches a terminal state,
// then reloading the current frame so any mask change lands on screen
// immediately. The object list's
// Activate/Deactivate toggle is what includes/excludes an object here --
// deactivating hides its overlay *and* leaves it out of propagation.

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// "Notify" (on by default): when a run ends -- succeeded, failed, paused,
// or the request itself errored, but not a user cancel -- play a short
// chime and, while the tab isn't focused, show a browser notification and
// mark the tab title until the tab is looked at again. The chime is
// synthesized with Web Audio, so there is no sound file to serve.
// Notifications need a secure context (https or localhost); when they are
// unavailable or denied, the chime and the title mark still work.
const PROPAGATION_NOTIFY_STORAGE_KEY = "vistlabel-propagation-notify";
const pageTitle = document.title;
let notifyAudioContext = null;

function readPropagationNotifyPreference() {
  try {
    return window.localStorage.getItem(PROPAGATION_NOTIFY_STORAGE_KEY) !== "0";
  } catch (error) {
    return true; // storage unavailable -- keep the default
  }
}

els.propagationNotify.checked = readPropagationNotifyPreference();
els.propagationNotify.addEventListener("change", () => {
  try {
    window.localStorage.setItem(
      PROPAGATION_NOTIFY_STORAGE_KEY, els.propagationNotify.checked ? "1" : "0"
    );
  } catch (error) {
    // storage unavailable -- the toggle still applies for this page load
  }
  if (els.propagationNotify.checked) prepareCompletionNotice();
});

// Browsers only let audio start and permission prompts open from a user
// gesture, so this runs on the Run click (and on ticking Notify), not when
// the job ends minutes later.
function prepareCompletionNotice() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (AudioContextClass && !notifyAudioContext) {
    try {
      notifyAudioContext = new AudioContextClass();
    } catch (error) {
      notifyAudioContext = null;
    }
  }
  if (notifyAudioContext && notifyAudioContext.state === "suspended") {
    notifyAudioContext.resume().catch(() => {});
  }
  if ("Notification" in window && Notification.permission === "default") {
    try {
      const pending = Notification.requestPermission();
      if (pending && pending.catch) pending.catch(() => {});
    } catch (error) {
      // very old Safari only takes a callback; skip the prompt there
    }
  }
}

function playCompletionChime(succeeded) {
  const ctx = notifyAudioContext;
  if (!ctx) return;
  // Rising major triad for success, two falling low tones for a failure.
  const notes = succeeded ? [523.25, 659.25, 783.99] : [392.0, 261.63];
  const step = succeeded ? 0.13 : 0.22;
  const start = ctx.currentTime + 0.02;
  notes.forEach((frequency, index) => {
    const oscillator = ctx.createOscillator();
    const gain = ctx.createGain();
    oscillator.type = succeeded ? "sine" : "triangle";
    oscillator.frequency.value = frequency;
    const t = start + index * step;
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.exponentialRampToValueAtTime(0.25, t + 0.015);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + step + 0.15);
    oscillator.connect(gain).connect(ctx.destination);
    oscillator.start(t);
    oscillator.stop(t + step + 0.2);
  });
}

function restorePageTitle() {
  if (document.hidden || !document.hasFocus()) return;
  document.title = pageTitle;
}
document.addEventListener("visibilitychange", restorePageTitle);
window.addEventListener("focus", restorePageTitle);

function notifyPropagationDone(succeeded, message) {
  if (!els.propagationNotify.checked) return;
  playCompletionChime(succeeded);
  if (!document.hidden && document.hasFocus()) return; // the status line says it
  document.title = `${succeeded ? "✓" : "✗"} ${pageTitle}`;
  if (!("Notification" in window) || Notification.permission !== "granted") return;
  try {
    const notice = new Notification(pageTitle, {
      body: message,
      tag: `vistlabel-propagation-${projectId}`,
    });
    notice.onclick = () => {
      window.focus();
      notice.close();
    };
  } catch (error) {
    // e.g. Chrome on Android only allows notifications from a service worker
  }
}

function setPropagationRunning(running) {
  state.propagationRunning = running;
  els.propagateButton.disabled = running || unsupportedReason("video_propagation") !== null;
  els.cancelPropagationButton.disabled = !running;
  els.propagationDirection.disabled = running;
  els.propagationEndFrame.disabled = running;
}

async function pollJobUntilDone(jobId) {
  while (true) {
    const record = await apiGet(`/api/v1/projects/${projectId}/jobs/${jobId}`);
    const percent = Math.round((record.progress_fraction || 0) * 100);
    setStatus(`Propagating... ${percent}%`);
    els.propagationStatus.textContent = `${record.status} (${percent}%)`;
    if (["succeeded", "failed", "cancelled", "paused"].includes(record.status)) {
      return record;
    }
    await sleep(400);
  }
}

async function startPropagation() {
  const frameObjects = (state.frameAnnotation && state.frameAnnotation.objects) || [];
  const seededIds = new Set(frameObjects.map((o) => o.object_id));
  const objectIds = state.objects
    .filter((o) => seededIds.has(o.object_id))
    .map((o) => o.object_id);
  if (objectIds.length === 0) {
    setStatus("No object has a saved mask on this frame to propagate from");
    return;
  }
  const endFrame = clampFrame(parseInt(els.propagationEndFrame.value, 10));
  const direction = els.propagationDirection.value;

  if (els.propagationNotify.checked) prepareCompletionNotice();
  setPropagationRunning(true);
  els.propagationStatus.textContent = "starting...";
  let notified = false;
  try {
    const { job_id: jobId } = await apiJson(
      "POST", `/api/v1/projects/${projectId}/propagate`,
      {
        backend_id: state.selectedBackendId,
        object_ids: objectIds,
        start_frame: state.currentFrame,
        end_frame: endFrame,
        direction,
      }
    );
    state.propagationJobId = jobId;
    const record = await pollJobUntilDone(jobId);
    state.propagationJobId = null;
    let message;
    if (record.status === "succeeded") {
      message = "Propagation complete";
    } else if (record.status === "failed") {
      message = `Propagation failed: ${record.error || "unknown error"}`;
    } else {
      message = `Propagation ${record.status}`;
    }
    setStatus(message);
    if (record.status !== "cancelled") {
      notifyPropagationDone(record.status === "succeeded", message);
    }
    notified = true; // a failed frame reload below isn't a failed run
    await loadFrame(state.currentFrame);
  } catch (error) {
    const message = `Propagation request failed: ${error.detail || error.message}`;
    setStatus(message);
    if (!notified) notifyPropagationDone(false, message);
  } finally {
    state.propagationJobId = null;
    setPropagationRunning(false);
  }
}

els.propagateButton.addEventListener("click", startPropagation);

els.cancelPropagationButton.addEventListener("click", async () => {
  if (!state.propagationJobId) return;
  await apiJson("POST", `/api/v1/projects/${projectId}/jobs/${state.propagationJobId}/cancel`, {});
  setStatus("Cancelling...");
});

els.propagationDirection.addEventListener("change", () => {
  els.propagationEndFrame.value =
    els.propagationDirection.value === "forward" ? String(frameCount - 1) : "0";
});

// ---------------------------------------------------------- shortcuts ----

// Letter shortcuts that act on the current frame or tool; while playing they
// stop playback first, so they apply to the frame on screen.
const EDITING_KEYS = new Set([..."bepnahdrBEPNAHDR"]);

// The number row picks the rail's tools top to bottom. Matched by physical
// key, so it works with or without Shift and on any layout -- on a Czech
// keyboard these keys type + ě š č ř; the characters are a fallback.
const TOOL_BY_KEY_CODE = {
  Digit1: "point_positive", Digit2: "point_negative", Digit3: "brush",
  Digit4: "eraser", Digit5: "axis_hint",
};
const TOOL_BY_KEY = {
  1: "point_positive", "+": "point_positive", 2: "point_negative", "ě": "point_negative",
  3: "brush", "š": "brush", 4: "eraser", "č": "eraser", 5: "axis_hint", "ř": "axis_hint",
};

// Fields where Space is typed text. The frame slider/number box keep focus
// after use, so Space must still toggle playback there.
const TEXT_INPUT_TYPES = new Set(["text", "search", "email", "url", "password", "tel"]);
const SHORTCUT_INPUT_TYPES = new Set(["range", "checkbox"]);

window.addEventListener("keydown", (event) => {
  const tag = event.target.tagName;
  const textEntry = tag === "TEXTAREA" || tag === "SELECT"
    || (tag === "INPUT" && TEXT_INPUT_TYPES.has(event.target.type));
  if (event.key === " " && !textEntry && !event.ctrlKey && !event.metaKey && !event.altKey) {
    // Space would otherwise scroll, or "click" whichever button has focus.
    event.preventDefault();
    if (tag === "BUTTON" || tag === "INPUT") event.target.blur();
    if (!event.repeat) togglePlayback();
    return;
  }
  if (tag === "TEXTAREA" || tag === "SELECT") return;
  // Sliders and checkboxes type nothing, so shortcuts still work after
  // using one (e.g. set the brush radius, then 4 for the eraser) -- except
  // the arrows, which move the slider.
  if (tag === "INPUT" && (
    !SHORTCUT_INPUT_TYPES.has(event.target.type) || event.key.startsWith("Arrow")
  )) return;
  if (event.ctrlKey && event.key === "z") { event.preventDefault(); undo(); return; }
  if (event.ctrlKey && event.key === "y") { event.preventDefault(); redo(); return; }
  // Leave browser shortcuts alone -- Ctrl+R must reload, not mark reviewed.
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  const numberTool = TOOL_BY_KEY_CODE[event.code] || TOOL_BY_KEY[event.key.toLowerCase()];
  if (numberTool) {
    stopPlayback();
    setTool(numberTool);
    return;
  }
  if (EDITING_KEYS.has(event.key)) stopPlayback();
  switch (event.key) {
    case "ArrowLeft": goToFrame(state.currentFrame - 1); break;
    case "ArrowRight": goToFrame(state.currentFrame + 1); break;
    case "b": case "B": setTool("brush"); break;
    case "e": case "E": setTool("eraser"); break;
    case "p": case "P": setTool("point_positive"); break;
    case "n": case "N": setTool("point_negative"); break;
    case "a": case "A": setTool("axis_hint"); break;
    case "h": case "H":
      if (state.selectedObjectId) toggleObjectViewHidden(state.selectedObjectId);
      break;
    case "d": case "D":
      if (state.selectedObjectId) {
        toggleObjectActiveOnFrame(state.selectedObjectId).catch(
          (error) => setStatus(`Could not (de)activate: ${error.detail || error.message}`)
        );
      }
      break;
    case "r": case "R":
      saveReviewStatus("reviewed").then(
        () => setStatus(`Frame ${state.currentFrame} reviewed`),
        (error) => setStatus(`Could not mark reviewed: ${error.detail || error.message}`)
      );
      break;
    default: break;
  }
});

// -------------------------------------------------------------- init ----

async function init() {
  els.canvas.width = frameWidth;
  els.canvas.height = frameHeight;
  els.frameSlider.max = String(frameCount - 1);
  els.frameNumberInput.max = String(frameCount - 1);
  els.frameCountLabel.textContent = `/ ${frameCount}`;
  els.propagationEndFrame.max = String(frameCount - 1);
  els.propagationEndFrame.value = String(frameCount - 1);
  setTool("point_positive");

  await loadBackends();
  await loadLabelsAndObjects();
  await loadFrame(state.currentFrame);

  panels = setUpWorkspacePanels({
    projectId,
    getCurrentFrame: () => state.currentFrame,
    getRevision: () => state.projectRevision,
    setRevision: (revision) => {
      state.projectRevision = revision;
    },
    hooks: {
      goToFrame,
      selectObject: async (objectId) => {
        if (state.objects.some((o) => o.object_id === objectId)) await selectObject(objectId);
      },
      markFrameReviewed,
      setStatus,
      // Auto pre-annotation creates objects server-side, so both the object
      // list and the frame's masks are stale once it finishes.
      reloadFrame: async (frameIndex) => {
        await loadLabelsAndObjects();
        await loadFrame(frameIndex);
      },
    },
  });

  setStatus("Ready");
}

init().catch((error) => {
  console.error(error);
  setStatus(`Failed to initialize: ${error.message}`);
});
