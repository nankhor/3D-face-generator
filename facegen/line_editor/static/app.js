"use strict";
// Jaw-line editor. All coordinates sent to the server are original photo pixels.

const COLORS = { jaw: "#57d08f", beard: "#e3aa45", hairline: "#78b6e6", anchor: "#f07a93" };
const ANCHOR_LABELS = {
  nose_tip: "Nose tip",
  eye_outer_right: "Outer corner, right eye",
  eye_outer_left: "Outer corner, left eye",
  mouth_corner_right: "Mouth corner, right",
  mouth_corner_left: "Mouth corner, left",
  ear_right: "Right ear opening",
  ear_left: "Left ear opening",
};

const GUIDE = {
  jaw: `
    <svg width="240" height="96" viewBox="0 0 240 96" aria-hidden="true">
      <path d="M24 12c-14 6-18 26-12 40 4 10 14 18 22 26 8 7 20 10 30 6" fill="none" stroke="#a39a8d" stroke-width="1.5"/>
      <path d="M64 84c-8-2-14-8-18-16M46 68c-3-8-4-18-2-28" fill="none" stroke="#57d08f" stroke-width="3" stroke-linecap="round"/>
      <circle cx="40" cy="38" r="5" fill="none" stroke="#a39a8d" stroke-width="1.5"/>
      <path d="M150 10c-20 0-30 16-30 34 0 20 12 40 30 44 18-4 30-24 30-44 0-18-10-34-30-34z" fill="none" stroke="#a39a8d" stroke-width="1.5"/>
      <path d="M122 50c2 14 10 28 28 38M178 50c-2 14-10 28-28 38" fill="none" stroke="#57d08f" stroke-width="3" stroke-linecap="round"/>
    </svg>
    <p>Feel your jawbone under the beard and draw where the <strong>bone</strong> is, not where the beard ends.</p>
    <p>Side photo: start at the chin, follow the bottom of the jaw to its corner, then go up the back edge toward the ear.</p>
    <p>Front photo: draw each side of the jaw from below the ear down to the chin. Two or three photos are enough.</p>`,
  anchor: `
    <p>Use anchors on photos where no face was detected (usually full side views), so the camera can be placed.</p>
    <p>Click the points you can see. The ear opening also helps on three-quarter photos: mark it there too.</p>`,
  beard: `<p>Optional. Trace the outer edge of the beard. Claude uses it when choosing a beard style.</p>`,
  hairline: `<p>Optional. Trace where the hair starts on the forehead and temples. Claude uses it when choosing a hairstyle.</p>`,
  erase: `<p>Click a line or an anchor to remove it.</p>`,
};

const state = {
  photos: [], anchors: [], lines: {}, current: null, tool: "jaw", anchorName: "nose_tip",
  img: null, imgScale: 1, view: { s: 1, x: 0, y: 0 }, draft: [], overlay: { landmarks: [], jaw: [] },
  history: [], spaceDown: false, pan: null, hover: null,
};

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.headers.get("content-type")?.includes("json") ? r.json() : r;
}

// ---------------------------------------------------------------- data
function entry(name) {
  if (!state.lines[name]) state.lines[name] = {};
  return state.lines[name];
}

function pushHistory() {
  state.history.push(JSON.stringify(state.lines));
  if (state.history.length > 100) state.history.shift();
}

let saveTimer = null;
function scheduleSave() {
  const el = $("save-state");
  el.textContent = "Saving…";
  el.className = "save-state saving";
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    try {
      await api("/api/lines", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(state.lines) });
      el.textContent = "Saved";
      el.className = "save-state";
      renderList();
    } catch (e) {
      el.textContent = "Not saved: " + e.message + ". Is the FaceGen server still running?";
      el.className = "save-state error";
    }
  }, 300);
}

// ---------------------------------------------------------------- photo list
function renderList() {
  const list = $("photo-list");
  list.innerHTML = "";
  let withJaw = 0;
  state.photos.forEach((p) => {
    const e = state.lines[p.name] || {};
    const jaw = (e.jaw || []).length;
    const anchors = Object.keys(e.anchors || {}).length;
    if (jaw) withJaw++;
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.className = "photo-item";
    b.setAttribute("aria-current", p.name === state.current ? "true" : "false");
    b.innerHTML = `<img alt="" loading="lazy" src="/api/photo/${encodeURIComponent(p.name)}?max=200">
      <span><span class="photo-name">${p.name}</span><span class="marks"></span></span>`;
    const marks = b.querySelector(".marks");
    if (p.face === false && !p.camera) marks.insertAdjacentHTML("beforeend", `<span class="mark warn" title="No face detected: add anchors">needs anchors</span>`);
    if (jaw) marks.insertAdjacentHTML("beforeend", `<span class="mark jaw">${jaw} jaw</span>`);
    if (anchors) marks.insertAdjacentHTML("beforeend", `<span class="mark anchor">${anchors} anchors</span>`);
    b.onclick = () => selectPhoto(p.name);
    li.appendChild(b);
    list.appendChild(li);
  });
  $("progress").textContent = `${state.photos.length} photos, jaw drawn on ${withJaw}`;
}

async function selectPhoto(name) {
  finishDraft(false);
  state.current = name;
  renderList();
  const p = state.photos.find((x) => x.name === name);
  const img = new Image();
  img.src = `/api/photo/${encodeURIComponent(name)}?max=2400`;
  await img.decode();
  state.img = img;
  state.imgScale = p.width / img.naturalWidth;
  fit();
  try { state.overlay = await api(`/api/overlay/${encodeURIComponent(name)}`); } catch { state.overlay = { landmarks: [], jaw: [] }; }
  if (p.face === false && !p.camera) setTool("anchor");
  draw();
}

// ---------------------------------------------------------------- view
function resize() {
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.round(r.width * devicePixelRatio);
  canvas.height = Math.round(r.height * devicePixelRatio);
  draw();
}

function fit() {
  if (!state.img) return;
  const p = state.photos.find((x) => x.name === state.current);
  const s = Math.min(canvas.width / p.width, canvas.height / p.height) * 0.95;
  state.view = { s, x: (canvas.width - p.width * s) / 2, y: (canvas.height - p.height * s) / 2 };
  draw();
}

function toPhoto(ev) {
  const r = canvas.getBoundingClientRect();
  const cx = (ev.clientX - r.left) * devicePixelRatio;
  const cy = (ev.clientY - r.top) * devicePixelRatio;
  return [(cx - state.view.x) / state.view.s, (cy - state.view.y) / state.view.s];
}

function zoomAt(factor, cx, cy) {
  const v = state.view;
  v.x = cx - (cx - v.x) * factor;
  v.y = cy - (cy - v.y) * factor;
  v.s *= factor;
  draw();
}

// ---------------------------------------------------------------- drawing
function polyline(points, color, width, dash) {
  if (points.length < 1) return;
  const v = state.view;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width * devicePixelRatio;
  ctx.lineJoin = ctx.lineCap = "round";
  if (dash) ctx.setLineDash(dash.map((d) => d * devicePixelRatio));
  ctx.beginPath();
  points.forEach(([x, y], i) => (i ? ctx.lineTo(x * v.s + v.x, y * v.s + v.y) : ctx.moveTo(x * v.s + v.x, y * v.s + v.y)));
  ctx.stroke();
  ctx.restore();
}

function dot(x, y, r, color, ring) {
  const v = state.view;
  ctx.beginPath();
  ctx.arc(x * v.s + v.x, y * v.s + v.y, r * devicePixelRatio, 0, Math.PI * 2);
  if (ring) { ctx.lineWidth = 2 * devicePixelRatio; ctx.strokeStyle = color; ctx.stroke(); }
  else { ctx.fillStyle = color; ctx.fill(); }
}

function draw() {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.img) return;
  const p = state.photos.find((x) => x.name === state.current);
  const v = state.view;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(state.img, v.x, v.y, p.width * v.s, p.height * v.s);
  if ($("show-landmarks").checked) state.overlay.landmarks.forEach(([x, y]) => dot(x, y, 1.6, "rgba(255,255,255,0.75)"));
  if ($("show-fit").checked && state.overlay.jaw.length) polyline(state.overlay.jaw, "#ff5a4f", 2, [6, 5]);
  const e = state.lines[state.current] || {};
  ["hairline", "beard", "jaw"].forEach((k) => (e[k] || []).forEach((l) => {
    polyline(l, "rgba(0,0,0,0.55)", 5);
    polyline(l, COLORS[k], 2.5);
    l.forEach(([x, y]) => dot(x, y, 2.5, COLORS[k]));
  }));
  Object.entries(e.anchors || {}).forEach(([n, [x, y]]) => {
    dot(x, y, 7, COLORS.anchor, true);
    dot(x, y, 2, COLORS.anchor);
    const sx = x * v.s + v.x + 10 * devicePixelRatio, sy = y * v.s + v.y - 8 * devicePixelRatio;
    ctx.font = `${12 * devicePixelRatio}px system-ui`;
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    const label = ANCHOR_LABELS[n] || n;
    ctx.fillRect(sx - 3, sy - 13 * devicePixelRatio, ctx.measureText(label).width + 6, 17 * devicePixelRatio);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, sx, sy);
  });
  if (state.draft.length) {
    const pts = state.hover ? [...state.draft, state.hover] : state.draft;
    polyline(pts, COLORS[state.tool] || "#fff", 2.5, [8, 6]);
    state.draft.forEach(([x, y]) => dot(x, y, 3, COLORS[state.tool] || "#fff"));
  }
}

// ---------------------------------------------------------------- tools
function setTool(tool) {
  finishDraft(true);
  state.tool = tool;
  document.querySelectorAll(".tool").forEach((b) => b.setAttribute("aria-checked", b.dataset.tool === tool ? "true" : "false"));
  $("anchor-picker").hidden = tool !== "anchor";
  $("guide").innerHTML = GUIDE[tool];
  updateHint();
  renderAnchorPicker();
}

function renderAnchorPicker() {
  const box = $("anchor-picker");
  box.innerHTML = "";
  const placed = (state.lines[state.current] || {}).anchors || {};
  state.anchors.forEach((n) => {
    const b = document.createElement("button");
    b.setAttribute("aria-pressed", n === state.anchorName ? "true" : "false");
    b.innerHTML = `<span>${ANCHOR_LABELS[n] || n}</span><span class="placed">${placed[n] ? "placed" : ""}</span>`;
    b.onclick = () => { state.anchorName = n; renderAnchorPicker(); updateHint(); };
    box.appendChild(b);
  });
}

function updateHint() {
  const hints = {
    jaw: state.draft.length ? "Keep clicking along the jaw. Double-click or press Enter to finish." : "Click along the jawbone to start a line.",
    anchor: `Click to place: ${ANCHOR_LABELS[state.anchorName] || state.anchorName}.`,
    beard: "Click along the outer edge of the beard. Double-click to finish.",
    hairline: "Click along the hairline. Double-click to finish.",
    erase: "Click a line or anchor to remove it.",
  };
  $("hint").textContent = hints[state.tool];
}

function finishDraft(keep) {
  if (keep && state.draft.length >= 2 && state.current) {
    pushHistory();
    const e = entry(state.current);
    (e[state.tool] = e[state.tool] || []).push(state.draft.map(([x, y]) => [Math.round(x * 10) / 10, Math.round(y * 10) / 10]));
    scheduleSave();
  }
  state.draft = [];
  updateHint();
  draw();
}

function eraseAt(x, y) {
  const e = state.lines[state.current];
  if (!e) return;
  const tol = 12 / state.view.s * devicePixelRatio;
  let best = null;
  for (const k of ["jaw", "beard", "hairline"]) {
    (e[k] || []).forEach((l, i) => {
      for (let j = 0; j < l.length; j++) {
        const a = l[j], b = l[Math.min(j + 1, l.length - 1)];
        const d = segDist([x, y], a, b);
        if (d < tol && (!best || d < best.d)) best = { k, i, d };
      }
    });
  }
  Object.entries(e.anchors || {}).forEach(([n, p]) => {
    const d = Math.hypot(p[0] - x, p[1] - y);
    if (d < tol * 1.5 && (!best || d < best.d)) best = { k: "anchors", n, d };
  });
  if (!best) return;
  pushHistory();
  if (best.k === "anchors") delete e.anchors[best.n];
  else e[best.k].splice(best.i, 1);
  scheduleSave();
  renderAnchorPicker();
  draw();
}

function segDist(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const L = dx * dx + dy * dy;
  let t = L ? ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L : 0;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy);
}

// ---------------------------------------------------------------- events
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("mousedown", (ev) => {
  canvas.focus();
  if (ev.button === 2 || ev.button === 1 || state.spaceDown) {
    state.pan = { x: ev.clientX, y: ev.clientY, vx: state.view.x, vy: state.view.y };
    canvas.classList.add("panning");
    return;
  }
  if (ev.button !== 0 || !state.current) return;
  const [x, y] = toPhoto(ev);
  if (state.tool === "anchor") {
    pushHistory();
    const e = entry(state.current);
    e.anchors = e.anchors || {};
    e.anchors[state.anchorName] = [Math.round(x * 10) / 10, Math.round(y * 10) / 10];
    const placed = e.anchors;
    const next = state.anchors.find((n) => !placed[n]);
    if (next) state.anchorName = next;
    scheduleSave();
    renderAnchorPicker();
    updateHint();
    draw();
  } else if (state.tool === "erase") {
    eraseAt(x, y);
  } else {
    if (ev.detail >= 2) { finishDraft(true); return; }
    state.draft.push([x, y]);
    updateHint();
    draw();
  }
});
window.addEventListener("mousemove", (ev) => {
  if (state.pan) {
    state.view.x = state.pan.vx + (ev.clientX - state.pan.x) * devicePixelRatio;
    state.view.y = state.pan.vy + (ev.clientY - state.pan.y) * devicePixelRatio;
    draw();
    return;
  }
  if (state.draft.length && ev.target === canvas) { state.hover = toPhoto(ev); draw(); }
});
window.addEventListener("mouseup", () => { state.pan = null; canvas.classList.remove("panning"); });
canvas.addEventListener("wheel", (ev) => {
  ev.preventDefault();
  const r = canvas.getBoundingClientRect();
  zoomAt(ev.deltaY < 0 ? 1.15 : 1 / 1.15, (ev.clientX - r.left) * devicePixelRatio, (ev.clientY - r.top) * devicePixelRatio);
}, { passive: false });

window.addEventListener("keydown", (ev) => {
  if (ev.target.tagName === "INPUT") return;
  const k = ev.key.toLowerCase();
  if (ev.key === " ") { state.spaceDown = true; ev.preventDefault(); return; }
  if ((ev.ctrlKey || ev.metaKey) && k === "z") {
    ev.preventDefault();
    if (state.draft.length) { state.draft.pop(); draw(); return; }
    const prev = state.history.pop();
    if (prev) { state.lines = JSON.parse(prev); scheduleSave(); renderAnchorPicker(); draw(); }
    return;
  }
  if (ev.key === "Enter") return finishDraft(true);
  if (ev.key === "Escape") return finishDraft(false);
  const tools = { j: "jaw", a: "anchor", b: "beard", h: "hairline", e: "erase" };
  if (tools[k]) return setTool(tools[k]);
  if (k === "f") return fit();
  if (k === "+" || k === "=") return zoomAt(1.25, canvas.width / 2, canvas.height / 2);
  if (k === "-") return zoomAt(0.8, canvas.width / 2, canvas.height / 2);
  if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") {
    const i = state.photos.findIndex((p) => p.name === state.current);
    const j = Math.max(0, Math.min(state.photos.length - 1, i + (ev.key === "ArrowRight" ? 1 : -1)));
    if (j !== i) selectPhoto(state.photos[j].name);
  }
});
window.addEventListener("keyup", (ev) => { if (ev.key === " ") state.spaceDown = false; });
window.addEventListener("resize", resize);
document.querySelectorAll(".tool").forEach((b) => (b.onclick = () => setTool(b.dataset.tool)));
$("fit").onclick = fit;
$("zoom-in").onclick = () => zoomAt(1.25, canvas.width / 2, canvas.height / 2);
$("zoom-out").onclick = () => zoomAt(0.8, canvas.width / 2, canvas.height / 2);
$("show-landmarks").onchange = draw;
$("show-fit").onchange = draw;

// ---------------------------------------------------------------- start
(async function start() {
  try {
    const s = await api("/api/state");
    state.photos = s.photos;
    state.anchors = s.anchors;
    state.lines = await api("/api/lines");
    $("project-name").textContent = s.project;
    resize();
    setTool("jaw");
    renderList();
    if (state.photos.length) selectPhoto(state.photos[0].name);
    else $("hint").textContent = "This project has no photos yet.";
  } catch (e) {
    $("hint").textContent = "Could not reach the FaceGen server: " + e.message;
  }
})();
