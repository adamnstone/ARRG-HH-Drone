// Calibration tab: which mic headers (PCB MK1..MK5) and motor outputs (M1..M8) are connected, which mic position each
// connected mic sits at (left, right, top, and ONE of front or back: they share the DSP's channel 0, back by default),
// and which hole of the band (from the STL, /api/geometry) each connected motor is mounted in. Mic positions can be
// set by hand or by an auto-calibration (claps from the front, right and left).
// Saved to data/calibration.json and applied to the headset by the server (the board's C / Y commands). Loaded after
// app.js and uses its globals ($, sendCmd, status, source, escapeHtml, benchShow, benchState); exposes
// window.calibShow, window.onCalibBench, window.onCalibSource.
(function () {
"use strict";

const POSITIONS = ["front", "back", "left", "right", "top"];
const POS_LABEL = { front: "Front", back: "Back", left: "Left", right: "Right", top: "Top" };
// the board's channels: C<w0>,<l>,<r>,<t> with channel 0 the back mic, or C<w0>,<l>,<r>,<t>,F with it the front one
const DSP_CH = { front: "F", back: "B", left: "L", right: "R", top: "T" };
const OTHER_SIDE = { front: "back", back: "front" };                     // front and back: one channel, never both
const AUTO_POS = ["back", "left", "right", "top"];                     // where a newly ticked header goes, if free
const N_CH = 4;                                                         // the DSP's channels (mics connected at most)
// The mic headers: each sits on one PDM data line in one SEL phase, and any of them can take any of the positions (at
// most four are connected). pos is the position its PCB label suggests (MK1 has none). Which DMA word a phase lands
// in depends on the slot order (right_first: the SEL 3V3 phase is the line's first word, as the v3 board does,
// verified by tapping; left_first: SEL GND first), which the calibration records. Five headers, six DMA words: one
// word has no header (w4 right first, w5 left first), so no mic can be on it; at most it mirrors its line partner,
// and a map that names it is moved to that partner (onHeaders).
const HEADERS = [
  { id: "MK1", pos: null, line: 0, sel: "GND" }, { id: "MK2", pos: "back", line: 0, sel: "3V3" },
  { id: "MK3", pos: "left", line: 1, sel: "GND" }, { id: "MK4", pos: "right", line: 1, sel: "3V3" },
  { id: "MK5", pos: "top", line: 2, sel: "GND" },
];
const N_ROWS = HEADERS.length;
const ALL_IN_USE = "All four positions are in use (front or back, left, right, top): untick a mic or change its position first.";
const FRONT_BACK = "front and back share one channel";
const SLOT_ORDERS = { left_first: "SEL GND slot first", right_first: "SEL 3V3 slot first (the v3 board)" };
const DEFAULT_ORDER = "right_first";                 // verified on the v3 board: it reads the SEL 3V3 phase first
const MOTORS = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"];
const MOTOR_NAME = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"];          // pins.h order
const MOTOR_DEFAULT_DEG = [0, 45, 90, 135, 180, -135, -90, -45];
const DUTY = 190;                 // pins.h MOTOR_DUTY_MAX
const BUZZ_MS = 600;
const CAPTURE_S = 3;
const DB_FLOOR = -80;             // level bar scale, as the bench panel
const STEPS = [
  { dir: "front", req: true, text: "Stand about 1 m in FRONT of the headset, level with it, and clap sharply twice while it records." },
  { dir: "right", req: true, text: "Stand about 1 m to the headset's RIGHT (the wearer's right), level with it, and clap sharply twice." },
  { dir: "left", req: true, text: "Stand about 1 m to the headset's LEFT, level with it, and clap sharply twice." },
  { dir: "back", req: false, text: "Optional: stand about 1 m BEHIND the headset and clap sharply twice." },
  { dir: "top", req: false, text: "Optional: hold your hands about 50 cm ABOVE the headset and clap sharply twice." },
];

const S = {
  built: false, on: false, loaded: false, apiOk: true,
  geom: null, cal: null, saved: null, device: null,
  words: [], lastLevels: 0, levelTimer: null,
  captures: {}, capturing: null, capT0: 0, capElapsed: 0, capTimer: null, step: 0, solve: null,
  isDevice: false, connected: false,
  selMotor: null, buzzing: null, buzzT: null, feel: null, feelTimer: null, lastHole: {},
  msg: "", msgBad: false,
};

// ---------------------------------------------------------------- geometry / calibration defaults
const wrap = (d) => { d = ((d + 180) % 360 + 360) % 360 - 180; return d === -180 ? 180 : d; };
const angDist = (a, b) => Math.abs(wrap(a - b));
const round1 = (x) => Math.round(x * 10) / 10;
const fmtAng = (d) => (d == null ? "—" : `${d > 0 ? "+" : d < 0 ? "−" : ""}${Math.abs(round1(d))}°`.replace(/\.0°/, "°"));

function placeholderGeometry(why) {
  // 8 holes at the default motor angles on an 83 x 110 mm ellipse, a slightly larger ellipse as the band outline
  const rAt = (deg, ax, ay) => { const a = deg * Math.PI / 180; return 1 / Math.hypot(Math.sin(a) / ax, Math.cos(a) / ay); };
  const holes = MOTOR_DEFAULT_DEG.map((deg, i) => {
    const r = rAt(deg, 83, 110), a = deg * Math.PI / 180;
    return { id: `H${i + 1}`, theta_deg: deg, x_mm: r * Math.sin(a), y_mm: r * Math.cos(a), z_mm: 8, diameter_mm: 8 };
  });
  const outline = [];
  for (let k = 0; k < 96; k++) { const t = 2 * Math.PI * k / 96; outline.push([92 * Math.sin(t), 120 * Math.cos(t)]); }
  return { source: "placeholder", units: "mm", placeholder: true, why: why || "", outline, holes, mic_sites: [] };
}

const holeById = (id) => (S.geom && S.geom.holes || []).find((h) => h.id === id) || null;
function nearestHole(deg) {
  let best = null;
  for (const h of S.geom.holes) if (!best || angDist(h.theta_deg, deg) < angDist(best.theta_deg, deg)) best = h;
  return best;
}

// the physical slot DMA word w reads under a slot order: data line w >> 1, and its first word is the SEL 3V3 phase when
// right_first, the SEL GND phase when left_first
function wordLine(w, order) {
  const right = (order || S.cal.slot_order) === "right_first";
  return `DIN${w >> 1} · SEL ${((w & 1) === 0) !== right ? "GND" : "3V3"}`;
}
// the DMA word of header k (0..4 = MK1..MK5) under a slot order
function headerWord(k, order) {
  const right = (order || S.cal.slot_order) === "right_first";
  const h = HEADERS[k];
  return 2 * h.line + ((h.sel === "GND") !== right ? 0 : 1);
}
// A map with only header words: a word with no header can only be a mirror of its line partner (w ^ 1, a header),
// so it moves there, or is dropped if another position already has that partner. Returns the positions it changed.
function onHeaders(mw, order) {
  const hw = HEADERS.map((_, k) => headerWord(k, order)), changed = [];
  for (const p of POSITIONS) {
    const w = mw[p];
    if (w == null || hw.includes(w)) continue;
    const partner = w ^ 1, taken = POSITIONS.some((q) => q !== p && mw[q] === partner);
    mw[p] = hw.includes(partner) && !taken ? partner : null;
    changed.push(p);
  }
  return changed;
}
function headerName(w, order) {
  const k = HEADERS.findIndex((_, i) => headerWord(i, order) === w);
  return k >= 0 ? HEADERS[k].id : "no header";
}
function pcbWords(order) {                          // MK2..MK5 connected at their PCB positions, MK1 not
  const mw = {};
  for (const p of POSITIONS) mw[p] = null;
  HEADERS.forEach((h, k) => { if (h.pos) mw[h.pos] = headerWord(k, order); });
  return mw;
}

function defaultCalibration() {
  const motors = {};
  MOTORS.forEach((m, i) => {                       // the hole nearest each pins.h direction, at that hole's angle
    const h = nearestHole(MOTOR_DEFAULT_DEG[i]);
    motors[m] = h ? { hole: h.id, theta_deg: round1(h.theta_deg) } : { hole: null, theta_deg: MOTOR_DEFAULT_DEG[i] };
  });
  return { version: 1, mic_words: pcbWords(DEFAULT_ORDER), mic_source: "default", slot_order: DEFAULT_ORDER,
           motors, motor_source: "default" };
}

function normalize(c) {
  const d = defaultCalibration();
  c = Object.assign({}, d, c || {});
  const mw = {};
  for (const p of POSITIONS) mw[p] = c.mic_words && c.mic_words[p] != null ? Number(c.mic_words[p]) : null;
  if (mw.front != null && mw.back != null) mw.front = null;   // one channel: the server never sends both
  c.mic_words = mw;
  c.slot_order = SLOT_ORDERS[c.slot_order] ? c.slot_order : DEFAULT_ORDER;
  onHeaders(c.mic_words, c.slot_order);
  const mo = {};
  for (const m of MOTORS) {
    const v = c.motors ? c.motors[m] : d.motors[m];
    mo[m] = v == null ? null : { hole: v.hole || null, theta_deg: Number(v.theta_deg) };
  }
  c.motors = mo;
  return c;
}
const calKey = (c) => JSON.stringify({ mw: c.mic_words, ms: c.mic_source, so: c.slot_order, mo: c.motors, mos: c.motor_source });
const dirty = () => !S.saved || calKey(S.cal) !== S.saved;
const wordAt = (pos) => S.cal.mic_words[pos];
const posOfWord = (w) => POSITIONS.find((p) => S.cal.mic_words[p] === w) || null;
const deviceOk = () => S.isDevice && S.connected;
const ready = () => S.built && !!S.cal && !!S.geom;   // the DOM is built and the geometry / calibration are loaded

// ---------------------------------------------------------------- server
async function api(method, path, body) {
  const r = await fetch(path, { method, headers: body ? { "Content-Type": "application/json" } : {},
                                body: body ? JSON.stringify(body) : undefined });
  if (!(r.headers.get("content-type") || "").includes("json")) {
    throw new Error(r.status === 404 ? "this dashboard server has no calibration API yet: restart scripts/dashboard.py"
                                     : `server answered HTTP ${r.status}`);
  }
  const d = await r.json();
  if (!r.ok || d.ok === false) throw new Error(d.error || `HTTP ${r.status}`);
  return d;
}

async function load() {
  try {
    const g = await api("GET", "/api/geometry");
    S.geom = g && g.holes && g.holes.length ? g : placeholderGeometry("the geometry file has no holes");
  } catch (e) {
    S.geom = placeholderGeometry(e.message);
  }
  try {
    const d = await api("GET", "/api/calibration");
    S.cal = normalize(d.calibration);
    S.saved = calKey(S.cal);
    if (d.device) S.device = d.device;
    for (const dir of d.captures || []) if (!S.captures[dir]) S.captures[dir] = { stored: true };
    S.apiOk = true;
  } catch (e) {
    S.cal = defaultCalibration();
    S.saved = null;
    S.apiOk = false;
    note(`Calibration API unavailable (${e.message}); showing the defaults, nothing can be saved.`, true);
  }
  S.loaded = true;
  buildHoleOptions();
  renderAll();
}

async function save() {
  try {
    const d = await api("POST", "/api/calibration", S.cal);
    S.cal = normalize(d.calibration || S.cal);
    S.saved = calKey(S.cal);
    note("Saved to data/calibration.json.");
    renderAll();
    return true;
  } catch (e) {
    note(`Not saved: ${e.message}`, true);
    return false;
  }
}

async function apply() {
  if (dirty() && !(await save())) return;
  $("calApply").disabled = true;
  note("Applying to the headset…");
  try {
    const d = await api("POST", "/api/calibration/apply");
    if (d.device) S.device = d.device;
    note("Applied: the headset saved the mic map and motor angles (they survive a reboot).");
  } catch (e) {
    note(`Not applied: ${e.message}`, true);
  }
  renderAll();
}

function note(text, bad) {
  S.msg = text; S.msgBad = !!bad;
  const n = $("calMsg");
  if (n) { n.textContent = text; n.className = "hint cal-msg" + (bad ? " bad" : ""); }
}

// ---------------------------------------------------------------- build (once)
function build() {
  if (S.built) return;
  S.built = true;
  const posOpts = `<option value="" hidden>—</option>` + POSITIONS.map((p) => `<option value="${p}">${POS_LABEL[p]}</option>`).join("");
  $("calib").innerHTML = `
  <div class="card-h cal-h">
    <h2>Calibration</h2>
    <span class="hint" id="calNote">which mic and which motor is where on the headset</span>
    <span class="cal-actions">
      <span class="cal-state" id="calState"></span>
      <button id="calSave" class="btn" title="write data/calibration.json">Save</button>
      <button id="calApply" class="btn" title="send the mic map and motor angles to the headset, which keeps them">Apply to headset</button>
      <button id="calClose" class="btn">Close</button>
    </span>
  </div>
  <div class="hint cal-msg" id="calMsg"></div>
  <div class="cal-grid">
    <div class="cal-col">
      <div class="bench-t"><span>Headset <small id="calGeomNote">top view · nose up</small></span></div>
      <svg id="calSvg" class="cal-svg" aria-label="headset, top view, nose up"></svg>
      <div class="hint" id="calHelp">Click a motor (M1–M8 here or in the list), then the hole it is mounted in.</div>
      <dl class="dev cal-dev" id="calDev"></dl>
    </div>
    <div class="cal-col">
      <div class="bench-t"><span>Microphones <small>tick the connected headers, set where each sits</small></span>
        <button id="calMicDefault" class="btn" title="MK2 back, MK3 left, MK4 right, MK5 top; MK1 not connected">Default</button></div>
      <div class="cal-mics">${Array.from({ length: N_ROWS }, (_, k) => `
        <div class="cal-mic" id="calMic${k}">
          <label class="cal-mic-k" title="tick if a mic is plugged into this header">
            <input type="checkbox" id="calOn${k}"><b>${HEADERS[k].id}</b>
            <span class="cal-hint" id="calW${k}"></span><span class="cal-tag" id="calTag${k}"></span></label>
          <div class="slot-bar"><i class="rms" id="calRms${k}"></i><i class="peak" id="calPk${k}"></i></div>
          <span class="cal-db" id="calDb${k}">—</span>
          <select id="calPos${k}" title="where this mic sits on the head">${posOpts}</select>
        </div>`).join("")}
      </div>
      <div class="cal-slot"><label for="calSlot">Slot order</label>
        <select id="calSlot" title="which SEL phase the board reads first on each data line: which DMA word each header lands in (a line with one mic reads it in both words, so either works for the DSP)">
          ${Object.entries(SLOT_ORDERS).map(([k, v]) => `<option value="${k}">${v}</option>`).join("")}</select></div>
      <div class="hint" id="calMicHint"></div>
      <div class="bench-t cal-sub"><span>Auto-calibrate <small>a sharp clap from each side: ${CAPTURE_S} s of recording, then the headset sends it (~${2 * CAPTURE_S + 1} s more)</small></span>
        <button id="calClear" class="btn" title="forget the recordings">Clear</button></div>
      <ol class="cal-steps" id="calSteps">${STEPS.map((s, i) => `
        <li id="calStep${i}" data-step="${i}"><span class="n">${i + 1}</span><span class="t">${POS_LABEL[s.dir]}${s.req ? "" : " <small>opt.</small>"}</span>
          <span class="st" id="calStepSt${i}">—</span><button class="btn" id="calRec${i}" data-rec="${i}">Record ${CAPTURE_S} s</button></li>`).join("")}
      </ol>
      <div class="hint cal-steptext" id="calStepText"></div>
      <div class="rec-prog cal-prog"><i id="calProg"></i></div>
      <div class="cal-solve"><button id="calSolve" class="btn">Solve</button><span class="hint" id="calSolveNote">needs Front, Right and Left</span></div>
      <div class="cal-prop" id="calProp"></div>
    </div>
    <div class="cal-col">
      <div class="bench-t"><span>Motors <small>tick the connected outputs, set each one's hole</small></span>
        <span class="cal-btns"><button id="calFeel" class="btn" title="buzz each connected motor in turn; click the hole where you feel it">Find by feel</button>
        <button id="calMotDefault" class="btn" title="all eight connected, each in the hole nearest its pins.h direction">Default</button></span></div>
      <div class="cal-mots">${MOTORS.map((m, i) => `
        <div class="cal-mot" id="calMot${i}">
          <input type="checkbox" id="calMotOn${i}" title="tick if a motor is wired to ${m}; the headset never drives the others">
          <button class="cal-mk" id="calMk${i}" data-motor="${i}" title="select, then click its hole on the diagram">${m}<small>${MOTOR_NAME[i]}</small></button>
          <select id="calHole${i}" title="the hole this motor is mounted in"></select>
          <span class="cal-deg" id="calDeg${i}">—</span>
          <button class="btn" id="calBuzz${i}" data-buzz="${i}" title="${BUZZ_MS / 1000} s at duty ${DUTY}/255">Buzz</button>
        </div>`).join("")}
      </div>
      <div class="cal-feel" id="calFeelBox"></div>
    </div>
  </div>`;

  $("calSave").addEventListener("click", save);
  $("calApply").addEventListener("click", apply);
  $("calClose").addEventListener("click", () => { if (location.hash === "#calibrate") history.replaceState(null, "", location.pathname + location.search); calibShow(false); });
  $("calMicDefault").addEventListener("click", () => { S.cal.mic_words = pcbWords(); S.cal.mic_source = "default"; renderAll(); });
  $("calMotDefault").addEventListener("click", () => { const d = defaultCalibration(); S.cal.motors = d.motors; S.cal.motor_source = "default"; renderAll(); });
  for (let k = 0; k < N_ROWS; k++) {
    $("calOn" + k).addEventListener("change", (e) => setHeaderOn(k, e.target.checked));
    $("calPos" + k).addEventListener("change", (e) => { if (e.target.value) setMicPos(headerWord(k), e.target.value); });
  }
  $("calSlot").addEventListener("change", (e) => setSlotOrder(e.target.value));
  $("calSteps").addEventListener("click", (e) => {
    const b = e.target.closest("[data-rec]");
    if (b) { record(Number(b.dataset.rec)); return; }
    const li = e.target.closest("[data-step]");
    if (li) { S.step = Number(li.dataset.step); renderSteps(); }
  });
  $("calClear").addEventListener("click", clearCaptures);
  $("calSolve").addEventListener("click", solve);
  $("calProp").addEventListener("click", (e) => { if (e.target.id === "calUse") useProposal(); });
  for (let i = 0; i < MOTORS.length; i++) {
    $("calMotOn" + i).addEventListener("change", (e) => setMotorOn(i, e.target.checked));
    $("calHole" + i).addEventListener("change", (e) => { if (e.target.value) assignMotor(i, e.target.value); });
    $("calMk" + i).addEventListener("click", () => selectMotor(i));
    $("calBuzz" + i).addEventListener("click", () => buzz(i));
  }
  $("calFeel").addEventListener("click", () => (S.feel ? feelStop(false) : feelStart()));
  $("calFeelBox").addEventListener("click", (e) => {
    const a = e.target.dataset && e.target.dataset.feel;
    if (a === "skip") feelNext();
    else if (a === "none") { assignMotor(S.feel.i, null); feelNext(); }
    else if (a === "again") feelBuzz();
    else if (a === "stop") feelStop(false);
  });
  $("calSvg").addEventListener("click", (e) => {
    const h = e.target.closest("[data-hole]"), m = e.target.closest("[data-motor]");
    if (h) holeClicked(h.dataset.hole);
    else if (m) selectMotor(Number(m.dataset.motor));
  });
}

function buildHoleOptions() {
  for (let i = 0; i < MOTORS.length; i++) {
    const sel = $("calHole" + i);
    sel.innerHTML = `<option value="" hidden>—</option>` +
      S.geom.holes.map((h) => `<option value="${escapeHtml(h.id)}">${escapeHtml(h.id)} · ${fmtAng(h.theta_deg)}</option>`).join("") +
      `<option value="__custom" hidden disabled>custom angle</option>`;
  }
}

// ---------------------------------------------------------------- edits
function setMicPos(w, pos) {
  const mw = S.cal.mic_words, prev = posOfWord(w);
  if (S.msg === ALL_IN_USE || S.msg.startsWith(FRONT_BACK)) note("");
  if (pos) {
    const other = mw[pos];
    mw[pos] = w;
    if (prev && prev !== pos) mw[prev] = other != null && other !== w ? other : null;   // swap with the word that had it
    // front and back share channel 0: choosing one drops the header that has the other
    const opp = OTHER_SIDE[pos], ow = opp ? mw[opp] : null;
    if (ow != null && ow !== w) {
      mw[opp] = null;
      note(`${FRONT_BACK}: ${headerName(ow)} unticked`);
    }
  } else if (prev) {
    mw[prev] = null;
  }
  S.cal.mic_source = "manual";
  renderAll();
}

function setHeaderOn(k, on) {
  const w = headerWord(k), mw = S.cal.mic_words, cur = posOfWord(w);
  if (!on) {
    if (cur) mw[cur] = null;
  } else if (!cur) {
    // its PCB position if free, else the first free one (back, left, right, top; with a front mic back counts as
    // taken, the two sharing one channel); with all four taken it stays off
    const free = (p) => mw[p] == null && !(OTHER_SIDE[p] && mw[OTHER_SIDE[p]] != null), pcb = HEADERS[k].pos;
    const p = pcb && free(pcb) ? pcb : AUTO_POS.find(free);
    if (!p) { note(ALL_IN_USE, true); renderMics(); return; }
    mw[p] = w;
  }
  if (S.msg === ALL_IN_USE || S.msg.startsWith(FRONT_BACK)) note("");
  S.cal.mic_source = "manual";
  renderAll();
}

function setSlotOrder(order) {
  if (!SLOT_ORDERS[order] || order === S.cal.slot_order) return;
  // each connected header keeps its position: every word swaps phase on its line (header words stay header words)
  S.cal.slot_order = order;
  const mw = {};
  for (const p of POSITIONS) mw[p] = S.cal.mic_words[p] == null ? null : S.cal.mic_words[p] ^ 1;
  S.cal.mic_words = mw;
  S.cal.mic_source = "manual";
  renderAll();
}

function setMotorOn(i, on) {
  const M = MOTORS[i], mo = S.cal.motors;
  if (!on) { assignMotor(i, null); return; }
  if (mo[M]) return;
  // back in the hole it had, else the one nearest its pins.h direction, else any free hole
  const taken = new Set(MOTORS.map((k) => mo[k] && mo[k].hole).filter(Boolean));
  const near = nearestHole(MOTOR_DEFAULT_DEG[i]);
  let id = [S.lastHole[M], near && near.id].find((x) => x && holeById(x) && !taken.has(x));
  if (!id) { const h = S.geom.holes.find((x) => !taken.has(x.id)); id = h ? h.id : null; }
  if (id) { assignMotor(i, id); return; }
  mo[M] = { hole: null, theta_deg: MOTOR_DEFAULT_DEG[i] };   // every hole taken: its nominal angle
  S.cal.motor_source = "manual";
  renderAll();
}

function assignMotor(i, holeId) {
  const M = MOTORS[i], mo = S.cal.motors, prev = mo[M];
  if (holeId == null) {
    if (prev && prev.hole) S.lastHole[M] = prev.hole;
    mo[M] = null;
  } else {
    const h = holeById(holeId);
    if (!h) return;
    for (const k of MOTORS) {                      // one motor per hole: the motor already there takes this one's old hole
      if (k !== M && mo[k] && mo[k].hole === holeId) {
        const ph = prev && prev.hole ? holeById(prev.hole) : null;
        mo[k] = ph ? { hole: ph.id, theta_deg: round1(ph.theta_deg) } : null;
      }
    }
    mo[M] = { hole: h.id, theta_deg: round1(h.theta_deg) };
  }
  S.cal.motor_source = "manual";
  renderAll();
}

function selectMotor(i) {
  if (S.feel) return;                              // the sequence decides which motor is next
  S.selMotor = S.selMotor === i ? null : i;
  renderMotors(); renderDiagram();
}

function holeClicked(id) {
  if (S.feel) { assignMotor(S.feel.i, id); feelNext(); return; }
  if (S.selMotor == null) { $("calHelp").textContent = `Hole ${id}: select a motor first (M1–M8), then click its hole.`; return; }
  const i = S.selMotor;
  S.selMotor = null;
  assignMotor(i, id);
  $("calHelp").textContent = `${MOTORS[i]} → ${id}. Save, then Apply to headset.`;
}

// ---------------------------------------------------------------- motors on the headset
function haptics(i, hz) {
  sendCmd({ type: "haptics", duty: MOTORS.map((_, k) => (k === i ? DUTY : 0)), hz: hz || 0 });
}
function buzzOff() {
  if (S.buzzT) { clearTimeout(S.buzzT); S.buzzT = null; }
  if (S.buzzing != null) sendCmd({ type: "haptics", duty: null });
  S.buzzing = null;
}
function buzz(i) {
  if (!deviceOk() || S.feel || S.capturing) return;
  buzzOff();
  haptics(i, 0);
  S.buzzing = i;
  S.buzzT = setTimeout(() => { S.buzzT = null; buzzOff(); renderMotors(); renderDiagram(); }, BUZZ_MS);
  renderMotors(); renderDiagram();
}

const nextConnected = (from) => { for (let i = from; i < MOTORS.length; i++) if (S.cal.motors[MOTORS[i]]) return i; return -1; };
function feelStart() {
  if (!deviceOk() || S.capturing) return;
  const i = nextConnected(0);
  if (i < 0) { renderFeel("No motor is ticked as connected."); return; }
  buzzOff();
  S.selMotor = null;
  S.feel = { i };
  feelBuzz();
}
function feelBuzz() {                              // pulsed at 2 Hz, repeated every second (the board drops 'h' after 3 s)
  if (S.feelTimer) clearInterval(S.feelTimer);
  const i = S.feel.i;
  haptics(i, 2);
  S.feelTimer = setInterval(() => haptics(i, 2), 1000);
  S.buzzing = i;
  renderMotors(); renderDiagram(); renderFeel();
}
function feelNext() {
  if (!S.feel) return;
  S.feel.i = nextConnected(S.feel.i + 1);
  if (S.feel.i < 0) feelStop(true); else feelBuzz();
}
function feelStop(done) {
  if (S.feelTimer) { clearInterval(S.feelTimer); S.feelTimer = null; }
  const was = !!S.feel;
  S.feel = null;
  if (was) sendCmd({ type: "haptics", duty: null });
  S.buzzing = null;
  renderMotors(); renderDiagram(); renderFeel(done ? "Done: every connected output visited. Save, then Apply to headset." : "");
}

// ---------------------------------------------------------------- auto-calibration
async function record(idx) {
  if (S.capturing || !deviceOk()) return;
  const dir = STEPS[idx].dir;
  feelStop(false); buzzOff();
  S.step = idx;
  S.capturing = dir; S.capT0 = performance.now(); S.capElapsed = 0;   // capElapsed: seconds of audio received so far
  S.captures[dir] = { pending: true };
  S.solve = null;
  if (S.capTimer) clearInterval(S.capTimer);
  S.capTimer = setInterval(renderProgress, 100);
  renderAll();
  try {
    S.captures[dir] = await api("POST", "/api/calibration/capture", { direction: dir, seconds: CAPTURE_S });
  } catch (e) {
    S.captures[dir] = { error: e.message };
  }
  S.capturing = null;
  clearInterval(S.capTimer); S.capTimer = null;
  const next = STEPS.findIndex((s) => s.req && !captureOk(s.dir));
  if (captureOk(dir) && next >= 0) S.step = next;
  renderAll();
}

const captureOk = (dir) => { const c = S.captures[dir]; return !!c && !c.pending && !c.error && (c.stored || !c.event || c.event.found !== false); };
const canSolve = () => S.apiOk && !S.capturing && ["front", "right", "left"].every((d) => S.captures[d] && !S.captures[d].error && !S.captures[d].pending);

async function clearCaptures() {
  try { await api("DELETE", "/api/calibration/capture"); } catch (e) { note(`Could not clear on the server: ${e.message}`, true); }
  S.captures = {}; S.solve = null; S.step = 0;
  renderAll();
}

async function solve() {
  $("calSolve").disabled = true;
  try {   // the slot order and the connected headers decide which word of a line with one mic (both read it) is used
    S.solve = await api("POST", "/api/calibration/solve", { slot_order: S.cal.slot_order, mic_words: S.cal.mic_words });
  } catch (e) {
    S.solve = { error: e.message };
  }
  renderAll();
}

// The slot order a proposal implies. Line 2 has one header (MK5): a mic heard on just one word of it shows which
// phase MK5 is read in (w4: left first, w5: right first). Usually the empty phase mirrors the mic, so both words read
// it and nothing is implied: the proposal then used the order it was solved with (d.slot_order), and the current one
// stands. Lines 0 and 1 have a header on both phases and cannot tell.
function impliedOrder(d) {
  const mir = (d && d.mirrors) || {}, votes = new Set();
  for (const v of Object.values((d && d.mic_words) || {})) {
    if (v == null) continue;
    const w = Number(v), hs = HEADERS.map((h, k) => k).filter((k) => HEADERS[k].line === w >> 1);
    if (hs.length !== 1 || mir[String(w)] != null || mir[String(w ^ 1)] != null) continue;
    for (const o of Object.keys(SLOT_ORDERS)) if (headerWord(hs[0], o) === w) votes.add(o);
  }
  if (votes.size === 1) return [...votes][0];
  return d && SLOT_ORDERS[d.slot_order] ? d.slot_order : S.cal.slot_order;
}

function useProposal() {
  if (!S.solve || !S.solve.mic_words) return;
  const mw = {};
  for (const p of POSITIONS) mw[p] = S.solve.mic_words[p] != null ? Number(S.solve.mic_words[p]) : null;
  S.cal.slot_order = impliedOrder(S.solve);
  const moved = onHeaders(mw, S.cal.slot_order);
  S.cal.mic_words = mw;
  S.cal.mic_source = "auto";
  note("Auto-calibration result copied into the mic map (the mics it heard are ticked as connected)"
       + (moved.length ? `; ${moved.join(", ")} moved to its line's header (the other slot has none)` : "")
       + ": Save, then Apply to headset.");
  renderAll();
}

// ---------------------------------------------------------------- render
function renderAll() {
  if (!ready()) return;
  renderMics(); renderSteps(); renderProposal(); renderMotors(); renderDiagram(); renderDevice(); renderEnabled();
  if (S.msg) note(S.msg, S.msgBad);
}

function wordLevel(w) { return S.words.find((x) => x.w === w) || null; }
const wordFlat = (s) => !s || s.rms <= -100 || Math.abs(s.dc) >= 0.5;
function mirrorPair(w) {                           // the other slot of the line, if both read the same signal
  const o = w ^ 1, a = wordLevel(w), b = wordLevel(o);
  if (!a || !b || wordFlat(a) || wordFlat(b)) return null;
  return Math.abs(a.rms - b.rms) < 0.2 && Math.abs(a.peak - b.peak) < 0.2 && Math.abs(a.dc - b.dc) < 0.003 ? o : null;
}

function renderMics() {
  if (!ready()) return;
  const fresh = Date.now() - S.lastLevels < 2500;
  const slot = $("calSlot");
  if (document.activeElement !== slot) slot.value = S.cal.slot_order;
  let warn = 0;
  for (let k = 0; k < N_ROWS; k++) {
    const w = headerWord(k), pos = posOfWord(w), on = !!pos, row = $("calMic" + k);
    const s = fresh ? wordLevel(w) : null, live = !!s && !wordFlat(s);
    $("calPk" + k).style.width = s ? `${100 * Math.max(0, Math.min(1, (s.peak - DB_FLOOR) / -DB_FLOOR))}%` : "0";
    $("calRms" + k).style.width = s ? `${100 * Math.max(0, Math.min(1, (s.rms - DB_FLOOR) / -DB_FLOOR))}%` : "0";
    $("calDb" + k).textContent = s ? (live ? `${s.peak.toFixed(0)} dBFS` : "silent") : "—";
    const wl = $("calW" + k);
    wl.textContent = `w${w}`;
    wl.title = `DMA word w${w}: ${wordLine(w)} (${SLOT_ORDERS[S.cal.slot_order]})`;
    // A line with one mic reads it in both phases, so an empty header's bar moves too: say so rather than warn. (MK5's
    // partner word on line 2 has no header: no tag there, MK5 is the mic.)
    const mir = live ? mirrorPair(w) : null, tag = $("calTag" + k);
    const mirHdr = mir != null && headerName(mir) !== "no header";
    let t = "", why = "";
    if (on && s && !live) { t = "no signal"; why = "ticked as connected, but this slot reads flat: check the plug, or untick it"; warn++; }
    else if (mirHdr) { t = `= ${headerName(mir)}`; why = `reads the same as ${headerName(mir)} (w${mir}): one mic on this line, the empty phase mirrors it`; }
    else if (!on && live) { t = "hears sound"; why = "not ticked, but a mic seems plugged in here: tick it if it is"; warn++; }
    tag.textContent = t; tag.title = why;
    $("calOn" + k).checked = on;
    const sel = $("calPos" + k);
    sel.disabled = !on;
    if (document.activeElement !== sel) sel.value = pos || "";
    row.classList.toggle("off", !on);
    row.classList.toggle("flat", !!s && !live);
    row.classList.toggle("hot", live && s.peak > -20);
    row.classList.toggle("assigned", on);
  }
  const n = POSITIONS.filter((p) => wordAt(p) != null).length;
  $("calMicHint").textContent = `${n} of ${N_CH} positions have a mic (${S.cal.mic_source}). ${dspSummary()} `
    + (!deviceOk() ? "Levels need the headset: start the dashboard with --port or --tcp."
       : !fresh ? "Waiting for mic levels from the headset…"
       : warn ? "Check the flagged rows." : "Tap a mic: its bar jumps.");
}

// What the board makes of the connected mics: detection from their mean; the direction from the robust solver,
// which fits the best three or four of them each frame and, with only three, assumes the drone is above the head
// where it has to choose between two mirror-image directions. A front mic does exactly what a back one does (the
// geometry is mirror-symmetric).
function dspSummary() {
  const on = POSITIONS.filter((p) => wordAt(p) != null), has = (p) => on.includes(p);
  if (!on.length) return "No mic connected: the headset cannot detect.";
  if (on.length <= 2) return "Detection only, no bearing (that needs three mics).";
  if (on.length >= 4) return "Bearing and elevation. If one mic is weak or drops out for a moment, the headset carries on with the best three.";
  if (!has("front") && !has("back")) return "Front and back cannot be told apart without a front or back mic: no reliable bearing (ambiguous).";
  if (!has("top")) return "Bearing; elevation only when the drone is clearly above the head (assumed above).";
  return `Bearing and elevation when the drone is clearly above the head; otherwise ambiguous, no bearing (no ${has("left") ? "right" : "left"} mic).`;
}

function renderSteps() {
  if (!ready()) return;
  for (let i = 0; i < STEPS.length; i++) {
    const s = STEPS[i], c = S.captures[s.dir], li = $("calStep" + i), st = $("calStepSt" + i);
    let txt = "—", cls = "";
    if (S.capturing === s.dir) { txt = `recording… ${S.capElapsed ? S.capElapsed.toFixed(1) + " s" : ""}`; cls = "busy"; }
    else if (!c) txt = s.req ? "needed" : "optional";
    else if (c.error) { txt = c.error; cls = "bad"; }
    else if (c.stored) { txt = "recorded earlier (kept on the server)"; cls = "done"; }
    else if (c.event && c.event.found === false) { txt = "no clap found: clap louder or closer, then record again"; cls = "bad"; }
    else if (c.event) {
      const live = (c.words || []).filter((x) => x.live && x.mirror_of == null).map((x) => "w" + x.w);
      txt = `clap at ${Number(c.event.t_s).toFixed(1)} s, ${Number(c.event.peak_dbfs).toFixed(0)} dBFS · mics ${live.join(" ") || "none"}`;
      cls = "done";
    }
    st.textContent = txt; st.title = txt;
    li.className = [cls, i === S.step ? "cur" : ""].join(" ").trim();
  }
  const s = STEPS[S.step] || STEPS[0];
  $("calStepText").textContent = `${S.step + 1}. ${POS_LABEL[s.dir]}: ${s.text}`;
  renderProgress();
}

function renderProgress() {
  const bar = $("calProg");
  if (!bar || !ready()) return;
  if (S.capturing) {
    // The headset records CAPTURE_S s into its PSRAM, then sends it at USB speed (6 words in real time are over
    // twice what its USB carries): clap while "recording", then wait while it transfers. The bar is audio received.
    const wall = (performance.now() - S.capT0) / 1000, got = S.capElapsed || 0;
    bar.style.width = `${100 * Math.min(1, got / CAPTURE_S)}%`;
    bar.parentElement.classList.remove("ok");
    const i = STEPS.findIndex((x) => x.dir === S.capturing), st = i >= 0 ? $("calStepSt" + i) : null;
    if (st) st.textContent = wall < CAPTURE_S + 0.3
      ? `recording: clap now (${Math.max(0, CAPTURE_S - wall).toFixed(1)} s left)`
      : `transferring from the headset… ${got.toFixed(1)} / ${CAPTURE_S} s`;
  } else {
    const c = S.captures[(STEPS[S.step] || {}).dir];
    bar.style.width = c && !c.error && !c.pending ? "100%" : "0";
    bar.parentElement.classList.toggle("ok", !!c && captureOk((STEPS[S.step] || {}).dir));
  }
}

function wordName(w, order) { return w == null ? "—" : `w${w} <small>${headerName(w, order)}</small>`; }

function renderProposal() {
  if (!ready()) return;
  const box = $("calProp"), d = S.solve;
  $("calSolveNote").textContent = canSolve() ? (d && !d.error ? "" : "ready: Front, Right and Left recorded") : "needs Front, Right and Left";
  if (!d) { box.innerHTML = ""; return; }
  if (d.error) { box.innerHTML = `<div class="hint bad">Solve failed: ${escapeHtml(d.error)}</div>`; return; }
  const conf = Number(d.confidence || 0), cc = conf >= 0.7 ? "good" : conf < 0.4 ? "bad" : "";
  const order = impliedOrder(d);
  const rows = POSITIONS.map((p) => {
    const w = d.mic_words ? d.mic_words[p] : null, cur = wordAt(p);
    return `<tr><td>${POS_LABEL[p]}</td><td>${wordName(w, order)}</td><td class="hint">${w === cur ? "same as now" : `now ${cur == null ? "—" : "w" + cur}`}</td></tr>`;
  }).join("") + (order !== S.cal.slot_order ? `<tr><td>Slot order</td><td colspan="2">${SLOT_ORDERS[order]} <small>(now ${SLOT_ORDERS[S.cal.slot_order]})</small></td></tr>` : "");
  const orders = Object.entries(d.per_direction || {}).map(([dir, v]) =>
    `<li><b>${POS_LABEL[dir] || escapeHtml(dir)}</b> first → last: ${(v.order || []).map((w) => "w" + w).join(" → ") || "—"}</li>`).join("");
  const notes = (d.notes || []).map((n) => `<li>${escapeHtml(String(n))}</li>`).join("");
  box.innerHTML = `
    <div class="cal-prop-h"><span>Proposal · confidence <b class="${cc}">${Math.round(100 * conf)}%</b></span>
      <button class="btn" id="calUse">Use this</button></div>
    <table class="cal-table">${rows}</table>
    ${orders ? `<ul class="cal-notes">${orders}</ul>` : ""}
    ${notes ? `<ul class="cal-notes">${notes}</ul>` : ""}`;
}

function renderMotors() {
  if (!ready()) return;
  for (let i = 0; i < MOTORS.length; i++) {
    const m = S.cal.motors[MOTORS[i]], sel = $("calHole" + i);
    $("calMotOn" + i).checked = !!m;
    sel.disabled = !m;
    if (document.activeElement !== sel) {
      const custom = sel.querySelector('option[value="__custom"]');
      if (m && !m.hole) { custom.hidden = false; custom.textContent = `custom ${fmtAng(m.theta_deg)}`; sel.value = "__custom"; }
      else { custom.hidden = true; sel.value = m && m.hole ? m.hole : ""; }
    }
    $("calDeg" + i).textContent = m ? fmtAng(m.theta_deg) : "—";
    const row = $("calMot" + i);
    row.classList.toggle("off", !m);
    row.classList.toggle("sel", S.selMotor === i || (S.feel && S.feel.i === i));
    row.classList.toggle("buzz", S.buzzing === i);
  }
}

function renderFeel(doneText) {
  if (!ready()) return;
  const box = $("calFeelBox");
  $("calFeel").textContent = S.feel ? "Stop" : "Find by feel";
  $("calFeel").classList.toggle("on", !!S.feel);
  if (!S.feel) { box.innerHTML = doneText ? `<div class="hint">${escapeHtml(doneText)}</div>` : ""; return; }
  const i = S.feel.i;
  box.innerHTML = `<div class="cal-feel-h"><b>${MOTORS[i]}</b> (${MOTOR_NAME[i]} output) is pulsing: click the hole on the diagram where you feel it.</div>
    <div class="cal-btns"><button class="btn" data-feel="again">Buzz again</button><button class="btn" data-feel="skip">Skip</button>
    <button class="btn" data-feel="none" title="untick it: nothing buzzed">Not connected</button><button class="btn" data-feel="stop">Stop</button></div>`;
}

// ---- head diagram (top view, nose up: x right, y forward -> screen y = -y)
const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs, text) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
}

function micSite(pos, bb) {
  // the STL's mic mounts if the geometry has them; else schematic spots well inside the band (the cardinal holes sit on
  // its edge, so the mic marks and labels must not land on them)
  const s = (S.geom.mic_sites || []).find((x) => x.position === pos);
  if (s && isFinite(s.x_mm) && isFinite(s.y_mm)) return { x: Number(s.x_mm), y: Number(s.y_mm) };
  const cx = (bb.x0 + bb.x1) / 2, cy = (bb.y0 + bb.y1) / 2, ix = 0.45 * (bb.x1 - bb.x0) / 2, iy = 0.55 * (bb.y1 - bb.y0) / 2;
  return { front: { x: cx, y: cy + iy }, back: { x: cx, y: cy - iy }, left: { x: cx - ix, y: cy },
           right: { x: cx + ix, y: cy }, top: { x: cx, y: cy } }[pos];
}

function renderDiagram() {
  if (!ready()) return;
  const svg = $("calSvg");
  if (!svg || !S.geom) return;
  const g = S.geom, pts = (g.outline && g.outline.length ? g.outline : []).map((p) => [Number(p[0]), Number(p[1])]);
  const all = pts.concat(g.holes.map((h) => [Number(h.x_mm), Number(h.y_mm)]));
  const bb = { x0: Math.min(...all.map((p) => p[0])), x1: Math.max(...all.map((p) => p[0])),
               y0: Math.min(...all.map((p) => p[1])), y1: Math.max(...all.map((p) => p[1])) };
  const pad = 48;
  svg.setAttribute("viewBox", `${bb.x0 - pad} ${-bb.y1 - pad} ${bb.x1 - bb.x0 + 2 * pad} ${bb.y1 - bb.y0 + 2 * pad}`);
  svg.classList.toggle("picking", S.selMotor != null || !!S.feel);
  svg.innerHTML = "";
  const cx = (bb.x0 + bb.x1) / 2, cy = (bb.y0 + bb.y1) / 2;
  if (pts.length > 2) svg.append(el("path", { class: "cal-band", d: "M" + pts.map((p) => `${p[0].toFixed(1)} ${(-p[1]).toFixed(1)}`).join(" L") + " Z" }));
  // nose / front marker
  // nose / front marker, beyond the badge of a front hole (badges sit 20 mm outside their hole)
  svg.append(el("path", { class: "cal-nose", d: `M${cx - 8} ${-bb.y1 - 31} L${cx} ${-bb.y1 - 43} L${cx + 8} ${-bb.y1 - 31} Z` }));
  svg.append(el("text", { class: "cal-t front", x: cx + 28, y: -bb.y1 - 33 }, "FRONT"));
  // mic sites (front and back both drawn: the one without a mic, usually front, as unused)
  for (const p of POSITIONS) {
    const s = micSite(p, bb), w = wordAt(p), lv = w != null ? wordLevel(w) : null;
    const hot = lv && !wordFlat(lv) && lv.peak > -20 && Date.now() - S.lastLevels < 2500;
    const gr = el("g", { class: `cal-site${w == null ? " none" : ""}${hot ? " hot" : ""}` });
    gr.append(el("circle", { cx: s.x, cy: -s.y, r: 7 }));
    const ly = s.y - 14;                                         // label under the mark
    gr.append(el("text", { class: "cal-t site", x: s.x, y: -ly - 1 }, POS_LABEL[p].toUpperCase()));
    gr.append(el("text", { class: "cal-t sub", x: s.x, y: -ly + 8 }, w == null ? "no mic" : `${headerName(w)}${DSP_CH[p] ? " · " + DSP_CH[p] : ""}`));
    gr.append(el("title", {}, w == null ? `${POS_LABEL[p]}: no mic connected` : `${POS_LABEL[p]}: ${headerName(w)} (DMA word w${w})`));
    svg.append(gr);
  }
  // holes and the motors in them
  const motorIn = {};
  MOTORS.forEach((m, i) => { const v = S.cal.motors[m]; if (v && v.hole) motorIn[v.hole] = i; });
  for (const h of g.holes) {
    const x = Number(h.x_mm), y = Number(h.y_mm), r = Math.max(4.5, Number(h.diameter_mm || 8) / 2);
    const i = motorIn[h.id], taken = i != null;
    const gr = el("g", { class: `cal-hole${taken ? " taken" : ""}${taken && S.buzzing === i ? " buzz" : ""}`, "data-hole": h.id });
    gr.append(el("circle", { cx: x, cy: -y, r }));
    const dx = x - cx, dy = y - cy, dn = Math.hypot(dx, dy) || 1;
    gr.append(el("text", { class: "cal-t hole", x: x - 13 * dx / dn, y: -(y - 13 * dy / dn) + 2.5 }, h.id));
    gr.append(el("title", {}, `${h.id}: ${fmtAng(Number(h.theta_deg))}${taken ? ` · ${MOTORS[i]}` : " · empty"}`));
    svg.append(gr);
    if (taken) {
      const bx = x + 20 * dx / dn, by = y + 20 * dy / dn;
      const b = el("g", { class: `cal-mb${S.buzzing === i ? " buzz" : ""}${S.selMotor === i ? " sel" : ""}`, "data-motor": i });
      b.append(el("rect", { x: bx - 12, y: -by - 7.5, width: 24, height: 15, rx: 4 }));
      b.append(el("text", { class: "cal-t mb", x: bx, y: -by + 3.5 }, MOTORS[i]));
      svg.append(b);
    }
  }
  // motors with an angle but no hole (custom), drawn on the outline direction
  MOTORS.forEach((m, i) => {
    const v = S.cal.motors[m];
    if (!v || v.hole) return;
    const a = v.theta_deg * Math.PI / 180, rr = Math.max(bb.x1 - cx, bb.y1 - cy) + 18;
    const bx = cx + rr * Math.sin(a), by = cy + rr * Math.cos(a);
    const b = el("g", { class: `cal-mb custom${S.buzzing === i ? " buzz" : ""}`, "data-motor": i });
    b.append(el("rect", { x: bx - 12, y: -by - 7.5, width: 24, height: 15, rx: 4 }));
    b.append(el("text", { class: "cal-t mb", x: bx, y: -by + 3.5 }, m));
    svg.append(b);
  });
  const note = g.placeholder ? `placeholder layout (${g.why || "no geometry file"})` : `${g.source || "STL"} · ${g.holes.length} holes · top view, nose up`;
  $("calGeomNote").textContent = note;
  const help = $("calHelp");
  if (S.feel) help.textContent = `${MOTORS[S.feel.i]} is pulsing: click the hole where you feel it.`;
  else if (S.selMotor != null) help.textContent = `${MOTORS[S.selMotor]} selected: click the hole it is mounted in.`;
  else if (/pulsing|selected/.test(help.textContent)) help.textContent = "Click a motor (M1–M8 here or in the list), then the hole it is mounted in.";
}

function renderDevice() {
  if (!ready()) return;
  const rows = [];
  const dev = S.device, live = deviceOk();
  rows.push(["Headset", live ? "connected" : S.isDevice ? "not connected" : "not the data source (demo / replay)", live ? "good" : "bad"]);
  if (dev && dev.mics) {                          // channel 0 as the board names it: F (front) or B (back)
    const mics = [ch0Of(dev.mics), "L", "R", "T"].map((c) => `${c}=${dev.mics[c] == null || dev.mics[c] < 0 ? "–" : "w" + dev.mics[c]}`).join(" ");
    rows.push(["Mics on it", `${mics}${dev.src ? " · " + dev.src : ""}`]);
  }
  if (dev && dev.motors) {
    rows.push(["Motors on it", MOTORS.map((m) => `${m} ${dev.motors[m] == null ? "off" : fmtAng(Number(dev.motors[m]))}`).join(" · ")]);
  }
  if (S.cal) {
    rows.push(["File", !S.apiOk ? "API unavailable" : dirty() ? "unsaved changes" : "saved", dirty() || !S.apiOk ? "bad" : "good"]);
    if (dev && dev.mics && dev.motors) {
      const same = sameAsDevice();
      rows.push(["Headset vs file", same ? "in sync" : "differs: Apply to headset", same ? "good" : "bad"]);
    }
  }
  $("calDev").innerHTML = rows.map(([k, v, c]) => `<dt>${k}</dt><dd class="${c || ""}" title="${escapeHtml(String(v))}">${escapeHtml(String(v))}</dd>`).join("");
  $("calState").textContent = !S.apiOk ? "" : dirty() ? "unsaved" : "saved";
  $("calState").className = "cal-state" + (dirty() ? " bad" : "");
}

const ch0Of = (mics) => (mics && "F" in mics && !("B" in mics) ? "F" : "B");
function sameAsDevice() {                         // what Apply would send: channel 0's side and word, L, R, T, motors
  const dev = S.device, side = wordAt("front") != null ? "front" : "back";
  if (ch0Of(dev.mics) !== DSP_CH[side]) return false;
  for (const [pos, ch] of [[side, DSP_CH[side]], ["left", "L"], ["right", "R"], ["top", "T"]]) {
    const a = wordAt(pos), b = dev.mics[ch];
    if ((a == null ? -1 : a) !== (b == null ? -1 : Number(b))) return false;
  }
  for (const m of MOTORS) {
    const a = S.cal.motors[m], b = dev.motors[m];
    if ((a == null) !== (b == null)) return false;
    if (a && Math.abs(wrap(a.theta_deg - Number(b))) > 0.15) return false;
  }
  return true;
}

function renderEnabled() {
  if (!ready()) return;
  const ok = deviceOk();
  $("calNote").textContent = ok ? "which mic and which motor is where on the headset"
    : "connect the headset (dashboard with --port or --tcp) to record, buzz and apply; the map can still be edited and saved";
  $("calSave").disabled = !S.apiOk || !S.cal;
  $("calApply").disabled = !ok || !S.apiOk;
  for (let i = 0; i < STEPS.length; i++) $("calRec" + i).disabled = !ok || !S.apiOk || !!S.capturing;
  $("calSolve").disabled = !canSolve();
  $("calClear").disabled = !!S.capturing || !Object.keys(S.captures).length;
  for (let i = 0; i < MOTORS.length; i++) $("calBuzz" + i).disabled = !ok || !!S.feel || !!S.capturing;
  $("calFeel").disabled = !S.feel && (!ok || !!S.capturing);
}

// ---------------------------------------------------------------- show / hide, hooks from app.js
function levelsKeepalive(on) {
  if (S.levelTimer) { clearInterval(S.levelTimer); S.levelTimer = null; }
  if (on) {
    sendCmd({ type: "levels" });
    S.levelTimer = setInterval(() => sendCmd({ type: "levels" }), 1000);   // the server polls 'L' while we keep asking
  }
}

function calibShow(on) {
  if (on === S.on) return;
  S.on = on;
  if (on) {
    build();
    if (typeof benchState !== "undefined" && benchState.on) benchShow(false);   // one panel at a time
  } else {
    feelStop(false); buzzOff(); S.selMotor = null;
  }
  $("calib").hidden = !on;
  document.body.classList.toggle("calib", on);
  $("calibBtn").classList.toggle("on", on);
  try { localStorage.setItem("calib", on ? "1" : "0"); } catch (e) { /* storage may be unavailable */ }
  levelsKeepalive(on);
  if (on) { if (!S.loaded) load(); else renderAll(); }
}

window.calibShow = calibShow;

window.onCalibBench = function (d) {
  if (!S.built) return;
  if (d.kind === "levels") {
    S.words = d.words || []; S.lastLevels = Date.now();
    if (S.on) { renderMics(); updateDiagramLive(); }
  } else if (d.kind === "cal_capture") {
    if (d.elapsed != null) S.capElapsed = Number(d.elapsed);
    if (d.state === "error" && S.capturing === d.direction) note(`Recording ${d.direction}: ${d.error || "error"}`, true);
    if (S.on) renderProgress();
  } else if (d.kind === "cal") {
    if (d.device) S.device = d.device;
    if (S.on) renderDevice();
  }
};

window.onCalibSource = function (src, connected) {
  const isDev = src === "serial", con = !!connected;
  if (isDev === S.isDevice && con === S.connected) return;
  S.isDevice = isDev; S.connected = con;
  if (!deviceOk()) { feelStop(false); S.buzzing = null; }
  if (S.on) { renderEnabled(); renderDevice(); renderMics(); }
};

function updateDiagramLive() {                    // mic-site glow follows the levels without rebuilding the SVG
  const svg = $("calSvg");
  if (!svg || !ready()) return;
  const sites = svg.querySelectorAll(".cal-site");
  POSITIONS.forEach((p, k) => {
    const w = wordAt(p), lv = w != null ? wordLevel(w) : null, g = sites[k];
    if (g) g.classList.toggle("hot", !!lv && !wordFlat(lv) && lv.peak > -20);
  });
}

// ---------------------------------------------------------------- init
$("calibBtn").addEventListener("click", () => calibShow(!S.on));
window.addEventListener("hashchange", () => calibShow(location.hash === "#calibrate"));
let want = location.hash === "#calibrate";
try { want = want || (location.hash !== "#bench" && localStorage.getItem("calib") === "1"); } catch (e) { /* ignore */ }
if (typeof source !== "undefined") window.onCalibSource(source, status && status.connected);
if (want) calibShow(true);
setInterval(() => { if (S.on && S.built && S.cal) { renderDevice(); if (Date.now() - S.lastLevels > 2500) renderMics(); } }, 1000);
})();
