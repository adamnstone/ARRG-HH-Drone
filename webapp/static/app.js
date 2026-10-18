// Headband dashboard: one WebSocket (/ws), ring buffers, hand-rolled Canvas. No dependencies, no build step.
// Angle convention (same as droneears): theta 0 = ahead (up on the radar), +90 = wearer's right, 180 = behind (down).
"use strict";

const WINDOW_MS = 20000;          // chart span
const TRAIL_MS = 2000;            // bearing trail on the radar
const MAXBUF = 1400;              // > 20 s at 47 fr/s
const BATT_FULL_V = 4.20, BATT_EMPTY_V = 3.30;   // firmware/esp32s3/pins.h; percent is linear between them
const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

let CFG = { harm_thresh_db: 5.5, persist_on: 70, persist_frames: 94, ml_thresh: 0.7, f0_min: 120, f0_max: 400,
            mic_pos: [[0, -0.1, 0], [-0.09, 0, 0], [0.09, 0, 0], [0, 0, 0.09]], mic_names: ["B", "L", "R", "T"],
            motor_angles: { F: 0, FR: 45, R: 90, BR: 135, B: 180, BL: -135, L: -90, FL: -45 },
            motor_names: ["F", "FR", "R", "BR", "B", "BL", "L", "FL"], motor_max_on: 3,
            overhead_elev_on: 60, overhead_elev_off: 45, batt_empty_v: 3.3, batt_full_v: 4.2, chunk_us_expect: 21333 };
const buf = { t: [], score: [], hits: [], ml: [], f0: [], det: [], theta: [], conf: [], elev: [] };
let last = null;                  // latest frame
let status = {};
let events = [];
let detT = null, ovhT = null;     // t_ms of the current detection / overhead onset
let lastLatency = null;
let wsOpen = false;
let wsRef = null;                 // the open socket, for bench commands
let benchDsp = null;              // last {"type":"bench","kind":"dsp"} message
let benchMotor = null;            // last motor test reply
let tapTimer = null;              // tap test: keepalive interval while on
let lastFrameWall = 0;
let source = "";
let COL = {};

function readColors() {
  COL = { cool: css("--cool"), alarm: css("--alarm"), muted: css("--muted"), faint: css("--faint"),
          line: css("--line"), line2: css("--line-2"), text: css("--text"), panel: css("--panel") };
}

// ---------------------------------------------------------------- data
function pushFrame(f) {
  if (window.onBatteryFrame) window.onBatteryFrame(f);                         // battery.js
  if (buf.t.length && f.t_ms < buf.t[buf.t.length - 1] - 1000) clearBuffers();   // source restarted
  buf.t.push(f.t_ms); buf.score.push(f.score); buf.hits.push(f.hits); buf.ml.push(f.ml);
  buf.f0.push(f.f0); buf.det.push(f.detected); buf.theta.push(f.theta); buf.conf.push(f.conf);
  buf.elev.push(f.elev ?? null);
  if (buf.t.length > MAXBUF) for (const k in buf) buf[k].splice(0, buf[k].length - MAXBUF);
  if (f.detected && !(last && last.detected) && detT === null) detT = f.t_ms;
  if (!f.detected) { detT = null; ovhT = null; }
  if (f.detected && f.overhead && ovhT === null) ovhT = f.t_ms;
  if (!f.overhead) ovhT = null;
  last = f; lastFrameWall = performance.now();
}
function clearBuffers() { for (const k in buf) buf[k].length = 0; detT = ovhT = null; }

function onEvent(ev, fresh) {
  events.push(ev); if (events.length > 30) events.shift();
  if (ev.kind === "detect_on") { detT = ev.t * 1000; if (ev.latency_s != null) lastLatency = ev.latency_s; }
  if (ev.kind === "overhead_on") ovhT = ev.t * 1000;
  renderLog(fresh);
}

// ---------------------------------------------------------------- helpers
function dirName(th) {
  if (th == null) return "";
  const names = ["front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left"];
  const i = Math.round((((th % 360) + 360) % 360) / 45) % 8;
  return names[i];
}
const fmtDeg = (th) => (th == null ? "—" : `${th > 0 ? "+" : th < 0 ? "−" : ""}${Math.abs(Math.round(th))}°`);
const fmtElev = (e) => (e == null ? "—" : `${Math.round(Math.max(-90, Math.min(90, e)))}°`);
function elevWord(e) {
  if (e == null) return " ";
  return e >= CFG.overhead_elev_on ? "steep / above" : e >= 30 ? "high" : e >= 10 ? "low" : "on the horizon";
}
const battPct = (v) => (v == null ? null : Math.round(100 * Math.max(0, Math.min(1, (v - CFG.batt_empty_v) / (CFG.batt_full_v - CFG.batt_empty_v)))));
function fmtDur(s) {
  if (s == null || !isFinite(s)) return "—";
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60), r = Math.floor(s % 60);
  if (m < 60) return `${m}:${String(r).padStart(2, "0")}`;
  return `${Math.floor(m / 60)}h${String(m % 60).padStart(2, "0")}`;
}
const clamp01 = (x) => Math.max(0, Math.min(1, x || 0));
const hhmmss = (wall) => new Date(wall * 1000).toLocaleTimeString([], { hour12: false });

// ---------------------------------------------------------------- bearing status
// Each frame's bstat (0 no bearing / unreliable, 1 ambiguous mirror, 2 three mics: azimuth only, or elevation assumed
// above, 3 full 3D) and mused (the mics in the frame's fit, B=1 L=2 R=4 T=8), from the T lines or the pipeline;
// null from older firmware, logs and pipelines, when the status is inferred from theta / elev. Bit 1 is the DSP's
// channel 0: the back mic, or the front one when the headset's '# cal' line says F= (C<w0>,<l>,<r>,<t>,F).
const MIC_BIT = { B: 1, L: 2, R: 4, T: 8 };
const NUM_WORD = ["no", "one", "two", "three", "four"];

// The letter of channel 0: "F" when the headset reports its front mic there, else "B" (simulated / replayed input
// always has the back mic).
function ch0Letter() {
  const m = source === "serial" && status && status.cal_device && status.cal_device.mics;
  return m && "F" in m && !("B" in m) ? "F" : "B";
}
const micLabel = (c) => (c === "B" ? ch0Letter() : c);

// Mics the source has: the headset's calibration (its '# cal' line), else the simulated input's channel count.
function micsConnected() {
  const m = status && status.cal_device && status.cal_device.mics;
  if (source === "serial") return m ? Object.keys(MIC_BIT).map(micLabel).filter((c) => m[c] != null && m[c] >= 0).length : null;
  return status && status.channels != null ? Number(status.channels) : null;
}

function noBearingWhy() {
  const n = micsConnected();
  if (n === 1 && source !== "serial") return "mono input: detection only";
  if (n != null && n < 3) return `${NUM_WORD[n] || n} mic${n === 1 ? "" : "s"} connected: detection only (a bearing needs three)`;
  return "no reliable direction this frame (the delays between the mics disagree)";
}

function bearingStatus(f) {
  if (!f) return null;
  const az = f.theta != null, el = f.elev != null;
  const b = f.bstat != null ? Number(f.bstat) : az ? (el ? 3 : 2) : 0;   // older sources: what the frame carries
  const n = f.mused == null ? null : Object.values(MIC_BIT).filter((x) => f.mused & x).length;
  const from = n == null ? "" : ` from ${NUM_WORD[n] || n} mic${n === 1 ? "" : "s"}`;
  if (b >= 3 && az) return { code: 3, label: "3D", title: `full 3D fix: bearing and elevation${from}` };
  if (b === 2 && az && el) return { code: 2, label: "assumed above", title: `bearing and elevation${from}, assuming the drone is above the head (its mirror image below is ruled out)` };
  if (b >= 2 && az) return { code: 2, label: "bearing only", title: `bearing${from}, no elevation (the top mic is not in the fit)` };
  if (b === 1) return { code: 1, label: "ambiguous", title: "two mirror-image directions fit equally well (front / back or left / right): no bearing this frame" };
  if (az) return { code: 0, label: "held", title: "no reliable fix this frame: the last bearing is held" };
  return { code: 0, label: "no bearing", title: noBearingWhy() };
}
const bstChip = (bs, extra) => (bs ? `<span class="bst s${bs.code}${extra ? " " + extra : ""}" title="${bs.title.replace(/"/g, "&quot;")}">${bs.label}</span>` : "");
function setHtml(el, html) { if (el && el._html !== html) { el.innerHTML = html; el._html = html; } }

// ---------------------------------------------------------------- DOM (throttled ~12 Hz)
function renderText() {
  const f = last;
  const now = performance.now();
  const stale = !wsOpen || !f || now - lastFrameWall > 2000;
  document.body.classList.toggle("stale", stale);
  // header
  const st = status;
  $("fps").textContent = st.frames_per_s ?? 0;
  $("uptime").textContent = fmtDur(st.uptime_s ?? 0).replace(" s", "s");
  const dot = $("connDot");
  if (!wsOpen) { dot.className = "dot off"; $("connText").textContent = "dashboard server offline"; }
  else if (!st.connected) { dot.className = "dot wait"; $("connText").textContent = source === "serial" ? "waiting for device" : "loading"; }
  else if (stale) { dot.className = "dot wait"; $("connText").textContent = "no frames"; }
  else { dot.className = "dot ok"; $("connText").textContent = source === "serial" ? (st.transport === "tcp" ? "device connected (Wi-Fi)" : "device connected") : "streaming"; }
  if (benchState.recBusy) { dot.className = "dot wait"; $("connText").textContent = "recording raw audio · detector paused"; }

  if (!f) return;
  const det = !!f.detected, ovh = det && !!f.overhead;
  const state = ovh ? "overhead" : det ? "detected" : "idle";
  document.body.dataset.state = state;
  const hasBearing = f.theta != null;
  const hasElev = f.elev != null;
  const bs = bearingStatus(f);
  const nMics = micsConnected() ?? (CFG.mic_names || []).length;

  // hero
  if (ovh) {
    $("stateLabel").textContent = "DRONE OVERHEAD";
    $("stateSub").textContent = `Elevation ${fmtElev(f.elev)} · source above the head · motors spinning round`;
  } else if (det) {
    $("stateLabel").textContent = "DRONE DETECTED";
    $("stateSub").textContent = hasBearing ? `Bearing ${fmtDeg(f.theta)} · ${dirName(f.theta)}${hasElev ? ` · elev ${fmtElev(f.elev)}` : ""} · f0 ${Math.round(f.f0)} Hz`
                                           : `f0 ${Math.round(f.f0)} Hz · ${bs.code === 1 ? "bearing ambiguous (mirror image)" : nMics < 3 ? `no bearing (${noBearingWhy()})` : "no bearing this frame"}`;
  } else {
    $("stateLabel").textContent = "IDLE";
    $("stateSub").textContent = nMics === 1 && source !== "serial" ? "Listening (mono input: detection only)"
      : nMics === 0 ? "No microphone connected (see Calibrate)"
      : nMics < 3 ? `Listening on ${NUM_WORD[nMics]} microphones (detection only, no bearing)`
      : `Listening on ${NUM_WORD[nMics] || nMics} microphones`;
  }
  $("hBearing").textContent = det && hasBearing ? (ovh ? "ABOVE" : fmtDeg(f.theta)) : "—";
  setHtml($("hBearingDir"), det ? `${hasBearing ? (ovh ? "overhead" : dirName(f.theta)) : ""} ${bstChip(bs, "sm")}`.trim() : "&nbsp;");
  $("hElev").textContent = det && hasElev ? fmtElev(f.elev) : "—";
  $("hElevS").textContent = det && hasElev ? (ovh ? "overhead" : bs.code === 2 ? "assumed above" : elevWord(f.elev)) : " ";
  if (det && detT != null) {
    $("hSinceK").textContent = "Detected for";
    $("hSince").textContent = fmtDur((f.t_ms - detT) / 1000);
    $("hSinceS").textContent = "since alarm";
  } else {
    const off = [...events].reverse().find((e) => e.kind === "detect_off");
    $("hSinceK").textContent = "Last drone";
    $("hSince").textContent = off ? fmtDur(Date.now() / 1000 - off.wall) : "none yet";
    $("hSinceS").textContent = off ? `ago · lasted ${fmtDur(off.duration_s)}` : " ";
  }
  $("hLatency").textContent = lastLatency == null ? "—" : `${lastLatency.toFixed(1)} s`;

  // radar readout: the bearing status of this frame (dimmed while idle) and the mics in its fit
  $("rDeg").textContent = det && hasBearing ? (ovh ? "ABOVE" : fmtDeg(f.theta)) : "—";
  $("rDir").textContent = det && hasBearing && !ovh ? dirName(f.theta) : "";
  setHtml($("rBst"), bstChip(bs, det ? "" : "idle"));
  const known = f.mused != null;
  $("rMics").classList.toggle("unknown", !known);
  $("rMics").title = known ? "mics in this frame's direction fit (lit)" : "this source does not report which mics are in the fit";
  for (const [m, bit] of Object.entries(MIC_BIT)) {
    const el = $("rMic" + m), lab = micLabel(m);
    el.classList.toggle("on", known && (f.mused & bit) !== 0);
    if (el.textContent !== lab) { el.textContent = lab; el.title = m === "B" ? (lab === "F" ? "front mic (channel 0)" : "back mic (channel 0)") : ""; }
  }
  $("rConf").textContent = f.conf == null || !det ? "—" : f.conf.toFixed(2);
  $("rElev").textContent = f.elev == null || !det ? "—" : fmtElev(f.elev);

  // chart values
  $("vScore").textContent = f.score == null ? "—" : `${f.score.toFixed(1)} dB`;
  $("vHits").textContent = `${f.hits} / ${CFG.persist_frames}`;
  $("vMl").textContent = f.ml == null ? "—" : f.ml.toFixed(2);
  $("vF0").textContent = f.f0 == null ? "—" : `${Math.round(f.f0)} Hz`;
  $("vF0").classList.toggle("dim", f.hits < CFG.persist_off);      // f0 of noise is meaningless: grey it out

  // haptics: one pad per motor, lit by the weight the board is applying (overhead: the rotating triplet)
  const hap = f.hap || [];
  const pulse = Math.max(0.35, 1.3 - 0.06 * (f.strength || 0));     // faster pulse with harmonic strength
  document.documentElement.style.setProperty("--pulse", `${pulse.toFixed(2)}s`);
  let nOn = 0;
  const lit = [];
  CFG.motor_names.forEach((m, i) => {
    const w = clamp01(hap[i]);
    const g = $("pad" + m); if (!g) return;
    const r = g.querySelector("rect");
    r.style.fill = w > 0.01 ? `rgba(255, 90, 54, ${0.15 + 0.85 * w})` : "";
    g.classList.toggle("on", w > 0.01);
    g.classList.toggle("pulse", det && !ovh && w > 0.01);           // the overhead ring moves instead of pulsing
    g.querySelector(".pct").textContent = w > 0.01 ? `${Math.round(w * 100)}` : "";
    if (w > 0.01) { nOn++; lit.push(`${m} ${Math.round(w * 100)}%`); }
  });
  $("hOn").textContent = nOn;
  $("hMax").textContent = CFG.motor_max_on ?? 3;
  $("hTop").textContent = lit.length ? lit.join(" · ") : " ";
  const manual = motAny();
  $("hapAbove").classList.toggle("on", ovh && !manual);
  $("hapNote").textContent = manual ? "manual: Bench motor toggles" : ovh ? "above: 3 at a time, spinning round" : det && hasBearing ? "panning between neighbours" : det ? (nOn ? "no bearing: the ring pulses" : "no bearing: motors idle") : "motors off";

  renderDevice();
}

function renderDevice() {
  const st = status, f = last;
  const rows = [];
  const serial = source === "serial";
  $("devTitle").textContent = serial ? "Device" : "Source";
  const wifi = st.transport === "tcp";
  $("devMode").textContent = serial ? (wifi ? "Wi-Fi · TCP" : st.baud ? `USB · ${st.baud} baud` : "USB") : source === "replay" ? "log replay" : "Python pipeline";
  if (serial) {
    rows.push([wifi ? "Host" : "Port", st.port || "—"]);
    if (st.wifi_mode) rows.push(["Wi-Fi", `${st.wifi_mode}${st.wifi_ssid ? " · " + st.wifi_ssid : ""}${st.wifi_ip ? " · " + st.wifi_ip : ""}`]);
    rows.push(["Firmware", st.firmware || "—"]);
    const cu = st.chunk_us, ok = cu && Math.abs(cu - CFG.chunk_us_expect) < 0.02 * CFG.chunk_us_expect;
    if (cu) rows.push(["Chunk period", `${cu} µs`, ok ? "good" : "bad"]);
    else {                       // the '# pdm chunk_us' line is printed once at boot; else estimate from t_ms
      const n = buf.t.length, k = Math.min(n - 1, 94);
      const est = k > 10 ? Math.round(1000 * (buf.t[n - 1] - buf.t[n - 1 - k]) / k) : null;
      const okE = est && Math.abs(est - CFG.chunk_us_expect) < 0.05 * CFG.chunk_us_expect;
      rows.push(["Frame period", est ? `≈${est} µs (t_ms)` : "—", est ? (okE ? "good" : "bad") : ""]);
    }
    rows.push(["Dropped chunks", st.dropped ?? "—", st.dropped ? "bad" : ""]);
    if (wifi || st.wifi_dropped) rows.push(["Wi-Fi dropped", st.wifi_dropped ?? "—", st.wifi_dropped ? "bad" : ""]);
    rows.push(["Free heap", st.free_heap != null ? `${(st.free_heap / 1024).toFixed(0)} KB` : "—"]);
    const bv = f && f.batt_v != null ? f.batt_v : st.batt_v;
    if (bv != null) { const pc = battPct(bv); rows.push(["Battery", `${bv.toFixed(2)} V · ${pc}%`, pc < 15 ? "bad" : pc > 30 ? "good" : ""]); }
    rows.push(["Frame", f ? f.frame : "—"]);
    rows.push(["Device uptime", st.device_uptime_s != null ? fmtDur(st.device_uptime_s) : "—"]);
    if (st.bad_lines) rows.push(["Malformed lines", st.bad_lines, "bad"]);
    if (st.error && !st.connected) rows.push(["Last error", st.error, "bad"]);
  } else {
    rows.push(["File", st.port || "—"]);
    rows.push(["Pipeline", st.firmware || "—"]);
    rows.push(["Frame", f ? f.frame : "—"]);
    rows.push(["Stream time", f ? fmtDur(f.t_ms / 1000) : "—"]);
    rows.push(["Loops", st.loops ?? 0]);
    if (f && f.batt_v != null) rows.push(["Battery", `${f.batt_v.toFixed(2)} V · ${battPct(f.batt_v)}%`]);
    if (st.error) rows.push(["Error", st.error, "bad"]);
  }
  const html = rows.map(([k, v, c]) => `<dt>${k}</dt><dd class="${c || ""}" title="${String(v).replace(/"/g, "&quot;")}">${escapeHtml(String(v))}</dd>`).join("");
  const dev = $("dev");
  if (dev._html !== html) { dev.innerHTML = html; dev._html = html; }
}
const escapeHtml = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

function renderLog(fresh) {
  const ul = $("log");
  const items = events.slice(-10).reverse();
  $("evCount").textContent = events.length ? `${events.length} this session` : "";
  if (!items.length) { ul.innerHTML = '<li class="empty">No events yet</li>'; return; }
  ul.innerHTML = items.map((e, i) => {
    let cls = "", txt = "";
    const dir = e.theta != null ? `${fmtDeg(e.theta)} ${dirName(e.theta)}` : "";
    const el = e.elev != null ? ` · elev ${fmtElev(e.elev)}` : "";
    if (e.kind === "detect_on") { cls = "on"; txt = `<b>Drone detected</b> <small>${dir}${el}${e.latency_s != null ? ` · after ${e.latency_s.toFixed(1)} s of evidence` : ""}</small>`; }
    else if (e.kind === "detect_off") { cls = "off"; txt = `Clear <small>lasted ${fmtDur(e.duration_s)}${dir ? ` · last ${dir}` : ""}</small>`; }
    else if (e.kind === "overhead_on") { cls = "ovh"; txt = `<b>Overhead</b> <small>${e.elev != null ? `elev ${fmtElev(e.elev)} · ` : ""}motor ring spinning</small>`; }
    else if (e.kind === "overhead_off") { cls = "off"; txt = `Overhead cleared <small>${fmtDur(e.duration_s)}</small>`; }
    return `<li class="${cls}${fresh && i === 0 ? " fresh" : ""}"><time>${hhmmss(e.wall)}</time><i></i><span class="txt">${txt}</span></li>`;
  }).join("");
}

// ---------------------------------------------------------------- canvas plumbing
function fitCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const r = cv.getBoundingClientRect();
  const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: r.width, h: r.height };
}

// ---------------------------------------------------------------- radar
function drawRadar(nowMs) {
  const { ctx, w, h } = fitCanvas($("radar"));
  ctx.clearRect(0, 0, w, h);
  const cx = w / 2, cy = h / 2;
  const R = Math.max(20, Math.min(h / 2 - 34, w / 2 - 66));
  const f = last;
  const det = f && f.detected, ovh = det && f.overhead;
  const bs = bearingStatus(f);
  const P = (thDeg, r) => { const a = thDeg * Math.PI / 180; return [cx + r * Math.sin(a), cy - r * Math.cos(a)]; };
  const canvasAng = (thDeg) => (thDeg - 90) * Math.PI / 180;

  // rings, spokes, ticks
  ctx.lineWidth = 1;
  for (const k of [1, 0.66, 0.33]) {
    ctx.strokeStyle = k === 1 ? COL.line2 : COL.line;
    ctx.beginPath(); ctx.arc(cx, cy, R * k, 0, 2 * Math.PI); ctx.stroke();
  }
  for (let a = 0; a < 360; a += 10) {
    const major = a % 90 === 0, mid = a % 30 === 0;
    const [x1, y1] = P(a, R), [x2, y2] = P(a, R - (major ? 14 : mid ? 9 : 5));
    ctx.strokeStyle = major ? COL.muted : COL.line2; ctx.lineWidth = major ? 2 : 1;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  }
  ctx.strokeStyle = COL.line; ctx.lineWidth = 1; ctx.setLineDash([3, 6]);
  for (const a of [0, 90]) { const [x1, y1] = P(a, R), [x2, y2] = P(a + 180, R); ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); }
  ctx.setLineDash([]);
  // cardinal labels outside the ring
  ctx.fillStyle = COL.muted; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  const fs = Math.max(11, Math.min(15, R / 14));
  ctx.font = `700 ${fs}px ${getComputedStyle(document.body).fontFamily}`;
  const lab = [[0, "FRONT", "0\u00b0"], [180, "BACK", "180\u00b0"]];
  for (const [a, s1, s2] of lab) { const [x, y] = P(a, R + 20); ctx.textAlign = "center"; ctx.fillText(`${s1} ${s2}`, x, y); }
  ctx.textAlign = "right"; ctx.fillText("LEFT", cx - R - 12, cy - fs * 0.7); ctx.fillText("\u221290\u00b0", cx - R - 12, cy + fs * 0.7);
  ctx.textAlign = "left"; ctx.fillText("RIGHT", cx + R + 12, cy - fs * 0.7); ctx.fillText("+90\u00b0", cx + R + 12, cy + fs * 0.7);

  // idle sweep: a slow, faint rotating line so the screen reads as alive
  if (!det) {
    const a = (nowMs / 4000 * 360) % 360;
    const g = ctx.createConicGradient ? ctx.createConicGradient(canvasAng(a) - 0.6, cx, cy) : null;
    if (g) {
      g.addColorStop(0, "rgba(124,196,255,0)"); g.addColorStop(0.095, "rgba(124,196,255,0.10)"); g.addColorStop(0.0955, "rgba(124,196,255,0)");
      ctx.fillStyle = g; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.fill();
    }
  }

  // expanding rings while detected
  if (det) {
    for (let k = 0; k < 3; k++) {
      const ph = ((nowMs / 1500) + k / 3) % 1;
      ctx.strokeStyle = `rgba(255,90,54,${0.45 * (1 - ph)})`; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(cx, cy, R * (0.3 + 0.7 * ph), 0, 2 * Math.PI); ctx.stroke();
    }
  }

  // elevation as the top-down projection: a source on the horizon sits on the ring, one straight above the head at
  // the centre (radius ~ cos(elev)), kept outside the head drawing
  const elevR = (e) => R * (0.3 + 0.7 * Math.cos(Math.max(0, Math.min(90, e ?? 0)) * Math.PI / 180));

  // trail: last 2 s of theta while detected
  if (det && f.theta != null) {
    const tEnd = buf.t[buf.t.length - 1];
    for (let i = buf.t.length - 1; i >= 0 && tEnd - buf.t[i] < TRAIL_MS; i--) {
      if (!buf.det[i] || buf.theta[i] == null) continue;
      const age = (tEnd - buf.t[i]) / TRAIL_MS;
      const [x, y] = P(buf.theta[i], elevR(buf.elev[i]) * 0.94);
      ctx.fillStyle = `rgba(255,90,54,${0.5 * (1 - age)})`;
      ctx.beginPath(); ctx.arc(x, y, 3.5, 0, 2 * Math.PI); ctx.fill();
    }
  }

  // bearing wedge + arrow
  if (det && f.theta != null && !ovh) {
    const c = clamp01(f.conf);
    const half = (30 - 22 * c) * Math.PI / 180;
    const a0 = canvasAng(f.theta);
    const op = 0.35 + 0.65 * c;
    const Re = elevR(f.elev);                                // wedge length = horizontal distance of the source
    const g = ctx.createRadialGradient(cx, cy, R * 0.15, cx, cy, Re);
    g.addColorStop(0, "rgba(255,90,54,0)"); g.addColorStop(1, `rgba(255,90,54,${0.55 * op})`);
    ctx.fillStyle = g; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.arc(cx, cy, Re, a0 - half, a0 + half); ctx.closePath(); ctx.fill();
    if (Re < R - 4) {                                        // the horizon position, faint, for reference
      ctx.strokeStyle = "rgba(255,120,90,0.35)"; ctx.lineWidth = 1.5; ctx.setLineDash([2, 5]);
      ctx.beginPath(); ctx.arc(cx, cy, R, a0 - half, a0 + half); ctx.stroke(); ctx.setLineDash([]);
    }
    const [x0, y0] = P(f.theta, R * 0.32), [x1, y1] = P(f.theta, Re - 2);
    ctx.strokeStyle = `rgba(255,120,90,${op})`; ctx.lineWidth = 4; ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.fillStyle = COL.alarm; ctx.shadowColor = COL.alarm; ctx.shadowBlur = 18;
    ctx.beginPath(); ctx.arc(x1, y1, 9, 0, 2 * Math.PI); ctx.fill(); ctx.shadowBlur = 0;
  }

  // head (top view): nose up, mics marked
  const s = R * 1.5;                                      // metres -> px for the mic layout
  const hx = 0.115 * s, hy = 0.14 * s;
  ctx.fillStyle = "#18222e"; ctx.strokeStyle = COL.line2; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(cx - hx * 0.25, cy - hy * 0.96); ctx.lineTo(cx, cy - hy * 1.28); ctx.lineTo(cx + hx * 0.25, cy - hy * 0.96); ctx.closePath(); ctx.fill(); ctx.stroke();
  ctx.beginPath(); ctx.ellipse(cx, cy, hx, hy, 0, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  for (const sx of [-1, 1]) { ctx.beginPath(); ctx.ellipse(cx + sx * hx * 1.04, cy, hx * 0.14, hy * 0.24, 0, 0, 2 * Math.PI); ctx.fill(); ctx.stroke(); }
  ctx.font = `700 ${Math.max(10, fs - 2)}px ${getComputedStyle(document.body).fontFamily}`;
  const front = ch0Letter() === "F";                       // channel 0 at the front: drawn there, labelled F
  CFG.mic_pos.forEach((p, i) => {
    const [mx, my, mz] = i === 0 && front ? (CFG.front_pos || [p[0], -p[1], p[2]]) : p;
    const x = cx + mx * s, y = cy - my * s;
    const bit = MIC_BIT[CFG.mic_names[i]];                 // lit when in this frame's fit (not reported: all lit)
    const lit = !f || f.mused == null || bit == null || (f.mused & bit) !== 0;
    ctx.fillStyle = lit ? COL.cool : COL.panel; ctx.strokeStyle = lit ? COL.cool : COL.faint; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(x, y, 4.5, 0, 2 * Math.PI); ctx.fill(); if (!lit) ctx.stroke();
    if (mz > 0.01) {                                         // a mic above the band plane (T, on the crown): ring it
      ctx.beginPath(); ctx.arc(x, y, 8.5, 0, 2 * Math.PI); ctx.stroke();
    }
    ctx.fillStyle = COL.muted; ctx.textAlign = "center";
    const dx = mx < 0 ? -14 : mx > 0 ? 14 : 0, dy = my < 0 ? 14 : my > 0 ? -14 : (mz > 0.01 ? -16 : 0);
    ctx.fillText(micLabel(CFG.mic_names[i]), x + dx * 1.3, y + dy + (dx ? 0 : 2));
  });
  drawElevGauge(ctx, w, h, R, f, det, ovh, fs);

  // overhead: filled pulsing disc + ring around everything
  if (ovh) {
    const ph = 0.5 + 0.5 * Math.sin(nowMs / 250);
    ctx.fillStyle = `rgba(255,90,54,${0.55 + 0.25 * ph})`;
    ctx.shadowColor = COL.alarm; ctx.shadowBlur = 30;
    ctx.beginPath(); ctx.arc(cx, cy, R * 0.3, 0, 2 * Math.PI); ctx.fill(); ctx.shadowBlur = 0;
    ctx.strokeStyle = `rgba(255,90,54,${0.35 + 0.4 * ph})`; ctx.lineWidth = 6;
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.stroke();
    ctx.fillStyle = "#fff"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.font = `800 ${Math.max(16, R / 9)}px ${getComputedStyle(document.body).fontFamily}`;
    ctx.fillText("ABOVE", cx, cy);
  }

  if (f && f.theta == null && !ovh) {
    const n = micsConnected();                               // a standing reason, else (detected) this frame's
    const txt = n === 1 && source !== "serial" ? "mono input: no bearing" : n != null && n < 3 ? "too few mics: no bearing"
      : !det ? "" : bs.code === 1 ? "ambiguous: mirror-image directions" : "no bearing this frame";
    if (txt) {
      ctx.fillStyle = COL.faint; ctx.textAlign = "center"; ctx.font = `600 ${fs}px ${getComputedStyle(document.body).fontFamily}`;
      ctx.fillText(txt, cx, cy + R * 0.55);
    }
  }
}

// Elevation gauge, bottom-left of the radar: a side view (horizon to the right, straight up at the top), the needle
// at the current elevation, the overhead threshold as a tick, and the number.
function drawElevGauge(ctx, w, h, R, f, det, ovh, fs) {
  const gr = Math.max(28, Math.min(64, R * 0.3));
  const ox = 12, oy = h - 12;
  const fam = getComputedStyle(document.body).fontFamily;
  const has = det && f && f.elev != null;
  const e = has ? Math.max(0, Math.min(90, f.elev)) : 0;
  const A = (deg) => -deg * Math.PI / 180;                 // canvas angle: 0 = right (horizon), -90 deg = up
  ctx.lineCap = "butt";
  ctx.strokeStyle = COL.line2; ctx.lineWidth = 6;
  ctx.beginPath(); ctx.arc(ox, oy, gr, A(90), A(0)); ctx.stroke();
  if (has) {
    ctx.strokeStyle = ovh ? COL.alarm : "rgba(255,90,54,0.75)";
    ctx.beginPath(); ctx.arc(ox, oy, gr, A(e), A(0)); ctx.stroke();
  }
  const on = CFG.overhead_elev_on ?? 60;                   // overhead threshold tick
  ctx.strokeStyle = COL.muted; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(ox + (gr - 7) * Math.cos(A(on)), oy + (gr - 7) * Math.sin(A(on)));
  ctx.lineTo(ox + (gr + 7) * Math.cos(A(on)), oy + (gr + 7) * Math.sin(A(on))); ctx.stroke();
  ctx.strokeStyle = COL.line; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox + gr + 6, oy); ctx.moveTo(ox, oy); ctx.lineTo(ox, oy - gr - 6); ctx.stroke();
  if (has) {
    ctx.strokeStyle = COL.text; ctx.lineWidth = 2.5; ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox + (gr - 2) * Math.cos(A(e)), oy + (gr - 2) * Math.sin(A(e))); ctx.stroke();
  }
  const lx = ox + gr + 12;                                   // labels to the right of the gauge, on the horizon line
  ctx.fillStyle = COL.faint; ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
  ctx.font = `700 ${Math.max(10, fs - 3)}px ${fam}`;
  ctx.fillText("ELEV", lx, oy - fs - 6);
  ctx.fillStyle = has ? (ovh ? COL.alarm : COL.text) : COL.faint;
  ctx.font = `750 ${Math.max(14, fs + 3)}px ${fam}`;
  ctx.fillText(has ? `${Math.round(e)}°` : "—", lx, oy - 2);
}

// ---------------------------------------------------------------- charts
const CHARTS = [
  { id: "cScore", key: "score", min: 0, max: () => Math.max(16, CFG.harm_thresh_db * 2.5), thr: () => CFG.harm_thresh_db, thrLabel: (v) => `${v} dB`, ticks: [0, 5, 10, 15] },
  { id: "cHits", key: "hits", min: 0, max: () => CFG.persist_frames, thr: () => CFG.persist_on, thrLabel: (v) => `${v} to alarm`, ticks: [0, 47, 94] },
  { id: "cMl", key: "ml", min: 0, max: () => 1, thr: () => CFG.ml_thresh, thrLabel: (v) => `gate ${v}`, ticks: [0, 0.5, 1] },
  { id: "cF0", key: "f0", min: () => CFG.f0_min - 20, max: () => CFG.f0_max + 20, thr: null, ticks: [150, 250, 350] },
];
const val = (x) => (typeof x === "function" ? x() : x);

function drawChart(ch, tNow) {
  const { ctx, w, h } = fitCanvas($(ch.id));
  ctx.clearRect(0, 0, w, h);
  const padL = 30, padR = 6, padT = 4, padB = 4;
  const pw = w - padL - padR, ph = h - padT - padB;
  if (pw < 10 || ph < 10) return;
  const lo = val(ch.min), hi = val(ch.max);
  const X = (t) => padL + pw * (1 - (tNow - t) / WINDOW_MS);
  const Y = (v) => padT + ph * (1 - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo));
  const n = buf.t.length;
  let i0 = 0; while (i0 < n && buf.t[i0] < tNow - WINDOW_MS) i0++;

  // detected shading
  ctx.fillStyle = "rgba(255,90,54,0.13)";
  for (let i = i0; i < n; i++) {
    if (!buf.det[i]) continue;
    let j = i; while (j + 1 < n && buf.det[j + 1]) j++;
    const xa = Math.max(padL, X(buf.t[i])), xb = X(buf.t[j]) + 1;
    ctx.fillRect(xa, padT, xb - xa, ph);
    i = j;
  }
  // grid + ticks
  ctx.font = `11px ${getComputedStyle(document.body).fontFamily}`; ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (const tv of ch.ticks) {
    if (tv < lo || tv > hi) continue;
    const y = Y(tv);
    ctx.strokeStyle = COL.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
    ctx.fillStyle = COL.faint; ctx.fillText(String(tv), padL - 6, y);
  }
  // threshold
  if (ch.thr) {
    const tv = ch.thr(), y = Y(tv);
    ctx.strokeStyle = COL.text; ctx.globalAlpha = 0.55; ctx.setLineDash([6, 5]); ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
  }
  // line + area; segments above threshold in the alarm colour
  const thr = ch.thr ? ch.thr() : null;
  if (n - i0 >= 2) {
    const grad = ctx.createLinearGradient(0, padT, 0, padT + ph);
    grad.addColorStop(0, "rgba(124,196,255,0.22)"); grad.addColorStop(1, "rgba(124,196,255,0)");
    ctx.beginPath(); let started = false, xFirst = 0, xLast = 0;
    for (let i = i0; i < n; i++) {
      const v = buf[ch.key][i]; if (v == null) continue;
      const x = X(buf.t[i]), y = Y(v);
      if (!started) { ctx.moveTo(x, y); xFirst = x; started = true; } else ctx.lineTo(x, y);
      xLast = x;
    }
    if (started) {
      ctx.lineJoin = "round"; ctx.lineWidth = 2; ctx.strokeStyle = COL.cool; ctx.stroke();
      ctx.lineTo(xLast, padT + ph); ctx.lineTo(xFirst, padT + ph); ctx.closePath(); ctx.fillStyle = grad; ctx.fill();
    }
    if (thr != null) {
      ctx.save(); ctx.beginPath(); ctx.rect(padL, padT, pw, Y(thr) - padT); ctx.clip();
      ctx.beginPath(); started = false;
      for (let i = i0; i < n; i++) {
        const v = buf[ch.key][i]; if (v == null) continue;
        const x = X(buf.t[i]), y = Y(v);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      }
      ctx.lineWidth = 2.4; ctx.strokeStyle = COL.alarm; ctx.stroke(); ctx.restore();
    }
    // head dot
    const v = buf[ch.key][n - 1];
    if (v != null) {
      ctx.fillStyle = thr != null && v > thr ? COL.alarm : COL.cool;
      ctx.beginPath(); ctx.arc(X(buf.t[n - 1]), Y(v), 3.5, 0, 2 * Math.PI); ctx.fill();
    }
  }
}

// ---------------------------------------------------------------- loop (text on a timer, canvases per animation frame)
setInterval(() => { try { renderText(); renderBench(); } catch (e) { console.error(e); } }, 100);
function frame(nowMs) {
  const tNow = buf.t.length ? buf.t[buf.t.length - 1] : 0;
  try {
    drawRadar(nowMs);
    for (const ch of CHARTS) drawChart(ch, tNow);
  } catch (e) { console.error(e); }
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------- websocket
function setSource(src) {
  source = src;
  const b = $("srcBadge");
  const map = { serial: ["LIVE", "live"], replay: ["REPLAY", "replay"], simulate: ["SIM", "sim"], demo: ["SIM · DEMO", "sim"] };
  const [txt, cls] = map[src] || [src.toUpperCase(), ""];
  b.textContent = txt; b.className = "badge " + cls;
  benchEnable(src === "serial");
  if (window.onCalibSource) window.onCalibSource(src, !!(status && status.connected));   // calibration.js
  if (window.onBatterySource) window.onBatterySource(src, !!(status && status.connected));   // battery.js
  ledRender();
}

// ---------------------------------------------------------------- stealth mode (the headset's detect LED off)
// The board's state comes from status.led_stealth and the "led" bench messages ('# led stealth=on|off'); the button
// POSTs /api/led, which waits for the board's reply.
const ledState = { stealth: null, busy: false, hold: 0 };

function ledRender() {
  const b = $("ledBtn"), ok = source === "serial" && !!(status && status.connected), st = ledState.stealth;
  b.disabled = !ok || ledState.busy;
  b.classList.toggle("on", ok && st === true);
  b.textContent = ok && st === true ? "Stealth on" : "Stealth";
  if (b.classList.contains("err")) return;                 // keep the error in the tooltip for now
  b.title = !ok ? "stealth mode (the headset's LED off) needs the device: start the dashboard with --port or --tcp"
    : st === true ? "stealth on: the headset's LED stays off. Click to bring the LED patterns back."
    : st === false ? "stealth off: the headset's LED blinks its patterns. Click to keep it off."
    : "the board has not reported its LED mode yet (older firmware?)";
}

function ledSync() {                                         // from a status message
  if (!ledState.busy && Date.now() > ledState.hold && status && typeof status.led_stealth === "boolean")
    ledState.stealth = status.led_stealth;
  ledRender();
}

async function ledToggle() {
  const b = $("ledBtn");
  ledState.busy = true; b.classList.remove("err"); ledRender();
  try {
    const r = await fetch("/api/led", { method: "POST", headers: { "Content-Type": "application/json" },
                                        body: JSON.stringify({ stealth: ledState.stealth !== true }) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok || !j.ok) throw new Error(j.error || `HTTP ${r.status}`);
    ledState.stealth = j.stealth;
    ledState.hold = Date.now() + 1500;                       // a status message already on its way is older
  } catch (e) {
    b.classList.add("err"); b.title = `stealth mode: ${e.message}`;
    setTimeout(() => { b.classList.remove("err"); ledRender(); }, 4000);
  } finally {
    ledState.busy = false; ledRender();
  }
}

// ---------------------------------------------------------------- bench panel (tap test, motor test, board status)
// the board's DMA words: the v3 board reads each data line's SEL 3V3 phase first (right-first slot order, verified)
const SLOT_NAMES = ["DIN0 · SEL 3V3 · MK2", "DIN0 · SEL GND · MK1", "DIN1 · SEL 3V3 · MK4", "DIN1 · SEL GND · MK3",
                    "DIN2 · SEL 3V3 · no header", "DIN2 · SEL GND · MK5"];
const DB_FLOOR = -80;             // bar scale: -80 dBFS .. 0
const benchState = { on: false, tap: false, lastLevels: 0, devHtml: "", rec: null, recBusy: false, play: null,
                     mot: CFG.motor_names.map(() => false), motLast: "", motAsk: 0, motAck: 0 };

function sendCmd(obj) {
  if (wsRef && wsRef.readyState === 1) wsRef.send(JSON.stringify(obj));
}

function benchBuild() {
  const host = $("slots");
  host.innerHTML = SLOT_NAMES.map((n, w) =>
    `<div class="slot" id="slot${w}"><div class="slot-k"><span><b>w${w}</b>${n}</span><span class="slot-ch" id="slotCh${w}"></span></div>` +
    `<div class="slot-bar"><i class="rms" id="slotRms${w}"></i><i class="peak" id="slotPk${w}"></i></div>` +
    `<div class="slot-v"><span id="slotPkV${w}">—</span><span class="dc" id="slotDc${w}">dc —</span></div></div>`).join("");
  $("benchBtn").addEventListener("click", () => benchShow(!benchState.on));
  $("tapBtn").addEventListener("click", () => tapTest(!benchState.tap));
  $("motorBtn").addEventListener("click", () => {
    motRelease(motAny());                                   // the sequence alone, not on top of manual motors
    sendCmd({ type: "cmd", cmd: "M" });
    $("motorNote").textContent = `Sent. Watch the pads under Haptics: ${CFG.motor_names.join(", then ")}.`;
  });
  $("motBtns").innerHTML = CFG.motor_names.map((m, i) => `<button id="mot${i}" class="btn mot" disabled>${m}</button>`).join("");
  CFG.motor_names.forEach((m, i) => $("mot" + i).addEventListener("click", () => motToggle(i)));
  $("motOff").addEventListener("click", () => motRelease(true));
  $("motDuty").addEventListener("input", motChanged);
  $("motHz").addEventListener("change", motChanged);
  motRender();
  $("statusBtn").addEventListener("click", () => sendCmd({ type: "cmd", cmd: "S" }));
  $("recBtn").addEventListener("click", startRecording);
  $("recMix").addEventListener("change", () => { if (benchState.play) playClip(benchState.play.label, benchState.play.name, false); });
  let want = location.hash === "#bench";
  try { want = want || localStorage.getItem("bench") === "1"; } catch (e) { /* storage may be unavailable */ }
  if (want) benchShow(true);
  window.addEventListener("hashchange", () => benchShow(location.hash === "#bench"));   // #bench also works on a loaded page
}

function benchShow(on) {
  if (on && window.calibShow) window.calibShow(false);     // one panel at a time (calibration.js)
  benchState.on = on;
  $("bench").hidden = !on;
  document.body.classList.toggle("bench", on);
  $("benchBtn").classList.toggle("on", on);
  try { localStorage.setItem("bench", on ? "1" : "0"); } catch (e) { /* ignore */ }
  if (!on) { tapTest(false); motRelease(motAny()); }
  if (on) loadRecordings();
}

// ---- motors by hand: 'h<F>,<FR>,...,<FL>,<hz>' to the board (each motor on / off at one strength, steady or pulsed),
// 'A' (duty null) hands them back to the detector. The board does that by itself 3 s after the last 'h', so the
// state is repeated every second while a motor is on: closing the page never leaves one running.
const MOT_DUTY_MAX = 190;         // firmware/esp32s3/pins.h: the PWM cap, ~3 V average across the 3 V motors
const MOT_IDS = () => [...CFG.motor_names.map((m, i) => "mot" + i), "motOff", "motDuty", "motHz"];
let motTimer = null;              // keepalive interval while a motor is on
let motThrottle = null;           // slider drags send at most every 100 ms
const motAny = () => benchState.mot.some(Boolean);

function motSend() {
  const duty = Number($("motDuty").value);
  const msg = { type: "haptics", duty: benchState.mot.map((on) => (on ? duty : 0)), hz: Number($("motHz").value) };
  const key = JSON.stringify(msg);
  if (key !== benchState.motLast) { benchState.motLast = key; benchState.motAsk = Date.now(); }   // the board replies to new states only
  sendCmd(msg);
}

function motToggle(i) {
  const maxOn = CFG.motor_max_on ?? 3;
  if (!benchState.mot[i] && benchState.mot.filter(Boolean).length >= maxOn) {   // the board would drop one anyway
    const n = $("motManNote"); n.className = "hint bad";
    n.textContent = `At most ${maxOn} motors at once (the 1 A supply): turn one off first.`;
    return;
  }
  benchState.mot[i] = !benchState.mot[i];
  if (!motAny()) { motRelease(true); return; }               // all off: the detector drives them again
  if (!motTimer) motTimer = setInterval(motSend, 1000);
  motChanged();
}

function motChanged() {
  motRender();
  if (motAny() && !motThrottle) motThrottle = setTimeout(() => { motThrottle = null; if (motAny()) motSend(); }, 100);
}

function motRelease(send) {
  benchState.mot = CFG.motor_names.map(() => false); benchState.motLast = ""; benchState.motAsk = 0;
  if (motTimer) { clearInterval(motTimer); motTimer = null; }
  if (send) sendCmd({ type: "haptics", duty: null });
  motRender();
}

function motRender() {
  const duty = Number($("motDuty").value);
  benchState.mot.forEach((on, i) => $("mot" + i).classList.toggle("on", on));
  const v = $("motDutyV");
  v.textContent = `${Math.round(100 * duty / MOT_DUTY_MAX)}% · ≈${(duty / 255 * 4).toFixed(1)} V`;   // 5 V rail less the ULN's ~1 V
  v.title = `PWM duty ${duty} / 255 (cap ${MOT_DUTY_MAX})`;
}

// ---- recording into data/team (the training folders)
async function startRecording() {
  motRelease(motAny());                                     // no motor noise in a training clip
  const body ={ label: $("recLabel").value, seconds: Number($("recSecs").value), holdout: $("recHoldout").checked,
                 tag: $("recTag").value.trim() };
  benchState.recBusy = true; $("recBtn").disabled = true; $("recBtn").classList.add("busy");
  $("recNote").className = "hint"; $("recNote").textContent = "starting…"; $("recBar").style.width = "0";
  $("recBar").parentElement.classList.remove("ok");
  try {
    const r = await fetch("/api/record", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!(r.headers.get("content-type") || "").includes("json")) {      // e.g. 404 from a dashboard started before this feature existed
      recIdle(`server returned ${r.status} ${r.statusText}: restart scripts/dashboard.py and reload this page`, true); return;
    }
    const d = await r.json();
    if (!d.ok) { recIdle(`refused: ${d.error || r.status}`, true); return; }
    $("recNote").textContent = `recording ${d.name} …`;
  } catch (e) {
    recIdle(`request failed: ${e}`, true);
  }
}

function recIdle(note, bad) {
  benchState.recBusy = false; $("recBtn").classList.remove("busy"); $("recBtn").disabled = source !== "serial";
  if (note != null) { $("recNote").textContent = note; $("recNote").className = "hint" + (bad ? " bad" : ""); }
}

const fmtLv = (lv) => (lv || []).map((l) => `${l.ch} ${l.peak.toFixed(0)}/${l.rms.toFixed(0)} dB${Math.abs(l.dc) > 0.01 ? ` dc ${(100 * l.dc).toFixed(1)}%` : ""}`).join(" · ");

function onRecord(d) {
  benchState.rec = d;
  if (d.state === "starting") {
    $("recNote").textContent = `recording ${d.name} … waiting for raw packets`;
  } else if (d.state === "recording") {
    $("recBar").style.width = `${100 * clamp01(d.elapsed / d.seconds)}%`;
    $("recNote").textContent = `${d.elapsed.toFixed(1)} / ${d.seconds} s · ${d.chunks} chunks` +
      (d.dropped ? ` · ${d.dropped} dropped` : "") + (d.no_packets ? " · NO PACKETS: is this the native USB port?" : "") +
      (d.levels ? ` · ${fmtLv(d.levels)}` : "");
    $("recNote").className = "hint" + (d.no_packets || d.dropped ? " bad" : "");
  } else if (d.state === "done") {
    $("recBar").style.width = "100%"; $("recBar").parentElement.classList.add("ok");
    const bad = /CLIPPING|SILENT|DC offset|STOPPED/.test(d.verdict) || d.dropped > 0;
    recIdle(`saved ${d.path}: ${d.seconds_recorded.toFixed(1)} s, ${d.chunks} chunks` +
      (d.dropped ? `, ${d.dropped} dropped (zero-filled)` : "") + ` · ${d.verdict} · ${fmtLv(d.levels)} · checking…`, bad);
    loadRecordings();
  } else if (d.state === "check") {
    const n = $("recNote"), t = n.textContent.replace(/ · checking…$/, "");
    if (d.error) { n.textContent = `${t} · check failed: ${d.error}`; return; }
    const pct = Math.round(100 * d.detected_frac);
    const want = d.label === "drone";
    const ok = want ? d.detected_frac >= 0.5 : d.detected_frac === 0;
    n.textContent = `${t} · detector (rules only, first ${Math.round(d.checked_s)} s): ${pct}% of frames DRONE` +
      (d.first_detect_s != null ? `, first at ${d.first_detect_s.toFixed(1)} s` : "") +
      (d.theta_median != null ? `, median bearing ${fmtDeg(d.theta_median)}` : "") +
      (ok ? " · as expected" : want ? " · weak: closer, louder, or longer helps" : " · false alarms: a good hard negative");
    if (!ok && !n.className.includes("bad")) n.className = "hint bad";
  } else if (d.state === "error") {
    recIdle(`recording failed: ${d.error}` + (d.seconds_recorded ? ` (${d.seconds_recorded.toFixed(1)} s kept in ${d.path})` : ""), true);
    loadRecordings();
  }
}

async function loadRecordings() {
  try {
    const d = await (await fetch("/api/recordings")).json();
    const s = d.summary || {}, min = (x) => (x / 60).toFixed(1);
    const part = (lab) => {
      const t = s[lab]; if (!t) return `${lab}: none`;
      return `${lab}: ${t.clips} clip${t.clips === 1 ? "" : "s"} · ${min(t.seconds)} min` +
        (t.holdout_clips ? ` (${t.holdout_clips} holdout, ${min(t.holdout_seconds)} min)` : "") +
        (t.unknown_length ? ` · ${t.unknown_length} without length` : "");
    };
    const last = (d.files || [])[0];
    $("recData").textContent = `${d.folder}/ — ${part("drone")} — ${part("background")}` +
      (last ? "" : " — empty: record something") + ` — then: python scripts/train.py`;
    const files = (d.files || []).slice(0, 8);
    $("recList").innerHTML = files.map((f, i) =>
      `<li data-i="${i}"><span class="lab">${f.label}${f.holdout ? " ·h" : ""}</span><span class="nm" title="${escapeHtml(f.name)}">${escapeHtml(f.name)}</span>` +
      `<span>${f.seconds != null ? `${Math.round(f.seconds)} s` : ""}</span><button type="button">play</button></li>`).join("");
    $("recList").querySelectorAll("li").forEach((li) => {
      const f = files[Number(li.dataset.i)];
      li.querySelector("button").addEventListener("click", () => playClip(f.label, f.name, true));
    });
    if (last && !benchState.play) playClip(last.label, last.name, false);     // newest clip loaded, not started
  } catch (e) {
    $("recData").textContent = `dataset summary unavailable: ${e}`;
  }
}

// Listening mix of a clip through /api/clip (Finder and QuickTime refuse the 3-channel originals; the mix is
// stereo L+R, mono or one channel, peak-normalised for listening; the file on disk is untouched).
function playClip(label, name, start) {
  benchState.play = { label, name };
  const a = $("recAudio");
  a.src = `/api/clip?label=${encodeURIComponent(label)}&name=${encodeURIComponent(name)}&mix=${$("recMix").value}`;
  a.title = `${label}/${name}`;
  $("recList").querySelectorAll("li").forEach((li) => li.classList.toggle("playing", li.querySelector(".nm").textContent === name));
  if (start) a.play().catch(() => { /* autoplay blocked: the controls still work */ });
}

function benchEnable(ok) {
  for (const id of ["tapBtn", "motorBtn", "statusBtn"]) $(id).disabled = !ok;
  $("recBtn").disabled = !ok || benchState.recBusy;
  $("benchNote").textContent = ok ? "tap test · motor test · board status" : "needs the device: start the dashboard with --port or --tcp";
  if (!ok) { tapTest(false); motRelease(false); }
}

function tapTest(on) {
  benchState.tap = on;
  $("tapBtn").textContent = on ? "Stop tap test" : "Start tap test";
  $("tapBtn").classList.toggle("on", on);
  if (tapTimer) { clearInterval(tapTimer); tapTimer = null; }
  if (on) {
    sendCmd({ type: "levels" });
    tapTimer = setInterval(() => sendCmd({ type: "levels" }), 1000);   // the server polls 'L' while we keep asking
  }
}

function onBench(d) {
  if (window.onCalibBench) window.onCalibBench(d);          // calibration.js: levels, cal_capture, cal, haptics
  if (d.kind === "levels") {
    benchState.lastLevels = Date.now();
    const chOf = {};
    for (const [ch, w] of Object.entries(d.map || {})) chOf[w] = ch;
    for (const s of d.words || []) {
      const w = s.w, el = $("slot" + w); if (!el) continue;
      $("slotPk" + w).style.width = `${100 * clamp01((s.peak - DB_FLOOR) / -DB_FLOOR)}%`;
      $("slotRms" + w).style.width = `${100 * clamp01((s.rms - DB_FLOOR) / -DB_FLOOR)}%`;
      $("slotPkV" + w).textContent = `${s.peak.toFixed(0)} dBFS · rms ${s.rms.toFixed(0)}`;
      $("slotDc" + w).textContent = `dc ${(100 * s.dc).toFixed(1)}%`;
      $("slotCh" + w).textContent = chOf[w] || "";
      el.classList.toggle("bad", Math.abs(s.dc) > 0.2);
      el.classList.toggle("hot", Math.abs(s.dc) <= 0.2 && s.peak > -20);     // a tap (or a loud room)
    }
    const h = $("slotHints"), hints = d.hints || [];
    h.textContent = hints.length ? hints.join(" · ") : "Tap each mic with a fingernail: one slot's peak jumps per mic.";
    h.classList.toggle("bad", hints.some((x) => /stuck|same SEL/.test(x)));
  } else if (d.kind === "record") {
    onRecord(d);
  } else if (d.kind === "dsp") {
    benchDsp = d;
  } else if (d.kind === "led") {
    if (typeof d.stealth === "boolean") { ledState.stealth = d.stealth; ledRender(); }
  } else if (d.kind === "motor_test") {
    benchMotor = d;
    $("motorNote").textContent = `${d.text} (${hhmmss(d.wall)})`;
  } else if (d.kind === "haptics") {
    benchState.motAck = Date.now();
    const head = d.text.split(" pulse_hz=")[0], hz = (d.text.match(/pulse_hz=(\d+)/) || [])[1];
    const on = [...head.matchAll(/\b([A-Z]{1,2})=(\d+)/g)].filter((x) => x[2] !== "0").map((x) => `${x[1]} ${x[2]}`);
    const n = $("motManNote");
    n.className = "hint";
    n.textContent = !d.manual ? `Board: the detector drives the motors again (${hhmmss(d.wall)})`
      : hz != null ? `Board: duty ${on.length ? on.join(" · ") : "all 0"} of 255 · ${hz === "0" ? "steady" : `${hz} Hz`} (${hhmmss(d.wall)})`
      : d.text;
  }
}

function renderBench() {
  if (!benchState.on) return;
  const motOk = source === "serial" && !benchState.recBusy;
  for (const id of MOT_IDS()) { const el = $(id); if (el) el.disabled = !motOk; }
  if (motAny() && benchState.motAsk && benchState.motAck < benchState.motAsk && Date.now() - benchState.motAsk > 2000) {
    const n = $("motManNote"), t = "No reply to 'h' from the board: flash the current firmware (firmware/esp32s3/flash.sh).";
    if (n.textContent !== t) { n.textContent = t; n.className = "hint bad"; }
  }
  const st = status || {}, rows = [];
  if (benchDsp && benchDsp.total != null) {
    const b = benchDsp.budget || CFG.chunk_us_expect || 21333, pct = Math.round(100 * benchDsp.total / b);
    rows.push(["DSP load", `${(benchDsp.total / 1000).toFixed(1)} / ${(b / 1000).toFixed(1)} ms (${pct}%)`, pct < 90 ? "good" : "bad"]);
    rows.push(["Stages", `spec ${benchDsp.spec} · mel ${benchDsp.mel} · det ${benchDsp.det} · bear ${benchDsp.bear} µs`]);
    rows.push(["DMA overflows", benchDsp.dma_ovf, benchDsp.dma_ovf ? "bad" : "good"]);
  } else {
    rows.push(["DSP load", st.connected ? "waiting for the first status reply" : "—"]);
  }
  if (st.pdm_us != null) rows.push(["Chunk period", `${st.pdm_us} µs`, Math.abs(st.pdm_us - (CFG.chunk_us_expect || 21333)) < 600 ? "good" : "bad"]);
  rows.push(["Free heap", st.free_heap != null ? `${(st.free_heap / 1024).toFixed(0)} KB` : "—", st.free_heap != null && st.free_heap < 20000 ? "bad" : ""]);
  if (st.psram_free != null) rows.push(["PSRAM free", `${(st.psram_free / 1048576).toFixed(2)} MB`]);
  if (st.dsp_stack_min != null) rows.push(["DSP stack free", `${(st.dsp_stack_min / 1024).toFixed(1)} KB`, st.dsp_stack_min < 2048 ? "bad" : "good"]);
  if (st.boot_line) rows.push(["Boot", `PSRAM bufs ${st.psram_bufs || "?"} · tasks failed ${st.tasks_failed || "?"}`, st.tasks_failed && st.tasks_failed !== "0x00" ? "bad" : "good"]);
  if (st.wifi_mode) rows.push(["Wi-Fi", `${st.wifi_mode}${st.wifi_ip ? " · " + st.wifi_ip : ""}${st.wifi_client && st.wifi_client !== "none" ? " · client " + st.wifi_client : ""}`]);
  if (benchState.tap) rows.push(["Levels", benchState.lastLevels && Date.now() - benchState.lastLevels < 1500 ? "polling 4×/s" : "no reply yet", benchState.lastLevels && Date.now() - benchState.lastLevels < 1500 ? "good" : "bad"]);
  const html = rows.map(([k, v, c]) => `<dt>${k}</dt><dd class="${c || ""}">${escapeHtml(String(v))}</dd>`).join("");
  if (html !== benchState.devHtml) { benchState.devHtml = html; $("benchDev").innerHTML = html; }   // no DOM churn
}

// One pad per motor around the head (top view, nose up), at the motor's angle; built from /api/config's motor_angles.
function buildPads() {
  const g = $("hapPads"); if (!g) return;
  const NS = "http://www.w3.org/2000/svg";
  const key = CFG.motor_names.join(",");
  if (g.dataset.key === key) return;
  g.dataset.key = key; g.innerHTML = "";
  const cx = 120, cy = 114, rx = 90, ry = 88, sz = 38;
  for (const m of CFG.motor_names) {
    const a = (CFG.motor_angles[m] ?? 0) * Math.PI / 180;
    const x = cx + rx * Math.sin(a), y = cy - ry * Math.cos(a);
    const p = document.createElementNS(NS, "g"); p.setAttribute("class", "pad"); p.id = "pad" + m;
    const r = document.createElementNS(NS, "rect");
    Object.entries({ x: x - sz / 2, y: y - sz / 2, width: sz, height: sz, rx: 11 }).forEach(([k, v]) => r.setAttribute(k, v));
    const t = document.createElementNS(NS, "text"); t.setAttribute("x", x); t.setAttribute("y", y + 1); t.textContent = m;
    const q = document.createElementNS(NS, "text"); q.setAttribute("x", x); q.setAttribute("y", y + 14); q.setAttribute("class", "pct");
    p.append(r, t, q); g.append(p);
  }
}

function applyConfig(cfg) {
  CFG = Object.assign(CFG, cfg);
  if (!CFG.motor_names && CFG.motor_angles) CFG.motor_names = Object.keys(CFG.motor_angles);
  buildPads();
  $("thrScore").textContent = `threshold ${CFG.harm_thresh_db} dB`;
  $("thrHits").textContent = `alarm at ${CFG.persist_on} of ${CFG.persist_frames}`;
  $("thrMl").textContent = `gate ${CFG.ml_thresh}`;
}

let backoff = 500;
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => { wsOpen = true; backoff = 500; wsRef = ws; };
  ws.onclose = () => { wsOpen = false; setTimeout(connect, backoff); backoff = Math.min(backoff * 2, 5000); };
  ws.onerror = () => ws.close();
  ws.onmessage = (m) => {
    const d = JSON.parse(m.data);
    if (d.type === undefined) { pushFrame(d); return; }
    if (d.type === "status") {
      status = d; ledSync();
      if (window.onCalibSource) window.onCalibSource(source, !!d.connected);
      if (window.onBatteryStatus) window.onBatteryStatus(d);
      if (window.onBatterySource) window.onBatterySource(source, !!d.connected);
      return;
    }
    if (d.type === "event") { onEvent(d, true); return; }
    if (d.type === "bench") { onBench(d); return; }
    if (d.type === "hello") {
      clearBuffers(); events = []; last = null; lastLatency = null;
      applyConfig(d.config); setSource(d.source); status = d.status || {}; ledSync();
      for (const f of d.history || []) pushFrame(f);
      for (const e of d.events || []) onEvent(e, false);
      for (const e of events) if (e.kind === "detect_on" && e.latency_s != null) lastLatency = e.latency_s;
      detT = d.det_t; ovhT = d.ovh_t;
      if (last && !last.detected) detT = null;
    }
  };
}

readColors();
applyConfig({});
benchBuild();
$("ledBtn").addEventListener("click", ledToggle);
connect();
requestAnimationFrame(frame);
