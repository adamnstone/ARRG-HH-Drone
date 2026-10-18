// Battery badge for the header: charge of the headset's LiPo cell from the batt_v the board reports once a second
// (T lines, so every frame carries the latest reading; 0.00 / null = not read yet or no battery sensed).
//
// Integration (the page owns these hooks; this file only defines the window.onBattery* functions):
//   index.html  <head>:   <link rel="stylesheet" href="/static/battery.css">
//               header (in .meta, next to the Bench / Calibrate buttons):   <span id="battBadge"></span>
//               after app.js:   <script src="/static/battery.js"></script>
//   app.js      for every frame object (live and the hello history):   if (window.onBatteryFrame) window.onBatteryFrame(f);
//               on source / connection changes (setSource and status messages):
//                   if (window.onBatterySource) window.onBatterySource(source, !!status.connected);
//               for status messages (fallback reading when frames carry no batt_v):
//                   if (window.onBatteryStatus) window.onBatteryStatus(d);
//
// Hardware: Adafruit PowerBoost 1000C, battery sense on ADC1 GPIO 1 through a 1:2 divider (pins.h). Its charge and LBO
// pins are not wired, so "charging" is inferred from the voltage (rising, or near full and not falling), hence the
// question mark. Percent comes from an approximate single-cell LiPo open-circuit curve, applied to a ~30 s EMA of the
// voltage so motor-load sag and ADC noise do not make it jump; the shown integer changes with hysteresis.
(function () {
  "use strict";

  // ---------------------------------------------------------------- pure functions (exported for tests)
  // Approximate resting-voltage -> state-of-charge table for one LiPo cell (typical published curves; real cells,
  // temperature and load differ by several percent, so the badge says "~").
  const CURVE = [[3.30, 0], [3.50, 5], [3.60, 12], [3.65, 20], [3.70, 30], [3.75, 40], [3.80, 50], [3.90, 65],
                 [4.00, 80], [4.10, 90], [4.20, 100]];
  const V_VALID = 0.5;          // below this the ADC reads no battery (0.00 = not read yet)

  function validV(v) { return typeof v === "number" && isFinite(v) && v > V_VALID; }

  function pctFromV(v) {        // float percent 0..100, null for no reading
    if (!validV(v)) return null;
    if (v <= CURVE[0][0]) return 0;
    if (v >= CURVE[CURVE.length - 1][0]) return 100;
    for (let i = 1; i < CURVE.length; i++) {
      const [v1, p1] = CURVE[i];
      if (v <= v1) {
        const [v0, p0] = CURVE[i - 1];
        return p0 + (p1 - p0) * (v - v0) / (v1 - v0);
      }
    }
    return 100;
  }

  function emaStep(prev, x, dtS, tauS) {   // exponential smoothing with time constant tauS for an update dtS apart
    if (prev == null || !isFinite(prev)) return x;
    const a = 1 - Math.exp(-Math.max(0, dtS) / tauS);
    return prev + a * (x - prev);
  }

  // Least-squares slope (units per second) of samples [{t (s), y}] within the last spanS seconds before nowS.
  function slope(samples, nowS, spanS, key) {
    const k = key || "y";
    const pts = samples.filter((s) => s.t >= nowS - spanS && s[k] != null);
    const n = pts.length;
    if (n < 2) return { slope: null, n, span: 0 };
    const span = pts[n - 1].t - pts[0].t;
    let st = 0, sy = 0;
    for (const p of pts) { st += p.t; sy += p[k]; }
    const mt = st / n, my = sy / n;
    let num = 0, den = 0;
    for (const p of pts) { num += (p.t - mt) * (p[k] - my); den += (p.t - mt) * (p.t - mt); }
    return { slope: den > 0 ? num / den : null, n, span };
  }

  // Seconds until empty from the smoothed percent and its slope (% per s), null when not discharging measurably.
  const MIN_DRAIN = -0.3 / 60;             // % per s: slower than 0.3 %/min counts as steady (noise / idle)
  function timeLeftS(pct, slopePctPerS) {
    if (pct == null || slopePctPerS == null || slopePctPerS > MIN_DRAIN) return null;
    return Math.max(0, pct / -slopePctPerS);
  }

  // "unknown" | "charging" | "critical" | "low" | "good". vSlope: smoothed volts per second (null if unknown).
  const RISE = 2e-3 / 60;                  // V/s: +2 mV a minute is rising (a charging cell climbs faster)
  const FALL = -0.5e-3 / 60;               // V/s: falling more than 0.5 mV a minute is discharging
  function stateOf(v, pct, vSlope, haveTrend) {
    if (!validV(v) || pct == null) return "unknown";
    if (haveTrend && vSlope != null && (vSlope > RISE || (v >= 4.15 && vSlope >= FALL))) return "charging";
    if (pct <= 8) return "critical";
    if (pct <= 20) return "low";
    return "good";
  }

  // Integer percent to show: follows pct, but only moves when it is at least `band` away from what is shown, so a
  // value sitting on a .5 boundary does not flicker.
  function showPct(prevShown, pct, band) {
    if (pct == null) return null;
    const b = band == null ? 0.75 : band;
    if (prevShown == null || Math.abs(pct - prevShown) >= b) return Math.round(pct);
    return prevShown;
  }

  function fmtLeft(s) {
    if (s == null) return null;
    if (s >= 48 * 3600) return "> 2 days";
    const h = Math.floor(s / 3600), m = Math.round((s % 3600) / 60);
    return h ? `~${h} h ${String(m).padStart(2, "0")} min` : `~${Math.max(1, m)} min`;
  }

  const api = { CURVE, pctFromV, emaStep, slope, timeLeftS, stateOf, showPct, fmtLeft, validV };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window === "undefined" || typeof document === "undefined") return;   // node (tests): no DOM
  window.__battTest = api;

  // ---------------------------------------------------------------- live state
  const TAU_S = 30;              // EMA time constant of the voltage
  const SAMPLE_S = 1;            // the board reads the battery once a second; frames repeat the value in between
  const HIST_S = 15 * 60;        // history kept (the sparkline shows the last 5 min)
  const TREND_S = 3 * 60;        // charging / discharging decided over the last 3 min of the smoothed voltage
  const LEFT_S = 5 * 60;         // time-left from the smoothed percent's slope over the last 5 min
  const MIN_TREND_S = 60;        // ... and only once that much history exists
  const STALE_S = 10;            // no reading for this long -> the badge dims ("last reading hh:mm")
  const B = {
    src: "", connected: false,
    vRaw: null, vEma: null, tLast: null, tFirst: null, tReading: null,
    hist: [],                    // [{t, v (smoothed), p (smoothed %)}]
    shown: null, state: "unknown", left: null, pctSlope: null,
    warned15: false, warned5: false, pinned: false, built: false,
  };
  const nowS = () => performance.now() / 1000;
  const el = (id) => document.getElementById(id);

  function reset() {
    B.vRaw = B.vEma = B.tLast = B.tFirst = B.tReading = null;
    B.hist = []; B.shown = null; B.state = "unknown"; B.left = null; B.pctSlope = null;
  }

  function addReading(v) {
    if (!validV(v)) return;
    const t = nowS();
    if (B.tLast != null && t - B.tLast < SAMPLE_S * 0.95) { B.vRaw = v; B.tReading = t; return; }   // 1 sample / s
    B.vEma = emaStep(B.vEma, v, B.tLast == null ? 0 : t - B.tLast, TAU_S);
    B.vRaw = v; B.tLast = B.tReading = t;
    if (B.tFirst == null) B.tFirst = t;
    const p = pctFromV(B.vEma);
    B.hist.push({ t, v: B.vEma, p });
    while (B.hist.length && B.hist[0].t < t - HIST_S) B.hist.shift();
    update();
  }

  function update() {
    const t = nowS();
    const p = pctFromV(B.vEma);
    const age = B.tFirst == null ? 0 : t - B.tFirst;
    const vs = slope(B.hist, t, TREND_S, "v");
    const haveTrend = age >= MIN_TREND_S && vs.span >= MIN_TREND_S * 0.8;
    B.state = stateOf(B.vEma, p, vs.slope, haveTrend);
    B.shown = showPct(B.shown, p);
    const ps = slope(B.hist, t, LEFT_S, "p");
    B.pctSlope = ps.span >= MIN_TREND_S * 2 ? ps.slope : null;
    B.left = B.state === "charging" ? null : timeLeftS(p, B.pctSlope);
    lowNotice(p, age);
    render();
  }

  // ---------------------------------------------------------------- low-battery notice (once at 15 %, once at 5 %)
  function lowNotice(p, age) {
    if (p == null || B.state === "charging" || age < 20) return;          // let the EMA settle first
    if (p > 25) { B.warned15 = B.warned5 = false; return; }               // recharged: arm again for next time
    if (p < 5 && !B.warned5) { B.warned5 = B.warned15 = true; toast(`Battery critical: ~${Math.round(p)} % (${B.vEma.toFixed(2)} V). The headset will shut down soon; charge it now.`, "critical"); }
    else if (p < 15 && !B.warned15) { B.warned15 = true; toast(`Battery low: ~${Math.round(p)} % (${B.vEma.toFixed(2)} V). Charge the headset soon.`, "low"); }
  }

  function toast(text, level) {
    let t = el("battToast");
    if (!t) {
      t = document.createElement("div");
      t.id = "battToast"; t.className = "batt-toast"; t.setAttribute("role", "alert");
      t.innerHTML = `<span class="batt-toast-t"></span><button type="button" class="batt-toast-x" aria-label="dismiss">×</button>`;
      t.querySelector(".batt-toast-x").addEventListener("click", () => { t.hidden = true; });
      document.body.appendChild(t);
    }
    t.querySelector(".batt-toast-t").textContent = text;
    t.className = `batt-toast ${level}`;
    t.hidden = false;
  }

  // ---------------------------------------------------------------- rendering
  const LABEL = { good: "good", low: "low", critical: "critical", charging: "charging?", unknown: "no battery data" };

  function build() {
    const host = el("battBadge");
    if (!host || B.built) return !!host;
    host.className = "batt";
    host.innerHTML =
      `<button type="button" class="batt-btn" aria-haspopup="true" aria-expanded="false">` +
      `<svg class="batt-ico" viewBox="0 0 28 14" aria-hidden="true">` +
      `<rect class="batt-shell" x="0.75" y="0.75" width="23.5" height="12.5" rx="2.6"/>` +
      `<rect class="batt-cap" x="25" y="4.4" width="2.4" height="5.2" rx="1"/>` +
      `<rect class="batt-fill" x="2.6" y="2.6" width="0" height="8.8" rx="1.3"/>` +
      `<path class="batt-bolt" d="M13.6 2.2 L9.4 7.6 h3.2 l-1.4 4.2 l4.6 -5.6 h-3.3 z"/></svg>` +
      `<span class="batt-txt">—</span></button>` +
      `<div class="batt-pop" role="dialog" aria-label="battery" hidden>` +
      `<div class="batt-row"><span>Voltage</span><b class="batt-v">—</b></div>` +
      `<div class="batt-row"><span>Charge</span><b class="batt-p">—</b></div>` +
      `<div class="batt-row"><span>State</span><b class="batt-s">—</b></div>` +
      `<div class="batt-row"><span>Time left</span><b class="batt-l">—</b></div>` +
      `<svg class="batt-spark" viewBox="0 0 200 44" preserveAspectRatio="none" aria-hidden="true">` +
      `<polyline class="batt-spark-l" points=""/></svg>` +
      `<div class="batt-foot">last 5 min · <span class="batt-tr">—</span></div>` +
      `<div class="batt-note">Approximate: single-cell LiPo resting-voltage curve on a 30 s average. Charging is ` +
      `inferred from the voltage (the PowerBoost's charge pin is not wired).</div></div>`;
    const btn = host.querySelector(".batt-btn"), pop = host.querySelector(".batt-pop");
    const open = (on) => { pop.hidden = !on; btn.setAttribute("aria-expanded", on ? "true" : "false"); if (on) render(); };
    host.addEventListener("mouseenter", () => open(true));
    host.addEventListener("mouseleave", () => { if (!B.pinned) open(false); });
    btn.addEventListener("click", () => { B.pinned = !B.pinned || pop.hidden; open(B.pinned); });
    btn.addEventListener("focus", () => open(true));
    btn.addEventListener("blur", () => { if (!B.pinned) open(false); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") { B.pinned = false; open(false); } });
    document.addEventListener("click", (e) => { if (B.pinned && !host.contains(e.target)) { B.pinned = false; open(false); } });
    B.built = true;
    return true;
  }

  function render() {
    if (!build()) return;
    const host = el("battBadge");
    const t = nowS();
    const stale = B.tReading != null && t - B.tReading > STALE_S;
    const st = B.state, p = B.shown;
    host.className = `batt ${st}${stale || (B.src && !B.connected && st !== "unknown") ? " stale" : ""}`;
    const fill = host.querySelector(".batt-fill");
    fill.setAttribute("width", st === "unknown" || p == null ? "0" : (19.8 * Math.max(0.04, p / 100)).toFixed(2));
    host.querySelector(".batt-txt").textContent = st === "unknown" ? "—" : `${p}%`;
    const tip = st === "unknown"
      ? "No battery data (no reading yet, or this source has no battery)"
      : `Battery ~${p} % · ${B.vEma.toFixed(2)} V · ${LABEL[st]}${B.left != null ? " · " + fmtLeft(B.left) + " left" : ""}${stale ? " · last reading " + Math.round(t - B.tReading) + " s ago" : ""}`;
    host.querySelector(".batt-btn").title = tip;
    const pop = host.querySelector(".batt-pop");
    if (pop.hidden) return;
    host.querySelector(".batt-v").textContent = B.vEma == null ? "—" : `${B.vEma.toFixed(2)} V` + (B.vRaw != null && Math.abs(B.vRaw - B.vEma) >= 0.01 ? ` (now ${B.vRaw.toFixed(2)})` : "");
    host.querySelector(".batt-p").textContent = p == null ? "—" : `~${p} %`;
    host.querySelector(".batt-s").textContent = LABEL[st] + (stale ? " (stale)" : "");
    const age = B.tFirst == null ? 0 : t - B.tFirst;
    host.querySelector(".batt-l").textContent = st === "unknown" ? "—" : st === "charging" ? "— (charging)"
      : B.left != null ? fmtLeft(B.left) : age < 2 * MIN_TREND_S ? "estimating…" : "steady (no measurable drain)";
    // sparkline of the smoothed voltage over the last 5 min, auto-scaled to at least 50 mV
    const pts = B.hist.filter((h) => h.t >= t - 300);
    const line = host.querySelector(".batt-spark-l");
    if (pts.length >= 2) {
      let lo = Math.min(...pts.map((h) => h.v)), hi = Math.max(...pts.map((h) => h.v));
      if (hi - lo < 0.05) { const m = (hi + lo) / 2; lo = m - 0.025; hi = m + 0.025; }
      line.setAttribute("points", pts.map((h) => `${(200 * (h.t - (t - 300)) / 300).toFixed(1)},${(41 - 38 * (h.v - lo) / (hi - lo)).toFixed(1)}`).join(" "));
    } else line.setAttribute("points", "");
    const vs = slope(B.hist, t, TREND_S, "v");
    host.querySelector(".batt-tr").textContent = vs.slope == null || vs.span < 20 ? "—"
      : `${vs.slope >= 0 ? "+" : "−"}${Math.abs(vs.slope * 60 * 1000).toFixed(1)} mV/min` +
        (B.pctSlope != null ? ` · ${B.pctSlope >= 0 ? "+" : "−"}${Math.abs(B.pctSlope * 60).toFixed(1)} %/min` : "");
  }

  // ---------------------------------------------------------------- hooks (called by app.js)
  window.onBatteryFrame = function (f) { if (f && f.batt_v != null) addReading(Number(f.batt_v)); };
  window.onBatteryStatus = function (st) {
    if (!st || st.batt_v == null) return;
    if (B.tReading == null || nowS() - B.tReading > 3) addReading(Number(st.batt_v));   // only when frames do not carry it
  };
  window.onBatterySource = function (src, connected) {
    if (src !== B.src) { B.src = src || ""; reset(); }
    B.connected = !!connected;
    render();
  };

  setInterval(() => { if (B.built) { if (B.vEma != null) update(); else render(); } }, 2000);   // stale / time-left refresh
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", render);
  else render();
})();
