"""pytest: the dashboard battery badge (webapp/static/battery.js), run under node: its pure functions, and the badge,
popover and low-battery toast driven through the window.onBattery* hooks against a stub DOM with a fake clock."""
import json, os, shutil, subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BATT_JS = os.path.join(ROOT, "webapp", "static", "battery.js")
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def node(script):
    """Run a node script with `B` = battery.js's exports; returns what it prints as JSON."""
    pre = "const B = require(process.env.BATT_JS);\n"
    r = subprocess.run([NODE, "-e", pre + script], capture_output=True, text=True, timeout=30,
                       env={**os.environ, "BATT_JS": BATT_JS})
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_parses_as_a_browser_script():
    r = subprocess.run([NODE, "-e", "new Function(require('fs').readFileSync(process.env.BATT_JS, 'utf8'))"],
                       capture_output=True, text=True, timeout=30, env={**os.environ, "BATT_JS": BATT_JS})
    assert r.returncode == 0, r.stderr


def test_pct_from_v_follows_the_lipo_curve():
    out = node("console.log(JSON.stringify([4.2, 4.1, 4.0, 3.9, 3.8, 3.75, 3.7, 3.65, 3.6, 3.5, 3.3, 3.85, 4.3, 3.1,"
               " 0, 0.3, null, NaN, undefined, '3.9'].map(B.pctFromV)))")
    assert out[:11] == [100, 90, 80, 65, 50, 40, 30, 20, 12, 5, 0]
    assert out[11] == pytest.approx(57.5)
    assert out[12] == 100 and out[13] == 0                       # clamped above full / below empty
    assert out[14:] == [None] * 6                                 # 0.00 = not read yet, no battery, junk -> unknown


def test_ema_settles_with_a_30_s_time_constant():
    out = node("""
      let e = null; e = B.emaStep(e, 4.0, 0, 30); const first = e;
      for (let t = 1; t <= 30; t++) e = B.emaStep(e, 3.8, 1, 30);
      const tau = e;
      for (let t = 31; t <= 150; t++) e = B.emaStep(e, 3.8, 1, 30);
      console.log(JSON.stringify({first, tau, settled: e, still: B.emaStep(3.9, 4.2, 0, 30)}));""")
    assert out["first"] == 4.0
    assert out["tau"] == pytest.approx(3.8 + 0.2 * 0.3679, abs=1e-3)   # 1 - 1/e of the step after one tau
    assert out["settled"] == pytest.approx(3.8, abs=0.002)
    assert out["still"] == 3.9                                      # dt 0: no change


def test_slope_time_left_and_hysteresis():
    out = node("""
      const s = []; for (let t = 0; t <= 300; t += 1) s.push({t, y: 50 - t / 120, v: 3.8});
      const sl = B.slope(s, 300, 120, 'y');
      const few = B.slope([{t: 0, y: 1}], 10, 60, 'y');
      const flat = B.slope(s, 300, 300, 'v');
      const shown = []; let p = null;
      for (const x of [50.2, 50.6, 50.9, 51.0, 50.4, 49.9, 49.2]) { p = B.showPct(p, x); shown.push(p); }
      console.log(JSON.stringify({sl, few, flat, shown,
        left: B.timeLeftS(40, -0.5 / 60), slow: B.timeLeftS(40, -0.1 / 60), up: B.timeLeftS(40, 0.2 / 60),
        none: B.timeLeftS(null, -1), fmt: [B.fmtLeft(80 * 60), B.fmtLeft(3 * 3600 + 5 * 60), B.fmtLeft(20), B.fmtLeft(3e5), B.fmtLeft(null)]}));""")
    assert out["sl"]["slope"] == pytest.approx(-1 / 120) and out["sl"]["span"] == 120 and out["sl"]["n"] == 121
    assert out["few"]["slope"] is None
    assert out["flat"]["slope"] == pytest.approx(0)
    assert out["shown"] == [50, 50, 51, 51, 51, 50, 49]           # moves only once 0.75 % away from what is shown
    assert out["left"] == pytest.approx(80 * 60)                  # 40 % at 0.5 %/min
    assert out["slow"] is None and out["up"] is None and out["none"] is None
    assert out["fmt"] == ["~1 h 20 min", "~3 h 05 min", "~1 min", "> 2 days", None]


def test_state_thresholds_and_charging_inference():
    out = node("""
      const mVmin = (x) => x / 1000 / 60;
      console.log(JSON.stringify({
        unknown: [B.stateOf(0, null, 0, true), B.stateOf(null, null, null, false)],
        good: B.stateOf(3.8, 50, mVmin(-3), true), low: B.stateOf(3.64, 18, mVmin(-3), true),
        low20: B.stateOf(3.65, 20, mVmin(-3), true), crit: B.stateOf(3.55, 8, mVmin(-3), true),
        rising: B.stateOf(3.9, 65, mVmin(5), true), risingNoTrend: B.stateOf(3.9, 65, mVmin(5), false),
        fullFlat: B.stateOf(4.17, 100, mVmin(0.1), true), fullFalling: B.stateOf(4.17, 100, mVmin(-2), true),
        fullNoTrend: B.stateOf(4.17, 100, null, false)}));""")
    assert out["unknown"] == ["unknown", "unknown"]
    assert (out["good"], out["low"], out["low20"], out["crit"]) == ("good", "low", "low", "critical")
    assert out["rising"] == "charging" and out["risingNoTrend"] == "good"
    assert out["fullFlat"] == "charging" and out["fullFalling"] == "good" and out["fullNoTrend"] == "good"


# A stub DOM: every querySelector returns a remembered stub element, so render() runs end to end; the clock is
# performance.now() under the test's control and setInterval only records its callback.
DOM = r"""
const fs = require('fs');
let T = 0; const toasts = []; const byId = {}; const ticks = [];
function mk(sel) {
  const e = { sel, id: '', className: '', title: '', hidden: sel === '.batt-pop', attrs: {}, on: {}, kids: {}, _t: '',
    setAttribute(k, v) { this.attrs[k] = String(v); }, getAttribute(k) { return this.attrs[k]; },
    addEventListener(ev, fn) { (this.on[ev] = this.on[ev] || []).push(fn); },
    querySelector(s) { return this.kids[s] || (this.kids[s] = mk(s)); }, contains() { return false; } };
  Object.defineProperty(e, 'textContent', { get() { return this._t; },
    set(v) { this._t = String(v); if (sel === '.batt-toast-t') toasts.push(this._t); } });
  Object.defineProperty(e, 'innerHTML', { get() { return ''; }, set(v) { this.kids = {}; } });
  return e;
}
const badge = mk('#battBadge'); byId.battBadge = badge;
const document = { readyState: 'complete', getElementById: (id) => byId[id] || null, createElement: () => mk('div'),
  addEventListener() {}, body: { appendChild(c) { if (c.id) byId[c.id] = c; } } };
const window = {};
const performance = { now: () => T * 1000 };
new Function('window', 'document', 'performance', 'setInterval', 'module', fs.readFileSync(process.env.BATT_JS, 'utf8'))(
  window, document, performance, (fn) => ticks.push(fn), undefined);
const q = (s) => badge.querySelector(s);
const snap = () => ({ cls: badge.className, txt: q('.batt-txt').textContent, title: q('.batt-btn').title,
  fill: Number(q('.batt-fill').getAttribute('width')), v: q('.batt-v').textContent, s: q('.batt-s').textContent,
  l: q('.batt-l').textContent, tr: q('.batt-tr').textContent, spark: q('.batt-spark-l').getAttribute('points') || '' });
const feed = (t0, t1, vOf, perS) => { for (let t = t0; t < t1; t += 1 / (perS || 3)) { T = t; window.onBatteryFrame({ batt_v: vOf(t) }); } };
"""


def dom(script):
    pre = DOM + "\n"
    r = subprocess.run([NODE, "-e", pre + script], capture_output=True, text=True, timeout=60,
                       env={**os.environ, "BATT_JS": BATT_JS})
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_badge_unknown_until_a_reading():
    out = dom("""
      const s0 = snap();
      window.onBatterySource('serial', true); feed(0, 5, () => 0); window.onBatteryFrame({ batt_v: null });
      window.onBatteryFrame({}); window.onBatteryStatus({ batt_v: 0 });
      const s1 = snap();
      console.log(JSON.stringify({ s0, s1, hooks: ['onBatteryFrame', 'onBatterySource', 'onBatteryStatus'].map((k) => typeof window[k]) }));""")
    for s in (out["s0"], out["s1"]):
        assert s["cls"] == "batt unknown" and s["txt"] == "—" and s["fill"] == 0
        assert "No battery data" in s["title"]
    assert out["hooks"] == ["function"] * 3


def test_discharge_warns_once_at_15_and_once_at_5_percent():
    out = dom("""
      window.onBatterySource('serial', true);
      badge.on.mouseenter[0]();                                     // open the popover so its rows render
      const v = (t) => 3.72 - 0.0005 * t + 0.004 * Math.sin(t * 2.1); // 30 mV / min down, a little ADC noise
      feed(0, 10, v); const early = snap();
      feed(10, 150, v); const mid = snap();
      feed(150, 900, v); const end = snap();
      console.log(JSON.stringify({ early, mid, end, toasts }));""")
    assert out["early"]["cls"] == "batt good" and out["early"]["l"] == "estimating…"
    assert out["mid"]["l"].startswith("~") and out["mid"]["l"].endswith("min")         # time left once 2 min of data
    assert "mV/min" in out["mid"]["tr"] and out["mid"]["tr"].startswith("−")
    assert out["mid"]["spark"].count(",") > 100
    assert out["end"]["cls"] == "batt critical" and out["end"]["txt"] == "0%"
    assert len(out["toasts"]) == 2, out["toasts"]
    assert out["toasts"][0].startswith("Battery low") and out["toasts"][1].startswith("Battery critical")


def test_charging_stale_and_source_reset():
    out = dom("""
      window.onBatterySource('serial', true);
      feed(0, 240, (t) => 3.90 + 0.0003 * t);                      // +18 mV / min: on the charger
      const charging = snap();
      window.onBatterySource('serial', true);                      // same source: history kept
      T = 255; ticks[0](); const stale = snap();                   // 15 s without a reading
      window.onBatterySource('fake', true); const reset = snap();  // new source: back to unknown
      window.onBatterySource('fake', false); feed(300, 310, () => 3.95);
      const disconnected = snap();
      console.log(JSON.stringify({ charging, stale, reset, disconnected }));""")
    assert out["charging"]["cls"] == "batt charging" and "charging?" in out["charging"]["title"]
    assert "left" not in out["charging"]["title"]
    assert out["stale"]["cls"] == "batt charging stale" and "last reading" in out["stale"]["title"]
    assert out["reset"]["cls"] == "batt unknown" and out["reset"]["txt"] == "—"
    assert out["disconnected"]["cls"].endswith(" stale")
