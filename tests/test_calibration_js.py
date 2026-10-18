"""pytest: the Calibrate tab (webapp/static/calibration.js) run under node against a stub DOM and a stub server: the
Connected checkboxes of the five mic headers (MK1..MK5, any of them at any position: front or back, left, right,
top) and the motor outputs, front and back sharing one channel, the word with no header, the slot order (right first
by default, as the v3 board) and the word labels that follow it, the summary of what the board can do with the ticked
mics, the head diagram's mic sites, the device row, the auto-calibration proposal and what Save sends."""
import json, os, shutil, subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CAL_JS = os.path.join(ROOT, "webapp", "static", "calibration.js")
GEOM = os.path.join(ROOT, "webapp", "static", "headband_geometry.json")
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

# Every getElementById returns a remembered stub element (the tab builds its DOM with innerHTML, which the stub
# ignores); fetch answers /api/geometry and /api/calibration and records what is POSTed.
DOM = r"""
const fs = require('fs');
const byId = {}, posts = [];
function mk(id) {
  const cls = new Set();
  const e = { id, value: '', checked: false, disabled: false, hidden: false, title: '', className: '', dataset: {},
    style: {}, attrs: {}, on: {}, textContent: '', innerHTML: '', parentElement: null,
    classList: { toggle(c, on) { if (on === undefined ? !cls.has(c) : on) cls.add(c); else cls.delete(c); },
                 add(c) { cls.add(c); }, remove(c) { cls.delete(c); }, contains(c) { return cls.has(c); } },
    setAttribute(k, v) { this.attrs[k] = String(v); }, addEventListener(ev, fn) { (this.on[ev] = this.on[ev] || []).push(fn); },
    querySelector() { return mk(''); }, querySelectorAll() { return []; }, closest() { return null; },
    kids: [], append(...c) { this.kids.push(...c); } };
  e.parentElement = { classList: e.classList };
  return e;
}
const $ = (id) => byId[id] || (byId[id] = mk(id));
const fire = (id, ev, patch) => { Object.assign($(id), patch || {}); for (const fn of $(id).on[ev] || []) fn({ target: $(id) }); };
const geometry = JSON.parse(fs.readFileSync(process.env.GEOM, 'utf8'));
let saved = null, caps = [], solveResult = null, solveBody = null;
async function fetch(path, opt) {
  let body = {};
  if (path === '/api/geometry') body = geometry;
  else if (path === '/api/calibration' && (!opt || opt.method === 'GET')) body = { ok: true, calibration: saved, device: null, captures: caps };
  else if (path === '/api/calibration') { saved = JSON.parse(opt.body); posts.push(saved); body = { ok: true, calibration: saved }; }
  else if (path === '/api/calibration/solve') { solveBody = JSON.parse(opt.body || 'null'); body = solveResult; }
  return { ok: true, status: 200, headers: { get: () => 'application/json' }, json: async () => body };
}
const document = { activeElement: null, body: mk('body'), createElementNS: () => mk('') };
const window = { addEventListener() {} };
const ctx = { window, document, fetch, $, sendCmd() {}, status: { connected: true }, source: 'serial',
  escapeHtml: (s) => String(s), benchShow() {}, benchState: { on: false },
  location: { hash: '', pathname: '/', search: '' }, history: { replaceState() {} },
  localStorage: { getItem: () => null, setItem() {} }, performance: { now: () => 0 }, setInterval: () => 0,
  clearInterval() {}, setTimeout: () => 0, clearTimeout() {} };
new Function(...Object.keys(ctx), fs.readFileSync(process.env.CAL_JS, 'utf8'))(...Object.values(ctx));
const tick = () => new Promise((r) => setImmediate(r));
const mics = () => [0, 1, 2, 3, 4].map((k) => ({ on: $('calOn' + k).checked, pos: $('calPos' + k).value,
  w: $('calW' + k).textContent, dis: $('calPos' + k).disabled, hidden: $('calMic' + k).hidden, line: $('calW' + k).title }));
// the head diagram's mic sites as the last render drew them: {LABEL: {none, sub}}
const sites = () => {
  const out = {};
  for (const g of $('calSvg').kids) {
    if (!/cal-site/.test(g.attrs.class || '')) continue;
    const t = g.kids.filter((k) => k.textContent);
    out[t[0].textContent] = { none: /\bnone\b/.test(g.attrs.class), sub: t[1].textContent };
  }
  return out;
};
const motors = () => [0, 1, 2, 3, 4, 5, 6, 7].map((i) => ({ on: $('calMotOn' + i).checked, hole: $('calHole' + i).value }));
"""


def run(script):
    r = subprocess.run([NODE, "-e", DOM + "\n(async () => {\n" + script + "\n})().catch((e) => { console.error(e); process.exit(1); });"],
                       capture_output=True, text=True, timeout=60, env={**os.environ, "CAL_JS": CAL_JS, "GEOM": GEOM})
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_mic_headers_connected_and_slot_order():
    out = run("""
      window.calibShow(true); await tick(); await tick();
      const start = mics();
      fire('calOn1', 'change', { checked: false });              // MK2 (back) not connected
      fire('calOn4', 'change', { checked: false });              // MK5 (top) not connected
      const unticked = mics();
      fire('calSlot', 'change', { value: 'left_first' });        // the headers keep their positions, words swap
      const flipped = mics();
      fire('calOn1', 'change', { checked: true });               // MK2 back: its PCB position is free again
      fire('calSave', 'click'); await tick(); await tick();
      console.log(JSON.stringify({ start, unticked, flipped, back: mics(), saved: posts[posts.length - 1],
                                   hint: $('calMicHint').textContent, slot: $('calSlot').value }));""")
    s = out["start"]
    assert len(s) == 5 and [m["on"] for m in s] == [False, True, True, True, True]   # MK1 not connected
    assert [m["pos"] for m in s] == ["", "back", "left", "right", "top"] and s[0]["dis"]
    assert [m["w"] for m in s] == ["w1", "w0", "w3", "w2", "w5"]           # right first (the default): the v3 board
    # each header's physical slot stays put, whatever word it lands in: MK1 / MK3 / MK5 SEL GND, MK2 / MK4 SEL 3V3
    assert [m["line"].split(" (")[0] for m in s] == ["DMA word w1: DIN0 · SEL GND", "DMA word w0: DIN0 · SEL 3V3",
                                                     "DMA word w3: DIN1 · SEL GND", "DMA word w2: DIN1 · SEL 3V3",
                                                     "DMA word w5: DIN2 · SEL GND"]
    u = out["unticked"]
    assert [m["on"] for m in u] == [False, False, True, True, False] and u[1]["dis"] and u[1]["pos"] == ""
    f = out["flipped"]
    assert [m["w"] for m in f] == ["w0", "w1", "w2", "w3", "w4"]          # left first
    assert [m["line"].split(" (")[0] for m in f] == ["DMA word w0: DIN0 · SEL GND", "DMA word w1: DIN0 · SEL 3V3",
                                                     "DMA word w2: DIN1 · SEL GND", "DMA word w3: DIN1 · SEL 3V3",
                                                     "DMA word w4: DIN2 · SEL GND"]
    assert [m["pos"] for m in f] == ["", "", "left", "right", ""]
    saved = out["saved"]
    assert saved["slot_order"] == "left_first" and saved["mic_source"] == "manual" and out["slot"] == "left_first"
    assert saved["mic_words"] == {"front": None, "back": 1, "left": 2, "right": 3, "top": None}
    assert "3 of 4 positions have a mic" in out["hint"] and "assumed above" in out["hint"]


def test_mk1_takes_any_position_and_four_is_the_limit():
    out = run("""
      window.calibShow(true); await tick(); await tick();
      const r = {};
      fire('calOn0', 'change', { checked: true });               // MK1 with all four positions taken: stays off
      r.full = { m: mics(), msg: $('calMsg').textContent, cls: $('calMsg').className };
      fire('calOn1', 'change', { checked: false });              // MK2 off: back is free
      fire('calOn0', 'change', { checked: true });               // MK1 takes it (no PCB position of its own)
      r.mk1 = { m: mics(), msg: $('calMsg').textContent };
      fire('calPos0', 'change', { value: 'top' });               // a taken position swaps with that header (MK5)
      r.swap = mics();
      fire('calOn2', 'change', { checked: false }); fire('calOn0', 'change', { checked: false });
      fire('calOn0', 'change', { checked: true });               // first free in the order back, left, right, top
      r.first = mics();
      fire('calSave', 'click'); await tick(); await tick();
      r.saved = posts[posts.length - 1];
      console.log(JSON.stringify(r));""")
    full = out["full"]
    assert [m["on"] for m in full["m"][:5]] == [False, True, True, True, True]
    assert "All four positions are in use" in full["msg"] and "bad" in full["cls"]
    mk1 = out["mk1"]
    assert mk1["m"][0]["on"] and mk1["m"][0]["pos"] == "back" and not mk1["m"][1]["on"] and mk1["msg"] == ""
    assert [m["pos"] for m in out["swap"][:5]] == ["top", "", "left", "right", "back"]
    first = out["first"]
    assert [m["pos"] for m in first[:5]] == ["left", "", "", "right", "back"]
    # right first: MK1 w1, MK4 w2, MK5 w5
    assert out["saved"]["mic_words"] == {"front": None, "back": 5, "left": 1, "right": 2, "top": None}


@pytest.mark.parametrize("order,top,want", [("left_first", 5, 4), ("right_first", 4, 5)])
def test_word_without_a_header_moves_to_its_line_partner(order, top, want):
    # five headers, six DMA words: the one with no header (w5 left first, w4 right first) can only mirror MK5
    pcb = {"left_first": (1, 2, 3), "right_first": (0, 3, 2)}[order]
    out = run(f"""
      saved = {{ version: 1, mic_words: {{ back: {pcb[0]}, left: {pcb[1]}, right: {pcb[2]}, top: {top} }},
                mic_source: 'manual', slot_order: '{order}', motors: null, motor_source: 'default' }};
      window.calibShow(true); await tick(); await tick();
      const start = mics();
      fire('calSave', 'click'); await tick(); await tick();
      console.log(JSON.stringify({{ start, saved: posts[posts.length - 1], row5: $('calW5').textContent }}));""")
    s = out["start"]
    assert [m["on"] for m in s] == [False, True, True, True, True] and s[4]["pos"] == "top" and s[4]["w"] == f"w{want}"
    assert out["saved"]["mic_words"] == {"front": None, "back": pcb[0], "left": pcb[1], "right": pcb[2], "top": want}
    assert out["saved"]["slot_order"] == order                   # a file that names its slot order keeps it
    assert out["row5"] == ""                                     # no sixth row is ever rendered


def test_summary_follows_the_ticked_mics():
    out = run("""
      window.calibShow(true); await tick(); await tick();
      const h = () => $('calMicHint').textContent, r = {};
      r.four = h();
      fire('calOn4', 'change', { checked: false }); r.noTop = h();
      fire('calOn4', 'change', { checked: true }); fire('calOn1', 'change', { checked: false }); r.noBack = h();
      fire('calOn0', 'change', { checked: true }); r.mk1Back = h();          // MK1 fills back
      fire('calOn0', 'change', { checked: false }); fire('calOn1', 'change', { checked: true });
      fire('calOn2', 'change', { checked: false }); r.noLeft = h();
      fire('calOn2', 'change', { checked: true }); fire('calOn3', 'change', { checked: false }); r.noRight = h();
      fire('calOn1', 'change', { checked: false }); r.two = h();
      fire('calOn2', 'change', { checked: false }); fire('calOn4', 'change', { checked: false }); r.none = h();
      console.log(JSON.stringify(r));""")
    assert "4 of 4 positions have a mic" in out["four"] and "Bearing and elevation" in out["four"]
    assert "best three" in out["four"] and "best three" in out["mk1Back"] and "4 of 4" in out["mk1Back"]
    assert "3 of 4" in out["noTop"] and "elevation only when the drone is clearly above" in out["noTop"]
    assert "Front and back cannot be told apart" in out["noBack"] and "ambiguous" in out["noBack"]
    assert "front or back mic" in out["noBack"]
    for k, side in (("noLeft", "left"), ("noRight", "right")):
        assert "clearly above the head; otherwise ambiguous" in out[k] and f"no {side} mic" in out[k]
    assert "2 of 4" in out["two"] and "Detection only, no bearing" in out["two"]
    assert "0 of 4" in out["none"] and "No mic connected" in out["none"]


def test_proposal_sends_the_map_and_implies_a_slot_order():
    out = run("""
      caps = ['front', 'left', 'right'];
      window.calibShow(true); await tick(); await tick();
      const useIt = () => { for (const fn of $('calProp').on.click) fn({ target: { id: 'calUse' } }); };
      // mirrored lines (both words read the mic): the words the server picked for the order it was asked about stand
      solveResult = { ok: true, mic_words: { front: null, back: 0, left: 3, right: 2, top: 5 }, confidence: 0.9,
                      notes: [], mirrors: { '1': 0, '5': 4 }, per_direction: {}, slot_order: 'right_first' };
      fire('calSolve', 'click'); await tick(); await tick();
      const asked = solveBody, table = $('calProp').innerHTML;
      useIt();
      const a = $('calSlot').value;
      // a lone mic on DIN2 whose empty slot reads flat: on w4, so MK5 is read first (SEL GND first: left first)
      solveResult = { ok: true, mic_words: { front: null, back: 1, left: 2, right: 3, top: 4 }, confidence: 0.9,
                      notes: [], mirrors: {}, per_direction: {}, slot_order: 'right_first' };
      fire('calSolve', 'click'); await tick(); await tick();
      useIt();
      fire('calSave', 'click'); await tick(); await tick();
      const b = posts[posts.length - 1], rowsB = mics();
      // a front mic: Use this carries it through (MK2 at the front, right first again from MK5 on w5)
      solveResult = { ok: true, mic_words: { front: 0, back: null, left: 3, right: 2, top: 5 }, confidence: 0.9,
                      notes: [], mirrors: {}, per_direction: { front: { order: [0, 3, 2, 5] } }, slot_order: 'left_first' };
      fire('calSolve', 'click'); await tick(); await tick();
      const tableF = $('calProp').innerHTML;
      useIt();
      fire('calSave', 'click'); await tick(); await tick();
      console.log(JSON.stringify({ asked, table, a, b, rowsB, tableF, f: posts[posts.length - 1], rowsF: mics(),
                                   hint: $('calMicHint').textContent }));""")
    assert out["asked"] == {"slot_order": "right_first",
                            "mic_words": {"front": None, "back": 0, "left": 3, "right": 2, "top": 5}}
    assert "<td>Back</td><td>w0 <small>MK2</small>" in out["table"] and "<td>Front</td><td>—</td>" in out["table"]
    assert out["a"] == "right_first"
    b = out["b"]
    assert b["slot_order"] == "left_first" and b["mic_source"] == "auto"
    assert b["mic_words"] == {"front": None, "back": 1, "left": 2, "right": 3, "top": 4}
    rows = out["rowsB"]                                 # left first: MK1 w0 (off), MK2 w1 (back), MK3 .. MK5
    assert [m["on"] for m in rows] == [False, True, True, True, True] and rows[1]["pos"] == "back"
    assert len(rows) == 5
    assert "<td>Front</td><td>w0 <small>MK2</small>" in out["tableF"]
    f = out["f"]
    assert f["slot_order"] == "right_first" and f["mic_words"] == {"front": 0, "back": None, "left": 3, "right": 2,
                                                                   "top": 5}
    assert [m["pos"] for m in out["rowsF"]] == ["", "front", "left", "right", "top"]
    assert "4 of 4" in out["hint"] and "best three" in out["hint"]


def test_front_and_back_share_one_channel():
    out = run("""
      window.calibShow(true); await tick(); await tick();
      const r = {}, msg = () => $('calMsg').textContent;
      r.options = $('calib').innerHTML.match(/<select id="calPos0"[^>]*>(.*?)<[/]select>/s)[1];
      r.sites0 = sites();
      fire('calPos1', 'change', { value: 'front' });             // MK2 moves from back to front: no one else moves
      r.front = { m: mics(), msg: msg(), hint: $('calMicHint').textContent, sites: sites() };
      fire('calOn0', 'change', { checked: true });               // MK1: back counts as taken, the rest are taken
      r.full = { m: mics(), msg: msg() };
      fire('calOn4', 'change', { checked: false }); r.flr = $('calMicHint').textContent;
      fire('calOn0', 'change', { checked: true });               // MK1 now gets top (back is still not free)
      r.mk1Top = mics();
      fire('calPos0', 'change', { value: 'back' });              // back while MK2 has front: MK2 is unticked
      r.back = { m: mics(), msg: msg(), cls: $('calMsg').className, sites: sites() };
      fire('calPos0', 'change', { value: 'front' });             // its own side: nothing else moves, the note goes
      r.mk1Front = { m: mics(), msg: msg() };
      fire('calOn1', 'change', { checked: true });               // MK2 (labelled back): back is taken, so top
      r.mk2 = mics();
      fire('calOn2', 'change', { checked: false }); r.frt = $('calMicHint').textContent;
      fire('calPos3', 'change', { value: 'back' });              // MK4 to back: MK1 (front) unticked
      r.mk4 = { m: mics(), msg: msg() };
      fire('calOn2', 'change', { checked: true });               // MK3 left (its PCB position)
      fire('calOn4', 'change', { checked: true });               // MK5: top and back taken, so right
      fire('calOn3', 'change', { checked: false });              // MK4 (back) off: L R T
      r.lrt = { m: mics(), hint: $('calMicHint').textContent };
      fire('calOn0', 'change', { checked: true });               // MK1: back is free again
      fire('calPos0', 'change', { value: 'front' });             // and to the front
      r.before = mics();
      fire('calSave', 'click'); await tick(); await tick();
      r.saved = posts[posts.length - 1];
      // the device row: channel 0 as the board names it, and whether it is what Apply would send
      const motors = {};
      for (const [m, v] of Object.entries(r.saved.motors)) motors[m] = v ? v.theta_deg : null;
      const w = r.saved.mic_words;
      window.onCalibBench({ kind: 'cal', device: { mics: { F: w.front, L: w.left, R: w.right, T: w.top }, motors, src: 'nvs' } });
      r.devF = $('calDev').innerHTML;
      window.onCalibBench({ kind: 'cal', device: { mics: { B: w.front, L: w.left, R: w.right, T: w.top }, motors, src: 'nvs' } });
      r.devB = $('calDev').innerHTML;
      fire('calMicDefault', 'click');                             // Default: MK2 back, MK3 left, MK4 right, MK5 top
      r.def = { m: mics(), sites: sites() };
      console.log(JSON.stringify(r));""")
    opts = out["options"]
    assert [x for x in ("Front", "Back", "Left", "Right", "Top") if f">{x}</option>" in opts] == \
        ["Front", "Back", "Left", "Right", "Top"]
    assert opts.index('value="front"') < opts.index('value="back"') < opts.index('value="left"')
    s0 = out["sites0"]                                           # the front site is drawn, unused by default
    assert set(s0) == {"FRONT", "BACK", "LEFT", "RIGHT", "TOP"} and s0["FRONT"]["none"] and not s0["BACK"]["none"]
    assert s0["FRONT"]["sub"] == "no mic" and s0["BACK"]["sub"] == "MK2 · B"
    fr = out["front"]
    assert [m["pos"] for m in fr["m"]] == ["", "front", "left", "right", "top"] and fr["msg"] == ""
    assert "4 of 4" in fr["hint"] and "best three" in fr["hint"]
    assert not fr["sites"]["FRONT"]["none"] and fr["sites"]["FRONT"]["sub"] == "MK2 · F" and fr["sites"]["BACK"]["none"]
    assert not out["full"]["m"][0]["on"] and "All four positions are in use" in out["full"]["msg"]
    assert "3 of 4" in out["flr"] and "elevation only when the drone is clearly above" in out["flr"]
    assert [m["pos"] for m in out["mk1Top"]] == ["top", "front", "left", "right", ""]
    bk = out["back"]
    assert [m["pos"] for m in bk["m"]] == ["back", "", "left", "right", ""]
    assert bk["msg"] == "front and back share one channel: MK2 unticked" and "bad" not in bk["cls"]
    assert bk["sites"]["FRONT"]["none"] and bk["sites"]["BACK"]["sub"] == "MK1 · B"
    assert [m["pos"] for m in out["mk1Front"]["m"]] == ["front", "", "left", "right", ""] and out["mk1Front"]["msg"] == ""
    assert [m["pos"] for m in out["mk2"]] == ["front", "top", "left", "right", ""]
    assert "otherwise ambiguous" in out["frt"] and "no left mic" in out["frt"]
    m4 = out["mk4"]
    assert [m["pos"] for m in m4["m"]] == ["", "top", "", "back", ""]
    assert m4["msg"] == "front and back share one channel: MK1 unticked"
    assert [m["pos"] for m in out["lrt"]["m"]] == ["", "top", "left", "", "right"]
    assert "3 of 4" in out["lrt"]["hint"] and "Front and back cannot be told apart" in out["lrt"]["hint"]
    assert [m["pos"] for m in out["before"]] == ["front", "top", "left", "", "right"]
    # right first: MK1 w1, MK2 w0, MK3 w3, MK5 w5
    assert out["saved"]["mic_words"] == {"front": 1, "back": None, "left": 3, "right": 5, "top": 0}
    assert "F=w" in out["devF"] and "in sync" in out["devF"] and "B=" not in out["devF"].split("Motors")[0]
    assert "B=w" in out["devB"] and "differs" in out["devB"]
    d = out["def"]
    assert [m["pos"] for m in d["m"]] == ["", "back", "left", "right", "top"] and d["sites"]["FRONT"]["none"]


def test_motor_connected_checkbox_remembers_the_hole():
    out = run("""
      window.calibShow(true); await tick(); await tick();
      const start = motors();
      fire('calMotOn2', 'change', { checked: false });           // M3 not connected
      const off = motors();
      fire('calMotOn2', 'change', { checked: true });            // back in the hole it had
      fire('calSave', 'click'); await tick(); await tick();
      console.log(JSON.stringify({ start, off, on: motors(), saved: posts[posts.length - 1] }));""")
    assert all(m["on"] for m in out["start"])
    assert not out["off"][2]["on"] and out["off"][2]["hole"] == ""
    assert out["on"][2] == out["start"][2]
    m3 = out["saved"]["motors"]["M3"]
    hole = next(h for h in json.load(open(GEOM))["holes"] if h["id"] == m3["hole"])
    assert m3["theta_deg"] == round(hole["theta_deg"], 1)        # the hole's measured angle, not the nominal 90
