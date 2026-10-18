"""pytest: the bearing status telemetry (bstat, mused) from the board's T lines / the Python pipeline to the page:
TelemetryParser and read_log with the appended columns and empty theta / elev, sources.py's wire frames, the fake
device's columns (channel 0 at the back or the front), the levels map and slot labels, and app.js's status labels
and channel-0 letter (F when the headset reports its front mic there; run under node)."""
import json, os, re, shutil, subprocess, sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from webapp import sources, fake_device

telemetry = sources.telemetry
NODE = shutil.which("node")
MOTORS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
HEADER = ("H,frame,t_ms,f0,score,strength,flat,hits,ml,detected,theta,conf,elev,overhead,batt_v,"
          + ",".join("hap_" + m for m in MOTORS))


def t_line(frame, theta, elev, bstat=None, mused=None, detected=1):
    hap = ",".join(["0.50", "0.50"] + ["0.00"] * 6)
    tail = "" if bstat is None else f",{bstat},{mused}"
    return f"T,{frame},{1000 + frame},250.0,8.10,9.00,0.410,80,0.910,{detected},{theta},0.800,{elev},0,3.95,{hap}{tail}"


def rows(text):
    p = telemetry.TelemetryParser()
    out = [x for k, x in p.feed(text.encode()) if k == "row"]
    return out, p.bad


def test_t_line_row_to_wire_frame():
    rs, bad = rows("\n".join([HEADER + ",bstat,mused",
                              t_line(1, "30.0", "40.0", 3, 15),        # four mics: 3D
                              t_line(2, "-75.5", "", 2, 7),            # B L R: bearing only, no elevation
                              t_line(3, "", "", 1, 14),               # L R T: ambiguous, no theta
                              t_line(4, "", "", 0, 0, detected=0)]) + "\n")
    assert bad == 0 and len(rs) == 4
    f = [sources.device_row_to_frame(r) for r in rs]
    assert (f[0]["theta"], f[0]["elev"], f[0]["bstat"], f[0]["mused"]) == (30.0, 40.0, 3, 15)
    assert (f[1]["theta"], f[1]["elev"], f[1]["bstat"], f[1]["mused"]) == (-75.5, None, 2, 7)
    assert (f[2]["theta"], f[2]["elev"], f[2]["bstat"], f[2]["mused"]) == (None, None, 1, 14)
    assert f[3]["bstat"] == 0 and f[3]["mused"] == 0 and f[3]["theta"] is None
    assert isinstance(f[0]["bstat"], int) and isinstance(f[0]["mused"], int)
    assert json.loads(json.dumps(f[2]))["theta"] is None                     # what the WebSocket carries


def test_older_firmware_rows_have_no_status():
    rs, bad = rows(HEADER + "\n" + t_line(1, "12.0", "5.0") + "\n")
    assert bad == 0
    f = sources.device_row_to_frame(rs[0])
    assert f["bstat"] is None and f["mused"] is None and f["theta"] == 12.0


def test_empty_fields_only_where_allowed():
    rs, bad = rows("\n".join([HEADER + ",bstat,mused",
                              t_line(1, "", "", 0, 0).replace(",250.0,", ",,"),    # an empty f0: malformed
                              t_line(2, "1.0", "2.0", 3, 15)[:-3],                 # torn (one column short)
                              t_line(3, "", "", 0, 0)]) + "\n")
    assert bad == 2 and [r["frame"] for r in rs] == [3.0]


def test_log_with_new_columns_replays(tmp_path):
    p = tmp_path / "run.csv"
    rs, _ = rows("\n".join([HEADER + ",bstat,mused", t_line(1, "30.0", "40.0", 3, 15), t_line(2, "", "", 1, 14)]) + "\n")
    cols = ["host_s"] + HEADER.split(",")[1:] + ["bstat", "mused"]
    p.write_text(",".join(cols) + "\n" + "\n".join(",".join(["0.5"] + r["_raw"]) for r in rs) + "\n")
    got, bad = telemetry.read_log(str(p))
    assert bad == 0 and len(got) == 2
    f = [sources.device_row_to_frame(r, "replay") for r in got]
    assert (f[0]["bstat"], f[0]["mused"], f[1]["theta"], f[1]["bstat"]) == (3, 15, None, 1)


def _result(**kw):
    r = dict(f0=250.0, score=8.0, strength=9.0, flat=0.4, hits=80, ml=0.9, detected=True, theta=20.0, conf=0.7,
             elev=35.0, overhead=False, haptics={"F": 0.6, "FR": 0.4})
    r.update(kw)
    return r


def test_pipeline_dict_to_wire_frame():
    new = sources.pipeline_to_frames([_result(bstat=2, mused=13, status="constrained_3d"),
                                      _result(theta=None, elev=None, bstat=1, mused=14, status="ambiguous")], "demo")
    assert (new[0]["bstat"], new[0]["mused"], new[0]["theta"], new[0]["elev"]) == (2, 13, 20.0, 35.0)
    assert (new[1]["bstat"], new[1]["mused"], new[1]["theta"]) == (1, 14, None)
    old = sources.pipeline_to_frames([_result()], "demo")[0]                 # a pipeline without the new keys
    assert old["bstat"] is None and old["mused"] is None and old["theta"] == 20.0
    assert sources.wire_frame("x", 0, 0, None, None, None, None, 0, None, 0, None, None, 0, [0] * 8)["bstat"] is None


def test_fake_device_columns():
    f = dict(_result(), detected=1, hap=[0.5, 0.5] + [0.0] * 6)
    assert fake_device.HEADER.rstrip("\n").endswith(",hap_FL,bstat,mused")
    lines = fake_device.HEADER
    for n, conn in ((1, "BLRT"), (2, "BLR"), (3, "LRT"), (4, "BRT"), (5, "BL")):
        lines += fake_device.t_line(n, 1000 + n, f, 3.9, conn)
    lines += fake_device.t_line(6, 2000, dict(f, elev=10.0), 3.9, "BLT")             # low: ambiguous with B L T
    lines += fake_device.t_line(7, 2000, dict(f, detected=0), 3.9, "BLRT")           # room noise
    rs, bad = rows(lines)
    assert bad == 0
    got = {int(r["frame"]): sources.device_row_to_frame(r) for r in rs}
    st = {n: (g["bstat"], g["mused"], g["theta"] is not None, g["elev"] is not None) for n, g in got.items()}
    assert st[1] == (3, 15, True, True)
    assert st[2] == (2, 7, True, True)                    # B L R, clearly above (35 deg): elevation assumed above
    assert st[3] == (1, 14, False, False)                 # L R T: front and back cannot be told apart
    assert st[4] == (2, 13, True, True)                   # B R T, clearly above
    assert st[5] == (0, 0, False, False)                  # two mics: detection only
    assert st[6] == (1, 11, False, False)
    assert st[7] == (0, 0, False, False)
    low = fake_device.bearing_fields(dict(f, elev=10.0), "BLR", 1)
    assert low == (20.0, None, 2, 7)                       # B L R, low: bearing only
    drop = [fake_device.bearing_fields(f, "BLRT", n)[3] for n in range(7 * 47, 8 * 47)]
    assert set(drop) == {14}                               # four mics, B out of the fit for a second
    passthrough = fake_device.bearing_fields(dict(f, bstat=2, mused=11), "BLRT", 1)
    assert passthrough[2:] == (2, 11)                      # a pipeline's own bstat / mused win with four mics
    # a front mic in channel 0 does exactly what the back one does (mused bit 1 either way)
    for conn in ("FLRT", "FLR", "LRT", "FRT", "FL", "FLT"):
        assert fake_device.bearing_fields(f, conn, 1) == fake_device.bearing_fields(f, conn.replace("F", "B"), 1)
    assert fake_device.bearing_fields(f, "FLRT", 1)[2:] == (3, 15)
    assert fake_device.bearing_fields(dict(f, elev=10.0), "FLR", 1) == (20.0, None, 2, 7)
    assert fake_device.bearing_fields(f, "FRT", 1)[2:] == (2, 13)
    assert {fake_device.bearing_fields(f, "FLRT", n)[3] for n in range(7 * 47, 8 * 47)} == {14}
    cal = fake_device.FakeCal()
    cal.set_mics("1,3,2,5,F")
    rs, bad = rows(fake_device.HEADER + fake_device.t_line(9, 1009, f, 3.9, cal.connected()))
    assert bad == 0 and sources.device_row_to_frame(rs[0])["mused"] == 15


def test_levels_map_front_or_back_and_slot_labels():
    msgs = []
    src = sources.SerialSource("/nonexistent", lambda f: None, lambda **kw: None, message=msgs.append)
    words = " ".join(f"w{w} rms=-60.0 peak=-50.0 dc=+0.001" for w in range(6))
    src._comment(f"# levels {words} | B=w0 L=w3 R=w2 T=w5")
    assert msgs[-1]["kind"] == "levels" and msgs[-1]["map"] == {"B": 0, "L": 3, "R": 2, "T": 5}
    src._comment(f"# levels {words} | F=w1 L=w3 R=w2 T=w5")              # channel 0 is the front mic
    assert msgs[-1]["map"] == {"F": 1, "L": 3, "R": 2, "T": 5}
    # the words' physical slots under the board's slot order (right first: SEL 3V3 is each line's first word)
    assert [w["slot"] for w in msgs[-1]["words"]] == ["DIN0 SEL 3V3 (MK2)", "DIN0 SEL GND (MK1)", "DIN1 SEL 3V3 (MK4)",
                                                       "DIN1 SEL GND (MK3)", "DIN2 SEL 3V3 (no header)",
                                                       "DIN2 SEL GND (MK5)"]
    cal = fake_device.FakeCal()
    cal.set_mics("1,3,2,5,F")
    src._comment(fake_device.levels_line({"front": 1, "left": 3, "right": 2, "top": 5}, cal).strip())
    assert msgs[-1]["map"] == {"F": 1, "L": 3, "R": 2, "T": 5} and len(msgs[-1]["words"]) == 6
    src._comment("# cal mics F=w1 L=w3 R=w2 T=w5 motors M1=0.0 src=nvs")
    assert src.cal_device["mics"] == {"F": 1, "L": 3, "R": 2, "T": 5} and msgs[-1]["device"] == src.cal_device
    src._comment("# cal mics B=w0 L=w3 R=w2 T=w5 saved")                  # back again: no stale F key
    assert src.cal_device["mics"] == {"B": 0, "L": 3, "R": 2, "T": 5}


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_app_js_status_labels():
    src = open(os.path.join(ROOT, "webapp", "static", "app.js"), encoding="utf-8").read()
    m = re.search(r"// -{10,} bearing status\n.*?\nfunction setHtml", src, re.S)
    assert m, "bearing status section not found in app.js"
    code = m.group(0)[: -len("function setHtml")]
    script = "let status = {}, source = 'serial';\n" + code + r"""
const fr = (o) => Object.assign({ theta: 10, elev: 30, detected: 1 }, o);
const out = {};
status = { cal_device: { mics: { B: 1, L: 2, R: 3, T: 4 } } };
out.full = bearingStatus(fr({ bstat: 3, mused: 15 }));
out.azOnly = bearingStatus(fr({ bstat: 2, mused: 7, elev: null }));
out.above = bearingStatus(fr({ bstat: 2, mused: 13 }));
out.ambig = bearingStatus(fr({ bstat: 1, mused: 14, theta: null, elev: null }));
out.none = bearingStatus(fr({ bstat: 0, mused: 0, theta: null, elev: null }));
out.held = bearingStatus(fr({ bstat: 0, mused: 0, elev: null }));     // the pipeline keeps its smoothed theta
out.legacy3d = bearingStatus(fr({}));
out.legacyNone = bearingStatus(fr({ theta: null, elev: null }));
status = { cal_device: { mics: { B: 1, L: null, R: -1, T: 4 } } };
out.twoMics = bearingStatus(fr({ bstat: 0, mused: 0, theta: null, elev: null }));
status = { cal_device: { mics: { F: 1, L: 3, R: 2, T: 5 } } };           // channel 0 is the front mic
out.front = { ch0: ch0Letter(), b: micLabel('B'), l: micLabel('L'), n: micsConnected(),
              s: bearingStatus(fr({ bstat: 3, mused: 15 })) };
status = { cal_device: { mics: { F: null, L: 3, R: 2, T: 5 } } };
out.frontOff = { ch0: ch0Letter(), n: micsConnected() };
status = { cal_device: { mics: { B: 0, L: 3, R: 2, T: 5 } } };
out.back = { ch0: ch0Letter(), b: micLabel('B'), n: micsConnected() };
status = {};
out.noCal = ch0Letter();
source = 'demo'; status = { channels: 1, cal_device: { mics: { F: 1, L: 3, R: 2, T: 5 } } };
out.demoF = micLabel('B');                                                  // simulated / replayed input: always B
source = 'replay'; out.replayF = micLabel('B');
source = 'demo'; status = { channels: 1 };
out.mono = bearingStatus(fr({ theta: null, elev: null }));
out.chip = bstChip(out.ambig, 'idle');
console.log(JSON.stringify(out));
"""
    r = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    o = json.loads(r.stdout)
    assert (o["full"]["code"], o["full"]["label"]) == (3, "3D") and "four mics" in o["full"]["title"]
    assert (o["azOnly"]["code"], o["azOnly"]["label"]) == (2, "bearing only")
    assert (o["above"]["code"], o["above"]["label"]) == (2, "assumed above") and "three mics" in o["above"]["title"]
    assert (o["ambig"]["code"], o["ambig"]["label"]) == (1, "ambiguous")
    assert (o["none"]["code"], o["none"]["label"]) == (0, "no bearing") and "disagree" in o["none"]["title"]
    assert (o["held"]["code"], o["held"]["label"]) == (0, "held")
    assert o["legacy3d"]["label"] == "3D" and o["legacyNone"]["label"] == "no bearing"
    assert "two mics connected" in o["twoMics"]["title"]
    assert "mono" in o["mono"]["title"]
    assert 'class="bst s1 idle"' in o["chip"] and ">ambiguous<" in o["chip"]
    assert o["front"]["ch0"] == "F" and o["front"]["b"] == "F" and o["front"]["l"] == "L" and o["front"]["n"] == 4
    assert o["front"]["s"]["label"] == "3D"
    assert o["frontOff"] == {"ch0": "F", "n": 3}
    assert o["back"] == {"ch0": "B", "b": "B", "n": 4} and o["noCal"] == "B"
    assert o["demoF"] == "B" and o["replayF"] == "B"
    # the radar's indicators and head drawing use the letter (rMicB's text follows micLabel)
    assert 'micLabel(m)' in src and "el.textContent = lab" in src and "micLabel(CFG.mic_names[i])" in src
