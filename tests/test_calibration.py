"""pytest: headset calibration (webapp/calibration.py), its endpoints and the fake device's C / Y / K / L / W."""
import json, os, socket, subprocess, sys, time
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from webapp import calibration as cal

# The v3 board reads the SEL 3V3 phase first (right_first, the default): MK1..MK5 on w1, w0, w3, w2, w5 (w4 has no
# header); left_first: w0..w4 (w5 has none).
RIGHT_FIRST_PCB = {"back": 0, "left": 3, "right": 2, "top": 5}       # MK2..MK5 at their labelled positions
LEFT_FIRST_PCB = {"back": 1, "left": 2, "right": 3, "top": 4}
# MK2..MK5 populated but the mics moved round the head; one mic on DIN0 and on DIN2, so their empty slots mirror them
PERMUTED = {"back": 3, "left": 4, "right": 1, "top": 2}               # left first: w0 copies w1, w5 copies w4
PERMUTED_RIGHT_FIRST = {"back": 2, "left": 5, "right": 0, "top": 3}   # right first: MK2 w0, MK3 w3, MK4 w2, MK5 w5
# a front mic instead of the back one, on the same headers (MK2 front, MK3 left, MK4 right, MK5 top)
FRONT_LEFT_FIRST = {"front": 1, "left": 2, "right": 3, "top": 4}
FRONT_RIGHT_FIRST = {"front": 0, "left": 3, "right": 2, "top": 5}
FRONT_PERMUTED = {"front": 2, "left": 5, "right": 0, "top": 3}         # right first


def full(wiring):
    return {p: wiring.get(p) for p in cal.POSITIONS}


def captures(wiring, dirs=("front", "right", "left"), **kw):
    return {d: cal.synth_capture(wiring, d, seconds=3.0, seed=i, **kw) for i, d in enumerate(dirs)}


def solve(caps, slot_order="right_first", current=None):
    """The solver as the tab calls it: with the slot order and the map in use (default: the PCB map, MK2..MK5)."""
    pcb = RIGHT_FIRST_PCB if slot_order == "right_first" else LEFT_FIRST_PCB
    return cal.solve(caps, slot_order, dict(pcb) if current is None else current)


def test_default_calibration_is_the_pcb_map_right_first():
    c = cal.default_calibration(cal.placeholder_geometry())
    assert c["slot_order"] == "right_first" == cal.DEFAULT_SLOT_ORDER
    assert c["mic_words"] == full(RIGHT_FIRST_PCB) and cal.PCB_WORDS == RIGHT_FIRST_PCB
    assert [c["motors"][m]["theta_deg"] for m in cal.MOTORS] == cal.MOTOR_DEFAULT_THETA
    assert cal.firmware_args(c) == ("0,3,2,5", "0,45,90,135,180,-135,-90,-45")
    assert cal.header_words("left_first") == {"MK1": 0, "MK2": 1, "MK3": 2, "MK4": 3, "MK5": 4}
    assert cal.header_words("right_first") == {"MK1": 1, "MK2": 0, "MK3": 3, "MK4": 2, "MK5": 5}
    assert cal.header_words() == cal.header_words("right_first")
    sites = {s["position"] for s in cal.placeholder_geometry()["mic_sites"]}
    assert sites == set(cal.POSITIONS)


def test_validation():
    g = cal.placeholder_geometry()
    good = cal.default_calibration(g)
    with pytest.raises(ValueError, match="used by both"):
        cal.validate(dict(good, mic_words={"left": 1, "back": 1}), g)
    with pytest.raises(ValueError, match="not a word"):
        cal.validate(dict(good, mic_words={"back": 6}), g)
    with pytest.raises(ValueError, match="unknown position"):
        cal.validate(dict(good, mic_words={"nose": 0}), g)
    bad_angle = dict(good, motors=dict(good["motors"], M2={"hole": None, "theta_deg": 200}))
    with pytest.raises(ValueError, match="outside"):
        cal.validate(bad_angle, g)
    with pytest.raises(ValueError, match="no hole"):
        cal.validate(dict(good, motors=dict(good["motors"], M1={"hole": "H99"})), g)
    v = cal.validate(dict(good, mic_words={"top": 4}, motors={"M3": {"hole": "H3"}}, mic_source="manual"), g)
    assert v["mic_words"]["back"] is None and v["mic_words"]["front"] is None and v["motors"]["M1"] is None
    assert v["motors"]["M3"]["theta_deg"] == 90.0                         # takes the hole's angle
    c, y = cal.firmware_args(v)
    assert c == "-1,-1,-1,4" and y.split(",")[2] == "90" and y.split(",")[0] == "999"
    # the slot order: right_first unless the file names one (left_first can still be chosen)
    assert good["slot_order"] == "right_first" and v["slot_order"] == "right_first"
    assert cal.validate(dict(good, slot_order="left_first"), g)["slot_order"] == "left_first"
    assert cal.validate({k: x for k, x in good.items() if k != "slot_order"}, g)["slot_order"] == "right_first"
    with pytest.raises(ValueError, match="slot_order"):
        cal.validate(dict(good, slot_order="sideways"), g)


def test_front_and_back_share_channel_0(tmp_path):
    g = cal.placeholder_geometry()
    good = cal.default_calibration(g)
    with pytest.raises(ValueError, match="front and back share one channel: choose one"):
        cal.validate(dict(good, mic_words={"front": 1, "back": 0, "left": 3}), g)
    front = cal.validate(dict(good, mic_words=FRONT_RIGHT_FIRST), g)
    assert front["mic_words"] == full(FRONT_RIGHT_FIRST)
    assert cal.channel0(front["mic_words"]) == ("F", 0)
    assert cal.firmware_args(front)[0] == "0,3,2,5,F"                   # C<front>,<l>,<r>,<t>,F
    assert cal.firmware_args(dict(front, mic_words=full({"front": 1, "right": 2})))[0] == "1,-1,2,-1,F"
    assert cal.firmware_args(good)[0] == "0,3,2,5"                       # back: no suffix
    neither = cal.validate(dict(good, mic_words={"left": 3, "right": 2, "top": 5}), g)
    assert cal.channel0(neither["mic_words"]) == ("B", None) and cal.firmware_args(neither)[0] == "-1,3,2,5"
    p = tmp_path / "cal.json"                                            # a saved front map loads as it was
    saved = cal.save_calibration(dict(good, mic_words=FRONT_PERMUTED, mic_source="auto"), str(p))
    back, err = cal.load_calibration(str(p))
    assert err is None and back["mic_words"] == full(FRONT_PERMUTED) == saved["mic_words"]
    p.write_text(json.dumps(dict(good, mic_words={"front": 1, "back": 0})))   # both: unusable, the defaults
    d, err = cal.load_calibration(str(p))
    assert "front and back share one channel" in err and d["mic_words"] == full(RIGHT_FIRST_PCB)


def test_save_and_load_roundtrip(tmp_path):
    p = str(tmp_path / "cal.json")
    c = cal.default_calibration()
    c["mic_words"] = dict(PERMUTED); c["mic_source"] = "auto"; c["slot_order"] = "left_first"
    saved = cal.save_calibration(c, p)
    assert saved["updated"]
    back, err = cal.load_calibration(p)
    assert err is None and back["mic_words"] == full(PERMUTED) and back["mic_source"] == "auto"
    assert back["slot_order"] == "left_first"                           # a file that names its slot order keeps it
    open(p, "w").write("{not json")
    d, err = cal.load_calibration(p)
    assert err and d["mic_source"] == "default" and d["slot_order"] == "right_first"


def test_parse_cal_lines():
    d = cal.parse_cal_line("cal mics B=w0 L=- R=w2 T=w5 motors M1=0.0 M2=45.0 M3=- M4=135.0 M5=180.0 M6=-135.0 "
                           "M7=-90.0 M8=-45.0 src=nvs")
    assert d["mics"] == {"B": 0, "L": None, "R": 2, "T": 5} and d["motors"]["M3"] is None
    assert d["motors"]["M6"] == -135.0 and d["src"] == "nvs" and not d.get("saved")
    assert cal.parse_cal_line("cal mics B=w0 L=w3 R=w2 T=w5 saved") == {"mics": {"B": 0, "L": 3, "R": 2, "T": 5},
                                                                         "saved": True}
    # channel 0 at the front: the board prints F= instead of B=
    assert cal.parse_cal_line("cal mics F=w1 L=w3 R=w2 T=w5 saved") == {"mics": {"F": 1, "L": 3, "R": 2, "T": 5},
                                                                         "saved": True}
    assert cal.parse_cal_line("cal mics F=- L=w3 R=w2 T=w5 src=nvs")["mics"] == {"F": None, "L": 3, "R": 2, "T": 5}
    assert "error" in cal.parse_cal_line("cal: bad 'C9', want C<w0>,<l>,<r>,<t>[,F] with words 0-5 or -1")
    assert "mics" not in cal.parse_cal_line("cal mics B=w1 L=w2 motors M1=0.0 src=nvs")      # a torn line
    assert "mics" not in cal.parse_cal_line("cal mics L=w2 R=w3 T=w4 src=nvs")              # no channel 0
    assert "mics" not in cal.parse_cal_line("cal mics B=- L=w2 R=w3 T=w4 F=w0 src=nvs")     # both: not this board


def test_device_matches_checks_channel_0_side():
    g = cal.placeholder_geometry()
    base = cal.default_calibration(g)
    motors = {m: base["motors"][m]["theta_deg"] for m in cal.MOTORS}
    dev = lambda line: dict(cal.parse_cal_line(line), motors=motors)
    front = dict(base, mic_words=full(FRONT_RIGHT_FIRST))
    assert cal.device_matches(dev("cal mics B=w0 L=w3 R=w2 T=w5"), base)
    assert not cal.device_matches(dev("cal mics F=w0 L=w3 R=w2 T=w5"), base)       # same word, other side
    assert cal.device_matches(dev("cal mics F=w0 L=w3 R=w2 T=w5"), front)
    assert not cal.device_matches(dev("cal mics B=w0 L=w3 R=w2 T=w5"), front)
    assert not cal.device_matches(dev("cal mics F=w1 L=w3 R=w2 T=w5"), front)      # other word
    assert not cal.device_matches(dev("cal mics F=w0 L=w3 R=w2 T=w4"), front)
    neither = dict(base, mic_words=full({"left": 3, "right": 2, "top": 5}))         # sent as C-1,3,2,5 (back form)
    assert cal.device_matches(dev("cal mics B=- L=w3 R=w2 T=w5"), neither)
    assert not cal.device_matches(dev("cal mics F=- L=w3 R=w2 T=w5"), neither)
    want = dict(base, mic_words=full({"left": 2, "right": 3, "top": 4}),
                motors={m: (None if m != "M3" else {"hole": None, "theta_deg": 82.9}) for m in cal.MOTORS})
    d = cal.parse_cal_line("cal mics B=- L=w2 R=w3 T=w4 motors M1=- M3=82.9 src=nvs")
    assert cal.device_matches(dict(d, motors={m: (None if m != "M3" else 82.9) for m in cal.MOTORS}), want)
    other = dict(d, mics=dict(d["mics"], T=5), motors={m: (None if m != "M3" else 82.9) for m in cal.MOTORS})
    assert not cal.device_matches(other, want)


def test_solver_recovers_a_permuted_wiring_left_first():
    r = solve(captures(PERMUTED), "left_first")
    assert r["mic_words"] == full(PERMUTED)
    assert r["confidence"] >= 0.5 and r["slot_order"] == "left_first"
    assert r["mirrors"] == {"1": 0, "5": 4}
    o = r["per_direction"]["front"]["order"]
    assert o[-1] == 3                                   # the back mic (on w3) hears the front clap last
    assert r["per_direction"]["right"]["order"][0] == 0 and r["per_direction"]["left"]["order"][0] == 4


def test_solver_right_first_slot_order():
    r = solve(captures(PERMUTED_RIGHT_FIRST))
    assert r["mic_words"] == full(PERMUTED_RIGHT_FIRST) and r["slot_order"] == "right_first"
    assert cal.solve(captures(PERMUTED_RIGHT_FIRST))["slot_order"] == "right_first"     # the default order


@pytest.mark.parametrize("order,wiring", [("left_first", FRONT_LEFT_FIRST), ("right_first", FRONT_RIGHT_FIRST),
                                          ("right_first", FRONT_PERMUTED)])
@pytest.mark.parametrize("dirs", [("front", "right", "left"), ("front", "right", "left", "back", "top"),
                                  ("front", "right", "left", "back")])
def test_solver_recovers_a_front_mic(order, wiring, dirs):
    r = solve(captures(wiring, dirs=dirs), order)
    assert r["mic_words"] == full(wiring), r["notes"]
    assert r["confidence"] >= 0.6 and not any("share one channel" in n for n in r["notes"])
    first = r["per_direction"]["front"]["order"][0]
    assert first in (wiring["front"], wiring["front"] ^ 1)  # the front mic (or its line's mirror) hears it first


@pytest.mark.parametrize("dirs", [("front", "right", "left"), ("front", "right", "left", "back")])
def test_solver_still_finds_a_back_mic(dirs):
    r = solve(captures(RIGHT_FIRST_PCB, dirs=dirs))
    assert r["mic_words"] == full(RIGHT_FIRST_PCB), r["notes"]
    assert not any("front / back" in n or "share one channel" in n for n in r["notes"])


def test_solver_front_and_back_both_clear_keeps_the_clearer():
    # a back-mic headset, but the "front" clap came from 35 deg up: the top mic leads the side ones on it, so it looks
    # like a front mic while the back mic is clearly last; only one fits channel 0, and the back lead is clearer
    w = RIGHT_FIRST_PCB
    caps = {"front": cal.synth_capture(w, (0.0, 35.0), seed=1), "right": cal.synth_capture(w, "right", seed=2),
            "left": cal.synth_capture(w, "left", seed=3)}
    r = solve(caps)
    assert r["mic_words"] == full(w), r["notes"]
    assert any("share one channel" in n and "kept back" in n for n in r["notes"])


def test_solver_with_a_top_capture_and_all_directions():
    r = solve(captures(PERMUTED, dirs=("front", "right", "left", "back", "top")), "left_first")
    assert r["mic_words"] == full(PERMUTED) and r["confidence"] >= 0.6
    assert not any("left over" in n for n in r["notes"])           # top came from the top clap


@pytest.mark.parametrize("wiring", [
    {"back": 0, "left": 3, "right": 2},                 # no top: azimuth only
    {"front": 0, "left": 3, "right": 2},
    {"left": 3, "right": 2, "top": 5},                  # no front or back: the front clap has no clear first or last
    {"back": 0, "right": 2, "top": 5},                  # no left: the left clap has no clear first mic
    {"front": 0, "right": 2, "top": 5},
    {"back": 0, "left": 3, "top": 5},                   # no right
    {"front": 0, "left": 3, "top": 5},
])
def test_solver_three_mics_leaves_the_missing_one_empty(wiring):
    r = solve(captures(wiring))
    assert r["mic_words"] == full(wiring), r["notes"]
    if "front" not in wiring and "back" not in wiring:
        assert any(n.startswith("front / back: no mic clearly") for n in r["notes"])


def test_solver_ignores_a_noisy_unconnected_word():
    w = {"back": 0, "left": 3, "right": 2}                                 # DIN2 empty, w4 picks up crosstalk
    r = solve(captures(w, crosstalk_words=(4,)))
    assert r["mic_words"] == full(w)
    assert 4 not in r["live_words"] and any("w4" in n for n in r["notes"])


def test_solver_one_mic_and_its_mirror():
    r = solve(captures({"back": 0}, dirs=("front", "right", "left", "back")))       # MK2 only, w1 copies w0
    assert r["mic_words"] == full({"back": 0})
    assert r["mirrors"] == {"1": 0} and r["live_words"] == [0]
    assert r["confidence"] <= 0.5 and any("only one mic" in n for n in r["notes"])
    r = solve(captures({"front": 0}, dirs=("front", "right", "left", "back")))      # ... at the front
    assert r["mic_words"] == full({"front": 0})


def test_solver_keeps_the_connected_header_of_a_mirrored_line():
    # only MK1 (w1, right first) fitted; w0 (MK2) mirrors it, so either word carries the mic
    caps = captures({"back": 1}, dirs=("front", "right", "left", "back"))
    mk1 = full({"back": 1})
    assert solve(caps, current=mk1)["mic_words"] == mk1                         # MK1 is ticked: it stays MK1
    assert solve(caps)["mic_words"]["back"] == 0                               # not ticked: MK2, labelled back
    w = {"back": 1, "left": 3, "right": 2, "top": 5}                           # MK1 back, MK3..MK5 as labelled
    assert solve(captures(w), current=w)["mic_words"] == full(w)
    assert solve(captures(w), current=dict(w, back=None))["mic_words"] == full(dict(w, back=0))
    wf = {"front": 1, "left": 3, "right": 2, "top": 5}                         # MK1 at the front
    assert solve(captures(wf), current=wf)["mic_words"] == full(wf)


def test_solver_needs_front_right_left():
    with pytest.raises(ValueError, match="right"):
        cal.solve({"front": cal.synth_capture(PERMUTED, "front")})


def test_parse_wiring_and_synth_take_front_or_back():
    assert cal.parse_wiring("back=0, left=3,right=2,top=5") == RIGHT_FIRST_PCB
    assert cal.parse_wiring("front=0,left=3,right=2,top=5") == FRONT_RIGHT_FIRST
    assert cal.MIC_SITES_M["front"] == (0.0, 0.10, 0.0)
    with pytest.raises(ValueError, match="front and back share one channel"):
        cal.parse_wiring("front=0,back=1")
    with pytest.raises(ValueError, match="front and back share one channel"):
        cal.synth_capture({"front": 0, "back": 1}, "front")
    with pytest.raises(ValueError, match="unknown position"):
        cal.parse_wiring("nose=0")


def test_capture_summary_flags():
    s = cal.capture_summary(cal.synth_capture({"back": 1}, "back"))
    w = {d["w"]: d for d in s["words"]}
    assert s["event"]["found"] and 0.9 < s["event"]["t_s"] < 1.2
    assert w[0]["live"] and w[0]["responds"] and w[1]["mirror_of"] == 0
    assert not w[2]["live"] and abs(w[2]["dc"] - cal.FLAT_WORD) < 1e-3
    quiet = cal.capture_summary(cal.synth_capture({"back": 1}, None))
    assert not quiet["event"]["found"]


def test_fake_device_cal_commands():
    from webapp.fake_device import FakeCal, levels_line
    c = FakeCal()
    assert c.report().startswith("# cal mics B=w0 L=w3 R=w2 T=w5 motors M1=0.0") and "F=" not in c.mics_text()
    assert c.set_mics("3,4,1,2") == "# cal mics B=w3 L=w4 R=w1 T=w2 saved\n" and c.connected() == "BLRT"
    assert c.set_mics("-1,2,3,4") == "# cal mics B=- L=w2 R=w3 T=w4 saved\n" and c.connected() == "LRT"
    assert c.set_mics("1,1,3,4").startswith("# cal: bad")
    assert c.set_mics("1,2,3,4,0").startswith("# cal: bad")          # a fifth value must be F or B
    assert c.set_mics("1,3,2,5,F") == "# cal mics F=w1 L=w3 R=w2 T=w5 saved\n" and c.connected() == "FLRT"
    d = cal.parse_cal_line(c.report()[2:])
    assert d["mics"] == {"F": 1, "L": 3, "R": 2, "T": 5}
    assert c.set_mics("-1,3,2,5,F") == "# cal mics F=- L=w3 R=w2 T=w5 saved\n" and c.connected() == "LRT"
    assert c.set_mics("0,3,2,5,B") == "# cal mics B=w0 L=w3 R=w2 T=w5 saved\n" and c.connected() == "BLRT"
    c.set_mics("1,3,2,5,F")
    from webapp.sources import telemetry
    words, chmap, _ = telemetry.levels_parse(levels_line(FRONT_RIGHT_FIRST, c, seed=3))
    assert chmap == {"F": 1, "L": 3, "R": 2, "T": 5} and sorted(words) == [0, 1, 2, 3, 4, 5]
    words, _, _ = telemetry.levels_parse(levels_line({"back": 0, "left": 3, "right": 2}, c))
    assert words[4][2] < -0.9 and words[5][2] < -0.9 and words[0][1] > -80        # DIN2 empty: flat


# ---------------------------------------------------------------- end to end: fake device -> dashboard API
def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


@pytest.mark.skipif(os.environ.get("HH_FAST_TESTS") == "1", reason="HH_FAST_TESTS=1: skip the ~25 s end-to-end test")
def test_end_to_end_auto_calibration_front_mic(tmp_path):
    # a headset with a front mic instead of the back one, the headers shuffled; the board's slot order (right first)
    aiohttp = pytest.importorskip("aiohttp")
    import asyncio
    tcp, http = _free_port(), _free_port()
    env = dict(os.environ, PYTHONWARNINGS="ignore", TF_CPP_MIN_LOG_LEVEL="3",
               HH_CALIBRATION=str(tmp_path / "calibration.json"), HH_CAL_CAPTURES=str(tmp_path / "captures"))
    wiring = ",".join(f"{k}={v}" for k, v in FRONT_PERMUTED.items())
    py = sys.executable
    fake = subprocess.Popen([py, "-m", "webapp.fake_device", "--tcp", str(tcp), "--no-ml", "--wiring", wiring,
                             "--cal-script", "front,right,left"], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    dash = None
    try:
        t0 = time.time()                                    # the fake device builds its demo scene first (~5 s)
        while time.time() - t0 < 30:
            try:
                socket.create_connection(("127.0.0.1", tcp), timeout=0.5).close()
                break
            except OSError:
                time.sleep(0.3)
        dash = subprocess.Popen([py, "scripts/dashboard.py", "--tcp", f"127.0.0.1:{tcp}", "--no-browser",
                                 "--http-port", str(http), "--no-ml"], cwd=ROOT, env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        base = f"http://127.0.0.1:{http}"

        async def run():
            async with aiohttp.ClientSession() as s:
                for _ in range(60):
                    try:
                        async with s.get(base + "/api/calibration") as r:
                            j = await r.json()
                            if j["device"]:
                                break
                    except Exception:
                        pass
                    await asyncio.sleep(0.3)
                assert j["device"]["mics"] == {"B": 0, "L": 3, "R": 2, "T": 5}   # the 'K' reply on connect
                assert j["calibration"]["slot_order"] == "right_first"
                async with s.get(base + "/api/geometry") as r:
                    assert len((await r.json())["holes"]) >= 1
                async with s.get(base + "/api/config") as r:
                    assert (await r.json())["front_pos"] == [0.0, 0.1, 0.0]
                for d in ("front", "right", "left"):
                    async with s.post(base + "/api/calibration/capture", json={"direction": d, "seconds": 2}) as r:
                        c = await r.json()
                        assert r.status == 200 and c["ok"], c
                        assert c["event"]["found"], c
                        assert len(c["words"]) == 6
                async with s.post(base + "/api/calibration/solve") as r:
                    sol = await r.json()
                assert sol["mic_words"] == full(FRONT_PERMUTED) and sol["slot_order"] == "right_first", sol
                body = dict(j["calibration"], mic_words=sol["mic_words"], mic_source="auto")
                async with s.post(base + "/api/calibration", json=body) as r:
                    assert (await r.json())["ok"]
                async with s.post(base + "/api/calibration/apply") as r:
                    ap = await r.json()
                assert ap["ok"] and ap["device_matches"], ap
                assert ap["device"]["mics"] == {"F": 2, "L": 5, "R": 0, "T": 3}
                assert ap["sent"]["C"] == "2,5,0,3,F"
                async with s.get(base + "/api/status") as r:
                    assert (await r.json())["cal_device"]["mics"] == {"F": 2, "L": 5, "R": 0, "T": 3}
                async with s.get(base + "/api/calibration") as r:
                    j2 = await r.json()
                assert j2["device_matches"] and j2["captures"] == ["front", "left", "right"]
                saved = json.load(open(tmp_path / "calibration.json"))
                assert saved["mic_words"] == full(FRONT_PERMUTED) and saved["slot_order"] == "right_first"
                assert os.path.exists(tmp_path / "captures" / "front.wav")
                # and back to a back mic: C<b>,<l>,<r>,<t> with no suffix, the board reports B= again
                body = dict(saved, mic_words=full(RIGHT_FIRST_PCB))
                async with s.post(base + "/api/calibration", json=body) as r:
                    assert (await r.json())["ok"]
                async with s.post(base + "/api/calibration/apply") as r:
                    ap = await r.json()
                assert ap["ok"] and ap["device_matches"] and ap["sent"]["C"] == "0,3,2,5", ap
                assert ap["device"]["mics"] == {"B": 0, "L": 3, "R": 2, "T": 5}
        asyncio.run(asyncio.wait_for(run(), 60))
    finally:
        for p in (dash, fake):
            if p is not None:
                p.terminate()
                try:
                    p.wait(5)
                except subprocess.TimeoutExpired:
                    p.kill()
