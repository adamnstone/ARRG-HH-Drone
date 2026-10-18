"""Headset calibration: which PDM DMA word carries which mic position, and which hole each motor sits in.

The headset has four mics: left, right and top (L, R, T), plus ONE of front or back (F or B, never both: they share
the DSP's channel 0; back is the default). The geometry is mirror-symmetric, so a front mic does exactly what a back
mic does. The v3 PCB has five mic headers, MK1 (DIN0, SEL GND), MK2 (DIN0, SEL 3V3), MK3 (DIN1, SEL GND), MK4 (DIN1,
SEL 3V3) and MK5 (DIN2, SEL GND), and any of them can carry the mic of any position (at most four connected). The
labels say where MK2..MK5 go by default (back, left, right, top); MK1 has no default position. The board reads its
mics as six int16 DMA words per sample (three PDM data lines x two SEL slots, w0..w5): with slot order left_first a
header's word is 2*line + (0 for SEL GND, 1 for SEL 3V3), right_first swaps the phases. right_first is what the v3
board does (verified by tapping: it reads the SEL 3V3 phase first), so it is the default: MK1..MK5 on w1, w0, w3, w2,
w5 (w4 has no header); left_first puts them on w0..w4 (w5 has none). Which word is which position depends on how the
headers were populated and on the hardware's slot order, so it is calibrated rather than assumed. Motors M1..M8
(pins.h order F, FR, R, BR, B, BL, L, FL) are panned by angle, and the angle of each is the hole of the band it sits
in (webapp/static/headband_geometry.json, from the STL).

  calibration file   data/calibration.json (HH_CALIBRATION overrides the path):
                     {"version": 1, "updated": .., "mic_words": {"front": null, "back": 0, "left": 3, "right": 2,
                      "top": 5}, "mic_source": .., "slot_order": "right_first" | "left_first",
                      "motors": {"M1": {"hole": "H1", "theta_deg": 0.0}, ..., "M8": null}, "motor_source": ..}
                     (front and back are never both set; a file without slot_order is right_first)
  to the firmware    C<word for back>,<left>,<right>,<top>   or, with a front mic,  C<front>,<left>,<right>,<top>,F
                     (null = not connected -> -1)  and  Y<theta M1>,...,<theta M8>  (null = not connected -> 999)
  from the firmware  "# cal mics B=w0 L=w3 R=- T=w5 motors M1=0.0 ... M8=- src=nvs", or F=w.. for channel 0 at the
                     front (parse_cal_line: {"mics": {"F" or "B": word, "L": .., "R": .., "T": ..}, ...})
The board's DSP uses exactly the connected mics: their mean for detection, and per frame the best three or four of
them for the direction (droneears/bearing.py, the firmware's DSP core). The T lines' mused bit 1 is channel 0, the
front or the back mic, whichever the '# cal' line names.

Auto-calibration (solve): three or more short raw captures of all six words while someone claps in front of / to the
right / to the left of the headset (back and top optional). Per capture: which words carry a mic (live, not a copy of
another word, and the clap is clearly above their noise), when the clap arrives at each (GCC-PHAT on the clap window),
and how loud it is there. Then right = the earliest word on the right clap, left = the earliest on the left clap;
front = the word that hears the front clap clearly first, else back = the one that hears it clearly last (the side
and top mics hear it at about the same time; with a back clap: back = the earliest on that); only one of the two can
be fitted, so if both show up the clearer lead wins (with a note). top = the earliest on a top clap, else the one mic
left over. A pick needs a clear lead over every other mic (LEAD_MIN_MS), so a headset without, say, a left mic leaves
left empty instead of guessing. A data line with one mic reads it in both SEL slots (the empty slot mirrors it), so
either word gives the DSP the same signal; the position goes to a word with a header under the slot order in use,
and when both have one, to the header that is connected in the current map, else the one whose PCB label is that
position. With a single mic there is no timing to compare, so it is placed by where the clap was loudest (low
confidence). synth_capture() makes the same captures from a known wiring (tests, webapp/fake_device.py).
"""
import json, math, os, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CAL_PATH = os.environ.get("HH_CALIBRATION", os.path.join(ROOT, "data", "calibration.json"))
CAPTURE_DIR = os.environ.get("HH_CAL_CAPTURES", os.path.join(ROOT, "data", "calibration", "captures"))
GEOMETRY_PATH = os.path.join(ROOT, "webapp", "static", "headband_geometry.json")

FS = 48000
N_WORDS = 6
POSITIONS = ["front", "back", "left", "right", "top"]
CH0 = {"F": "front", "B": "back"}                      # DSP channel 0: the front or the back mic, never both
DSP_CHANNELS = [("L", "left"), ("R", "right"), ("T", "top")]   # channels 1..3: C<ch0>,<l>,<r>,<t>[,F]
FRONT_BACK = "front and back share one channel: choose one"
MOTORS = [f"M{k}" for k in range(1, 9)]
MOTOR_DEFAULT_THETA = [0.0, 45.0, 90.0, 135.0, 180.0, -135.0, -90.0, -45.0]    # F, FR, R, BR, B, BL, L, FL
NOT_FITTED = 999                                       # Y argument for a motor that is not fitted
# the PCB's mic headers: (id, its labelled position or None, PDM data line, SEL strap); any header can take any position
HEADERS = [("MK1", None, 0, "GND"), ("MK2", "back", 0, "3V3"), ("MK3", "left", 1, "GND"), ("MK4", "right", 1, "3V3"),
           ("MK5", "top", 2, "GND")]
LINE_PAIRS = [(0, 1), (2, 3), (4, 5)]                  # words sharing one PDM data line (SEL GND, SEL 3V3)
DIRECTIONS = ["front", "right", "left", "back", "top"]
REQUIRED = ["front", "right", "left"]
DIRECTION_ANGLES = {"front": (0.0, 0.0), "right": (90.0, 0.0), "left": (-90.0, 0.0), "back": (180.0, 0.0),
                    "top": (0.0, 90.0)}                # (theta, elevation) in degrees
MIC_SOURCES = ("default", "manual", "auto")
MOTOR_SOURCES = ("default", "manual")
SLOT_ORDERS = ("left_first", "right_first")            # which SEL phase the DMA puts first: MKx header -> word
DEFAULT_SLOT_ORDER = "right_first"                     # the v3 board reads the SEL 3V3 phase first (verified)


def header_words(slot_order=DEFAULT_SLOT_ORDER):
    """{header id: DMA word} under a slot order: 2 * line + (0 for SEL GND, 1 for SEL 3V3) when left_first, the
    phases swapped when right_first."""
    right = slot_order == "right_first"
    return {h: 2 * line + (0 if (sel == "GND") != right else 1) for h, _, line, sel in HEADERS}


PCB_WORDS = {pos: header_words()[h] for h, pos, _, _ in HEADERS if pos}   # back 0, left 3, right 2, top 5 (right first)

# solver thresholds
LEAD_MIN_MS = 0.08         # a pick (earliest / latest) needs this lead over every other mic, else no mic there
SIDE_SPLIT_MS = 0.3        # the right mic leads the left one on the right clap by ~0.5 ms (17 cm across)
LIVE_STD = 3e-5            # full scale: below this a word is flat (no mic, the line is held)
MAX_DC = 0.5               # an undriven PDM line decimates to about -0.94 full scale
MIRROR_LSB = 2             # a word within this many LSB of another everywhere is a copy (empty SEL phase)
EVENT_MIN_DB = 15.0        # the clap window must be this far above the capture's median 100 ms level
RESPONDS_DB = 10.0         # ... and a word carries a mic only if the clap shows in it by this much
BAND = (200.0, 8000.0)     # GCC-PHAT band
MAX_LAG_S = 0.8e-3         # the head is ~20 cm across: 0.6 ms at most, plus margin
MARGIN_FULL_MS = 0.15      # an arrival margin of this much counts as fully confident


# ---------------------------------------------------------------- geometry
def placeholder_geometry():
    """8 holes at the default motor angles on an 83 x 110 mm ellipse, until the STL-derived file exists."""
    ax, ay = 83.0, 110.0
    holes = []
    for k, th in enumerate(sorted(MOTOR_DEFAULT_THETA, key=lambda t: t % 360.0)):
        r = math.radians(th)
        holes.append(dict(id=f"H{k + 1}", theta_deg=th, x_mm=round(ax * math.sin(r), 2),
                          y_mm=round(ay * math.cos(r), 2), z_mm=8.0, diameter_mm=8.0))
    outline = [[round(ax * math.sin(t), 2), round(ay * math.cos(t), 2)] for t in np.linspace(0, 2 * math.pi, 73)]
    sites = [dict(position=p, theta_deg=th, x_mm=round(ax * math.sin(math.radians(th)), 2),
                  y_mm=round(ay * math.cos(math.radians(th)), 2), z_mm=8.0)
             for p, th in (("front", 0.0), ("right", 90.0), ("back", 180.0), ("left", -90.0))]
    sites.append(dict(position="top", theta_deg=0.0, x_mm=0.0, y_mm=0.0, z_mm=90.0))
    return {"source": "placeholder (no headband_geometry.json yet)", "units": "mm",
            "frame": "x right, y forward, z up, origin at the band centre; theta_deg = atan2(x, y): 0 front, +90 right",
            "front_axis_note": "placeholder: 8 holes at the default motor angles", "outline": outline,
            "holes": holes, "mic_sites": sites, "placeholder": True}


def load_geometry(path=GEOMETRY_PATH):
    if os.path.exists(path):
        with open(path) as f:
            g = json.load(f)
        g.setdefault("holes", []); g.setdefault("mic_sites", []); g.setdefault("outline", [])
        g["placeholder"] = False
        return g
    return placeholder_geometry()


def _adiff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def nearest_hole(theta, geometry):
    holes = geometry.get("holes") or []
    return min(holes, key=lambda h: _adiff(float(h["theta_deg"]), theta))["id"] if holes else None


# ---------------------------------------------------------------- calibration file
def default_calibration(geometry=None):
    """PCB mic map; each motor in the hole nearest its pins.h direction, at that hole's measured angle (the STL's
    holes are not 45 deg apart: e.g. M2 'FR' sits at +36.7 deg, M3 'R' at +82.9), or the nominal angle without
    geometry."""
    g = geometry or load_geometry()
    holes = {h["id"]: float(h["theta_deg"]) for h in g.get("holes") or []}
    motors = {}
    for m, th in zip(MOTORS, MOTOR_DEFAULT_THETA):
        hole = nearest_hole(th, g)
        motors[m] = {"hole": hole, "theta_deg": round(holes.get(hole, th), 1)}
    return {"version": 1, "updated": None, "mic_words": {p: PCB_WORDS.get(p) for p in POSITIONS},
            "mic_source": "default", "slot_order": DEFAULT_SLOT_ORDER, "motors": motors, "motor_source": "default"}


def validate(cal, geometry=None):
    """A clean copy of `cal` (see the module docstring) or ValueError naming the problem. Missing positions / motors
    are not fitted (null); a motor with a hole and no theta takes the hole's angle; unknown keys are dropped. Front and
    back share the DSP's channel 0, so at most one of them has a word."""
    if not isinstance(cal, dict):
        raise ValueError("calibration must be a JSON object")
    g = geometry or load_geometry()
    holes = {h["id"]: float(h["theta_deg"]) for h in g.get("holes") or []}
    mw = cal.get("mic_words") or {}
    if not isinstance(mw, dict):
        raise ValueError("mic_words must be an object {position: word}")
    bad = sorted(set(mw) - set(POSITIONS))
    if bad:
        raise ValueError(f"mic_words: unknown position(s) {bad}; positions are {POSITIONS}")
    words, used = {}, {}
    for p in POSITIONS:
        w = mw.get(p)
        if w is None:
            words[p] = None
            continue
        if isinstance(w, bool) or not isinstance(w, (int, float)) or int(w) != w or not 0 <= int(w) < N_WORDS:
            raise ValueError(f"mic_words.{p}: {w!r} is not a word 0..{N_WORDS - 1} or null")
        w = int(w)
        if w in used:
            raise ValueError(f"mic_words: word {w} is used by both {used[w]} and {p}")
        used[w] = p; words[p] = w
    if words["front"] is not None and words["back"] is not None:
        raise ValueError(FRONT_BACK)
    mo = cal.get("motors") or {}
    if not isinstance(mo, dict):
        raise ValueError("motors must be an object {M1: {hole, theta_deg} | null, ...}")
    bad = sorted(set(mo) - set(MOTORS))
    if bad:
        raise ValueError(f"motors: unknown output(s) {bad}; outputs are M1..M8")
    motors = {}
    for m in MOTORS:
        v = mo.get(m)
        if v is None:
            motors[m] = None
            continue
        if not isinstance(v, dict):
            raise ValueError(f"motors.{m}: want {{\"hole\": .., \"theta_deg\": ..}} or null")
        hole = v.get("hole")
        if hole is not None and hole not in holes:
            raise ValueError(f"motors.{m}: no hole {hole!r} in the headband geometry")
        th = v.get("theta_deg")
        if th is None:
            if hole is None:
                raise ValueError(f"motors.{m}: needs a hole or theta_deg (or null if not fitted)")
            th = holes[hole]
        try:
            th = float(th)
        except (TypeError, ValueError):
            raise ValueError(f"motors.{m}.theta_deg: {th!r} is not a number")
        if not math.isfinite(th) or not -180.0 <= th <= 180.0:
            raise ValueError(f"motors.{m}.theta_deg: {th} is outside -180..180")
        motors[m] = {"hole": hole, "theta_deg": round(th, 2)}
    ms, os_ = cal.get("mic_source", "manual"), cal.get("motor_source", "manual")
    if ms not in MIC_SOURCES:
        raise ValueError(f"mic_source must be one of {MIC_SOURCES}")
    if os_ not in MOTOR_SOURCES:
        raise ValueError(f"motor_source must be one of {MOTOR_SOURCES}")
    so = cal.get("slot_order") or DEFAULT_SLOT_ORDER
    if so not in SLOT_ORDERS:
        raise ValueError(f"slot_order must be one of {SLOT_ORDERS}")
    return {"version": 1, "updated": cal.get("updated"), "mic_words": words, "mic_source": ms,
            "slot_order": so, "motors": motors, "motor_source": os_}


def load_calibration(path=None):
    """(calibration, error): the saved file validated, or the defaults (error says why when a file exists but is
    unusable)."""
    path = path or CAL_PATH
    if not os.path.exists(path):
        return default_calibration(), None
    try:
        with open(path) as f:
            return validate(json.load(f)), None
    except (OSError, ValueError) as e:
        return default_calibration(), f"{os.path.relpath(path, ROOT)}: {e}; using the defaults"


def save_calibration(cal, path=None):
    path = path or CAL_PATH
    c = validate(cal)
    c["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(c, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)
    return c


def _num(t):
    return format(round(float(t), 2), "g")


def channel0(mic_words):
    """(letter, word) of the DSP's channel 0: ("F", the front word) when a front mic is set, else ("B", the back word
    or None)."""
    if mic_words.get("front") is not None:
        return "F", mic_words["front"]
    return "B", mic_words.get("back")


def firmware_args(cal):
    """(C argument, Y argument) for the device: words for channel 0, L, R, T (-1 = not connected), then ",F" when
    channel 0 is the front mic (C<front>,<l>,<r>,<t>,F; the back form has no suffix), and the 8 motor angles
    (999 = not connected)."""
    mw = cal["mic_words"]
    side, w0 = channel0(mw)
    c = ",".join(str(-1 if w is None else int(w)) for w in [w0] + [mw.get(p) for _, p in DSP_CHANNELS])
    c += ",F" if side == "F" else ""
    y = ",".join(str(NOT_FITTED) if cal["motors"].get(m) is None else _num(cal["motors"][m]["theta_deg"])
                 for m in MOTORS)
    return c, y


def parse_cal_line(body):
    """A '# cal ...' reply without the '# ' -> {"mics": {"B": 0, "L": 3, "R": None, "T": 5},
    "motors": {"M1": 0.0, .., "M3": None}, "src": "nvs", "saved": bool} (only the parts the line carries), or
    {"error": text} for '# cal: bad ...'. Channel 0 is keyed by the letter the board prints: "F" when it is the
    front mic (C<w0>,<l>,<r>,<t>,F), "B" for the back one; exactly one of the two."""
    body = body.strip()
    if body.startswith("cal:"):
        return {"error": body[4:].strip()}
    out = {}
    for tok in body.split()[1:]:
        if "=" not in tok:
            if tok == "saved":
                out["saved"] = True
            continue
        k, v = tok.split("=", 1)
        if k in CH0 or k in ("L", "R", "T"):
            out.setdefault("mics", {})[k] = None if v in ("-", "w-") else int(v.lstrip("w"))
        elif k in MOTORS:
            out.setdefault("motors", {})[k] = None if v == "-" else float(v)
        elif k == "src":
            out["src"] = v
    m = out.get("mics")
    if m is not None and (("F" in m) == ("B" in m) or not all(ch in m for ch, _ in DSP_CHANNELS)):
        del out["mics"]                 # the board always prints F or B, then L, R, T: a line missing one lost bytes
    return out


def device_matches(device, cal):
    """True when the device's reported cal (parse_cal_line, merged) equals what `cal` would send: channel 0's side
    (front or back) and word, the L, R and T words and the motor angles."""
    if not device or "mics" not in device or "motors" not in device:
        return False
    mw = cal["mic_words"]
    side, w0 = channel0(mw)
    if side not in device["mics"] or device["mics"][side] != w0:
        return False
    for ch, p in DSP_CHANNELS:
        if device["mics"].get(ch) != mw.get(p):
            return False
    for m in MOTORS:
        want = None if cal["motors"].get(m) is None else cal["motors"][m]["theta_deg"]
        got = device["motors"].get(m)
        if (want is None) != (got is None) or (want is not None and abs(want - got) > 0.05):
            return False
    return True


# ---------------------------------------------------------------- captures
def to_float(x):
    """int16 (n, 6) from the raw packets -> float32 full scale 1."""
    x = np.asarray(x)
    return (x.astype(np.float32) / 32768.0) if x.dtype.kind == "i" else x.astype(np.float32)


def _mirrors(x, live):
    """{word: the lower word it copies} for live words that equal another live word to within MIRROR_LSB."""
    out = {}
    tol = MIRROR_LSB / 32768.0
    for j in range(x.shape[1]):
        for i in range(j):
            if live[i] and live[j] and i not in out and np.max(np.abs(x[:, i] - x[:, j])) <= tol:
                out[j] = i
                break
    return out


def _window_rms_db(v, win):
    n = len(v) // win
    if n < 1:
        return np.array([20 * math.log10(max(float(np.std(v)), 1e-9))])
    r = np.sqrt(np.mean(v[: n * win].reshape(n, win) ** 2, axis=1))
    return 20 * np.log10(np.maximum(r, 1e-9))


def find_event(x, words):
    """The loudest short burst (a clap) in the sum of the given words' envelopes: dict(found, i0, i1, t_s, peak_dbfs,
    above_db). The window runs 30 ms before to 70 ms after the envelope peak."""
    if not words:
        return dict(found=False, i0=0, i1=0, t_s=None, peak_dbfs=None, above_db=0.0)
    ac = x[:, words] - np.mean(x[:, words], axis=0)
    env = np.sum(np.abs(ac), axis=1)
    k = max(1, int(0.005 * FS))
    env = np.convolve(env, np.ones(k) / k, mode="same")
    i = int(np.argmax(env))
    i0, i1 = max(0, i - int(0.03 * FS)), min(len(env), i + int(0.07 * FS))
    lv = _window_rms_db(np.sum(ac, axis=1), int(0.1 * FS))
    ev = 20 * math.log10(max(float(np.sqrt(np.mean(np.sum(ac[i0:i1], axis=1) ** 2))), 1e-9))
    above = ev - float(np.median(lv))
    peak = 20 * math.log10(max(float(np.max(np.abs(x[i0:i1, words]))), 1e-9))
    return dict(found=bool(above >= EVENT_MIN_DB), i0=i0, i1=i1, t_s=round(i / FS, 3), peak_dbfs=round(peak, 1),
                above_db=round(above, 1))


def capture_summary(x):
    """Per-word stats and the clap of one capture (float (n, 6) or int16), in the shape the capture endpoint returns:
    {"words": [{"w", "rms_dbfs", "peak_dbfs", "dc", "live", "mirror_of", "responds", "event_db"}], "event": {...}}."""
    x = to_float(x)
    nw = x.shape[1]
    dc = x.mean(axis=0)
    std = x.std(axis=0)
    live = [bool(std[w] > LIVE_STD and abs(dc[w]) < MAX_DC) for w in range(nw)]
    mir = _mirrors(x, live)
    cand = [w for w in range(nw) if live[w] and w not in mir]
    ev = find_event(x, cand)
    words = []
    for w in range(nw):
        v = x[:, w] - dc[w]
        rms = float(np.sqrt(np.mean(v ** 2)))
        resp, ev_db = False, None
        if live[w] and ev["i1"] > ev["i0"]:
            e = 20 * math.log10(max(float(np.sqrt(np.mean(v[ev["i0"]:ev["i1"]] ** 2))), 1e-9))
            ev_db = round(e - float(np.median(_window_rms_db(v, int(0.1 * FS)))), 1)
            resp = bool(ev["found"] and ev_db >= RESPONDS_DB)
        words.append(dict(w=w, rms_dbfs=round(20 * math.log10(max(rms, 1e-9)), 1),
                          peak_dbfs=round(20 * math.log10(max(float(np.max(np.abs(x[:, w]))), 1e-9)), 1),
                          dc=round(float(dc[w]), 4), live=live[w], mirror_of=mir.get(w), responds=resp,
                          event_db=ev_db))
    event = {k: ev[k] for k in ("found", "t_s", "peak_dbfs", "above_db")}
    return {"words": words, "event": event}


def gcc_phat_tau(a, b, max_lag_s=MAX_LAG_S, band=BAND):
    """Delay of b relative to a (seconds; positive = b later) by band-limited GCC-PHAT, parabolic interpolation."""
    n = 1 << int(math.ceil(math.log2(2 * len(a))))
    A, B = np.fft.rfft(a, n), np.fft.rfft(b, n)
    G = np.conj(A) * B
    G /= np.abs(G) + 1e-12
    f = np.fft.rfftfreq(n, 1 / FS)
    G[(f < band[0]) | (f > band[1])] = 0
    r = np.fft.irfft(G, n)
    L = int(math.ceil(max_lag_s * FS)) + 1
    lags = np.arange(-L, L + 1)
    seg = r[lags % n]
    k = int(np.argmax(seg))
    frac = 0.0
    if 0 < k < len(seg) - 1:
        y0, y1, y2 = seg[k - 1], seg[k], seg[k + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            frac = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
    return (lags[k] + frac) / FS


def arrivals_ms(x, words, i0, i1):
    """Arrival time of the event at each word relative to their mean (ms): least squares over all pairwise GCC-PHAT
    delays, a_k = mean_j tau(j -> k)."""
    seg = x[i0:i1, words] - np.mean(x[i0:i1, words], axis=0)
    seg = seg * np.hanning(len(seg))[:, None]
    n = len(words)
    tau = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            t = gcc_phat_tau(seg[:, i], seg[:, j])
            tau[i, j], tau[j, i] = t, -t
    a = tau.mean(axis=0)                      # a_k - mean(a) = (1/n) sum_j (a_k - a_j)
    return {w: 1000.0 * a[k] for k, w in enumerate(words)}


def analyse(x):
    """One capture -> dict(summary, mics (words carrying a mic and hearing the clap), arrivals {w: ms},
    levels {w: dBFS in the clap window}, event)."""
    x = to_float(x)
    s = capture_summary(x)
    mics = [d["w"] for d in s["words"] if d["live"] and d["mirror_of"] is None and d["responds"]]
    ev = find_event(x, [d["w"] for d in s["words"] if d["live"] and d["mirror_of"] is None])
    levels = {}
    for w in mics:
        v = x[ev["i0"]:ev["i1"], w] - np.mean(x[:, w])
        levels[w] = round(20 * math.log10(max(float(np.sqrt(np.mean(v ** 2))), 1e-9)), 1)
    arr = arrivals_ms(x, mics, ev["i0"], ev["i1"]) if len(mics) >= 2 else {w: 0.0 for w in mics}
    return dict(summary=s, mics=mics, arrivals=arr, levels=levels, event=s["event"])


def _header_word_for(w, position, mirrors, slot_order, taken, fitted=()):
    """A mic word on a data line with one mic: both SEL slots read it (`mirrors` {copy: word}), so either word gives
    the DSP the same signal. Report a slot that has a header under `slot_order` (the lower word, which the mirror
    detection keeps, may have none); when both have one (MK1 / MK2 on DIN0, MK3 / MK4 on DIN1), the one connected in
    the current map (`fitted` words), else the one whose PCB label is `position` (MK4 for right), else w."""
    hw = {v: h for h, v in header_words(slot_order).items()}
    pcb = {h: pos for h, pos, _, _ in HEADERS}
    c = w ^ 1
    if c in taken or not (mirrors.get(str(c)) == w or mirrors.get(str(w)) == c) or c not in hw:
        return w
    if w not in hw:
        return c
    if (c in fitted) != (w in fitted):
        return c if c in fitted else w
    return c if pcb[hw[c]] == position and pcb[hw[w]] != position else w


def solve(captures, slot_order=None, current=None):
    """captures: {direction: (n, 6) samples} with at least front, right, left -> the proposal (module docstring):
    dict(ok, mic_words {position: word or None}, confidence 0..1, notes, live_words, mirrors {copy: word},
    per_direction {direction: order, tdoa_ms, levels_dbfs, event}, slot_order). slot_order (default right_first) and
    current (the mic_words in use: which headers are connected) decide which slot of a mirrored line gets the
    position. ValueError when a required capture is missing."""
    missing = [d for d in REQUIRED if d not in captures]
    if missing:
        raise ValueError(f"missing capture(s): {', '.join(missing)} (record front, right and left first)")
    slot_order = slot_order if slot_order in SLOT_ORDERS else DEFAULT_SLOT_ORDER
    fitted = {int(w) for w in (current or {}).values() if isinstance(w, int) and not isinstance(w, bool)}
    notes = []
    an = {d: analyse(x) for d, x in captures.items() if d in DIRECTIONS}
    for d, a in an.items():
        if not a["event"]["found"]:
            notes.append(f"{d}: no clear clap found ({a['event']['above_db']:.0f} dB above the noise); clap louder "
                         f"or closer and record again")
    # words that carry a mic: heard the clap in most of the captures that have one
    counts = {}
    with_event = [d for d in an if an[d]["event"]["found"]]
    for d in with_event:
        for w in an[d]["mics"]:
            counts[w] = counts.get(w, 0) + 1
    need = max(1, (len(with_event) + 1) // 2)
    mic_words = sorted(w for w, c in counts.items() if c >= need)
    mirrors = {}
    for a in an.values():
        for d in a["summary"]["words"]:
            if d["mirror_of"] is not None and d["mirror_of"] in mic_words:
                mirrors[str(d["w"])] = d["mirror_of"]
    noisy = sorted({d["w"] for a in an.values() for d in a["summary"]["words"]
                    if d["live"] and d["mirror_of"] is None and d["w"] not in mic_words})
    if noisy:
        notes.append(f"{', '.join('w%d' % w for w in noisy)}: signal but no clap (noise or crosstalk on an empty "
                     f"line), not a mic")
    per = {}
    for d, a in an.items():
        arr = {w: a["arrivals"][w] for w in mic_words if w in a["arrivals"]}
        per[d] = {"order": sorted(arr, key=lambda w: arr[w]), "tdoa_ms": {str(w): round(v, 3) for w, v in arr.items()},
                  "levels_dbfs": {str(w): a["levels"][w] for w in mic_words if w in a["levels"]},
                  "event": a["event"]}
    result = {p: None for p in POSITIONS}
    conf = []

    def finish():
        taken = set(w for w in result.values() if w is not None)
        for p, w in result.items():                          # a mirrored line: the slot with the header
            if w is not None:
                c = _header_word_for(w, p, mirrors, slot_order, taken, fitted)
                if c != w:
                    taken.discard(w); taken.add(c); result[p] = c
        hw = {w: h for h, w in header_words(slot_order).items()}
        for cp, src in sorted(mirrors.items()):
            a, b = sorted((int(cp), src))
            p = next((q for q, w in result.items() if w in (a, b)), None)
            at = result[p] if p else src
            notes.append(f"w{a} and w{b} read the same (one mic on that data line, the empty slot mirrors it)"
                         + (f": {p} goes to w{at}" if p else "")
                         + (f", {hw[at]}'s slot with the {slot_order.replace('_', ' ')} slot order" if at in hw else ""))
        return dict(ok=True, mic_words=result, confidence=round(min(conf) if conf else 0.0, 2), notes=notes,
                    live_words=mic_words, mirrors=mirrors, per_direction=per, slot_order=slot_order)

    if not mic_words:
        notes.append("no mic heard the claps: check the board is in calibration mode and clap louder")
        return finish()
    if len(mic_words) == 1:
        w = mic_words[0]
        lv = {d: per[d]["levels_dbfs"].get(str(w)) for d in per if per[d]["levels_dbfs"].get(str(w)) is not None
              and an[d]["event"]["found"]}
        side = {d: v for d, v in lv.items() if d in POSITIONS}     # a clap from a side that can have a mic
        best = max(side, key=lambda d: side[d]) if side else None
        if best:
            margin = side[best] - max([v for d, v in lv.items() if d != best], default=side[best])
            result[best] = w
            conf.append(max(0.0, min(0.5, 0.1 * margin)))
            notes.append(f"only one mic (w{w}): no arrival times to compare, so it is placed where the clap was "
                         f"loudest ({best}, {margin:+.1f} dB against the next); check it by hand"
                         + "".join(f"; record a {d} clap too" for d in ("back", "top") if d not in lv))
        return finish()

    picked = {}                                              # position -> (word, lead in ms)

    def ahead(direction, latest=False):
        """(word, ms) of the earliest (latest) mic on that clap and its lead over the next one; None without a clap
        or with fewer than two mics."""
        if direction not in an or not an[direction]["event"]["found"]:
            return None
        arr = {w: an[direction]["arrivals"][w] for w in mic_words if w in an[direction]["arrivals"]}
        if len(arr) < 2:
            return None
        order = sorted(arr, key=lambda w: arr[w], reverse=latest)
        return order[0], abs(arr[order[1]] - arr[order[0]])

    def pick(direction, position, latest=False):
        """The earliest (latest) mic on that clap, if it leads every other mic clearly."""
        r = ahead(direction, latest)
        if r is None:
            return
        w, lead = r
        what = "last" if latest else "first"
        if lead < LEAD_MIN_MS:
            notes.append(f"{position}: no mic clearly {what} on the {direction} clap (w{w} by {lead:.2f} ms): no "
                         f"{position} mic, or clap again from straight {direction}")
            return
        other = next((p for p, (v, *_) in picked.items() if v == w), None)
        if other is not None:
            keep = position if lead > picked[other][1] else other
            notes.append(f"w{w} fits both {other} and {position}; kept {keep} (the clearer lead)")
            if keep == other:
                return
            del picked[other]
        c = min(1.0, lead / MARGIN_FULL_MS)
        lv = per[direction]["levels_dbfs"]
        if not latest and len(lv) > 1 and max(lv, key=lambda k: lv[k]) != str(w):
            notes.append(f"{position}: w{w} heard the {direction} clap first but w{max(lv, key=lambda k: lv[k])} "
                         f"heard it loudest")
            c *= 0.8
        if lead < 0.1:
            notes.append(f"{position}: w{w} only {lead:.2f} ms ahead of the next mic on the {direction} clap "
                         f"(weak; clap closer, from straight {direction})")
        picked[position] = (w, lead, c)

    pick("right", "right")
    pick("left", "left")
    # channel 0: a front mic hears the front clap clearly first, a back mic clearly last (the side and top mics hear
    # it together), or first on a back clap if there is one. Only one of the two can be fitted.
    back_clap = "back" in an and an["back"]["event"]["found"]
    fr = ahead("front")
    bk = ahead("back") if back_clap else ahead("front", latest=True)
    how = "first on the back clap" if back_clap else "last on the front clap"
    clear = lambda r: r is not None and r[1] >= LEAD_MIN_MS
    keep = None
    if clear(fr) and clear(bk):
        keep = "front" if fr[1] >= bk[1] else "back"
        notes.append(f"w{fr[0]} is clearly first on the front clap (a front mic) and w{bk[0]} clearly {how} (a back "
                     f"mic), but {FRONT_BACK.split(':')[0]}: kept {keep} (the clearer lead); check the wiring")
    elif clear(fr):
        keep = "front"
    elif clear(bk):
        keep = "back"
    elif fr is not None or bk is not None:
        if back_clap:
            what = "first on the front clap or on the back clap"
            seen = [f"{d} clap w{r[0]} by {r[1]:.2f} ms" for d, r in (("front", fr), ("back", bk)) if r]
        else:
            what = "first or last on the front clap"
            seen = [f"first w{fr[0]} by {fr[1]:.2f} ms", f"last w{bk[0]} by {bk[1]:.2f} ms"]
        notes.append(f"front / back: no mic clearly {what} ({', '.join(seen)}): no front or back mic, or clap again "
                     f"from straight front{' / back' if back_clap else ''}")
    if keep == "front":
        pick("front", "front")
    elif keep == "back" and back_clap:
        pick("back", "back")
    elif keep == "back":
        pick("front", "back", latest=True)
    if "top" in an and an["top"]["event"]["found"]:
        pick("top", "top")
    for p, (w, _, c) in picked.items():
        result[p] = w
        conf.append(c)
    free = sorted(set(mic_words) - set(result.values()))
    if result["top"] is None and len(free) == 1 and not ("top" in an and an["top"]["event"]["found"]):
        taken = {w for w in result.values() if w is not None}    # name the word finish() will report
        result["top"] = _header_word_for(free.pop(), "top", mirrors, slot_order, taken, fitted)
        conf.append(0.7)
        notes.append(f"top: w{result['top']} is the one mic left over (record a top clap to confirm it)")
    elif free:
        notes.append(f"{len(free)} mic{'s' if len(free) > 1 else ''} not placed ({', '.join('w%d' % w for w in free)})"
                     f": record a top clap, or set {'them' if len(free) > 1 else 'it'} by hand")
        conf.append(0.3)
    # consistency: the right mic leads the left one by ~0.5 ms on the right clap (and the reverse); a front clap
    # reaches the back mic last, a back clap the front mic
    for d, near, far in (("right", "right", "left"), ("left", "left", "right")):
        wn, wf = result.get(near), result.get(far)
        if d in an and wn is not None and wf is not None:
            a = an[d]["arrivals"]
            if wn in a and wf in a and a[wf] - a[wn] < SIDE_SPLIT_MS:
                notes.append(f"{d} clap: {near} (w{wn}) leads {far} (w{wf}) by only {a[wf] - a[wn]:+.2f} ms (expect "
                             f"~0.5); one of them may be the front, back or top mic")
                conf.append(0.4)
    wb = result.get("back")
    if wb is not None and "front" in an and an["front"]["event"]["found"]:
        a = an["front"]["arrivals"]
        others = [a[w] for p, w in result.items() if p != "back" and w is not None and w in a]
        if wb in a and others and a[wb] - max(others) < LEAD_MIN_MS:
            notes.append(f"front clap: back (w{wb}) is not clearly the last to hear it "
                         f"({a[wb] - max(others):+.2f} ms); the assignment may be wrong")
            conf.append(0.4)
    wf = result.get("front")
    if wf is not None and back_clap:
        a = an["back"]["arrivals"]
        others = [a[w] for p, w in result.items() if p != "front" and w is not None and w in a]
        if wf in a and others and a[wf] - max(others) < LEAD_MIN_MS:
            notes.append(f"back clap: front (w{wf}) is not clearly the last to hear it "
                         f"({a[wf] - max(others):+.2f} ms); the assignment may be wrong")
            conf.append(0.4)
    return finish()


def save_capture_wav(direction, x, folder=None):
    import soundfile as sf
    folder = folder or CAPTURE_DIR
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{direction}.wav")
    sf.write(path, to_float(x), FS, subtype="PCM_16")
    return path


# ---------------------------------------------------------------- synthetic captures (tests, fake device)
MIC_SITES_M = {"front": (0.0, 0.10, 0.0), "back": (0.0, -0.105, 0.0), "left": (-0.085, 0.0, 0.0),
               "right": (0.085, 0.0, 0.0), "top": (0.0, 0.0, 0.09)}     # x right, y forward, z up (metres)
FLAT_WORD = -0.944                                     # what an undriven PDM line reads on the board


def parse_wiring(text):
    """'back=0,left=3,right=2,top=5' or 'front=0,...' -> {position: word} (any subset of the positions; front and
    back never both)."""
    out = {}
    for part in filter(None, (s.strip() for s in str(text).split(","))):
        p, w = part.split("=")
        p = p.strip()
        if p not in POSITIONS:
            raise ValueError(f"wiring: unknown position {p!r} (the headset's mics are {', '.join(POSITIONS)})")
        out[p] = int(w)
    if "front" in out and "back" in out:
        raise ValueError(f"wiring: {FRONT_BACK}")
    return out


def synth_capture(wiring, direction=None, seconds=3.0, click_t=1.0, seed=0, noise_dbfs=-62.0, level=0.3,
                  crosstalk_words=(), shadow=0.5):
    """Six words (n, 6) float32 as the board's 'W' stream would carry them for a headset wired as `wiring`
    ({position: word}) hearing two claps from `direction` (a DIRECTIONS name or (theta, elev) degrees; None = quiet).
    Plane-wave arrivals at MIC_SITES_M, the near mic louder by up to `shadow` (head shadow), independent mic noise;
    an empty slot on a line with one mic copies that mic, a line with none reads FLAT_WORD; crosstalk_words carry
    noise that does not hear the claps. A headset has a front or a back mic, never both (ValueError)."""
    if wiring.get("front") is not None and wiring.get("back") is not None:
        raise ValueError(f"wiring: {FRONT_BACK}")
    from droneears.simulate import fractional_delay, unit_vector
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    clean = np.zeros(n, np.float32)
    if direction is not None:
        for t in (click_t, click_t + 0.45):
            i = int(t * FS); m = int(0.025 * FS)
            burst = rng.standard_normal(m) * np.exp(-np.arange(m) / (0.004 * FS))
            clean[i:i + m] += level * burst[: max(0, min(m, n - i))] / np.max(np.abs(burst))
    th, el = DIRECTION_ANGLES[direction] if isinstance(direction, str) else (direction or (0.0, 0.0))
    u = unit_vector(th, el)
    out = np.full((n, N_WORDS), FLAT_WORD, np.float32)
    wiring = {p: w for p, w in wiring.items() if w is not None}
    for p, w in wiring.items():
        site = np.array(MIC_SITES_M[p])
        sig = np.zeros(n, np.float32)
        if direction is not None:
            gain = 1.0 + shadow * float(np.dot(u, site / np.linalg.norm(site)))
            sig = gain * fractional_delay(clean, -(site @ u) / 343.0, FS)
        out[:, w] = sig + (10 ** (noise_dbfs / 20)) * rng.standard_normal(n)
    for a, b in LINE_PAIRS:                                # one mic on a line: its empty slot mirrors it
        has = [w in wiring.values() for w in (a, b)]
        if has[0] != has[1]:
            src, dst = (a, b) if has[0] else (b, a)
            out[:, dst] = out[:, src]
    for w in crosstalk_words:
        out[:, w] = 10 ** (-48 / 20) * rng.standard_normal(n)
    q = np.clip(np.round(out * 32768), -32768, 32767).astype(np.int16)
    return q
