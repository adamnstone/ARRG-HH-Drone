"""Frame sources for the dashboard. Each source runs in its own daemon thread and reports through two callbacks:

  emit(frame)      one dict per pipeline frame, in the wire format of server.py (frame, t_ms, f0, ..., hap, src)
  status(**kw)     partial updates of the source status (connected, port, firmware, chunk_us, free_heap, dropped, ...)

SerialSource reads the device protocol with scripts/telemetry.py's TelemetryParser (imported, not copied), over USB
serial or, with tcp=HOST[:PORT], over Wi-Fi (telemetry.open_tcp; same bytes, same parser);
ReplaySource replays a telemetry.py --log CSV; AudioSource runs droneears.pipeline.Pipeline over an audio file
(--simulate) or over the scripted synthetic scene of demo_scene() (--demo). Audio and replay sources are paced
in real time from t_ms and loop forever, with frame and t_ms offset on each loop so both stay monotonic.
"""
import collections, importlib.util, os, re, sys, threading, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location("hh_telemetry", os.path.join(ROOT, "scripts", "telemetry.py"))
telemetry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(telemetry)
_spec2 = importlib.util.spec_from_file_location("hh_recorder", os.path.join(ROOT, "scripts", "record_device.py"))
recorder = importlib.util.module_from_spec(_spec2)          # RawParser, Stats, dbfs: the raw-audio packet format
_spec2.loader.exec_module(recorder)
from webapp import calibration as calib                         # the headset calibration ('W' captures, C / Y / K)
TEAM_DIR = os.environ.get("TEAM_DIR", os.path.join(ROOT, "data", "team"))   # training clips: <label>/<file>; same
                                                              # override as droneears/dataset.py
CHECK_MAX_S = 120.0                                           # the post-record detector check runs on at most this

DEFAULT_MODEL = os.path.join(ROOT, "artifacts", "drone_cls_int8.tflite")


def _r(x, nd):
    return None if x is None else round(float(x), nd)


def _i(x):
    return None if x is None else int(x)


def motor_names():
    """Motor order of the wire frame's hap list = droneears/config.py MOTOR_ANGLES (F, FR, R, BR, B, BL, L, FL)."""
    from droneears import config as C
    return list(C.MOTOR_ANGLES)


MOTOR_MAX_ON = 3          # the board never energises more than 3 motors (pins.h MOTOR_MAX_ON)
OVERHEAD_STEP_MS = 100    # overhead chase: the triplet advances every 100 ms (headband.ino OVERHEAD_STEP_MS)


def applied_haptics(weights, overhead, t_ms):
    """What the board's alert_task actually drives, from haptic_weights() output (list in motor order): overhead
    (all-ones marker) -> the rotating triplet k, k+3, k+6 (mod n) with k = t_ms // 100; otherwise the weights with
    at most MOTOR_MAX_ON non-zero (the panner returns 2). Used for the simulated sources, so the dashboard shows the
    same chase as a real board."""
    n = len(weights)
    if overhead and n:
        k = int(t_ms // OVERHEAD_STEP_MS) % n
        on = {(k + 3 * j) % n for j in range(3)}
        return [1.0 if m in on else 0.0 for m in range(n)]
    w = [float(x or 0.0) for x in weights]
    while sum(1 for x in w if x > 0) > MOTOR_MAX_ON:
        w[min((i for i in range(n) if w[i] > 0), key=lambda i: w[i])] = 0.0
    return w


def wire_frame(src, frame, t_ms, f0, score, strength, flat, hits, ml, detected, theta, conf, overhead, hap,
               elev=None, batt_v=None, bstat=None, mused=None):
    """The one frame format every source produces (and the browser consumes). hap: list in motor_names() order.
    theta / elev None = no azimuth / elevation this frame. bstat: bearing status (0 none or unreliable, 1 ambiguous
    mirror, 2 three mics: azimuth only or elevation assumed above, 3 full 3D); mused: bitmask of the mics in this
    frame's fit (B=1, L=2, R=4, T=8); both None from sources that do not report them (older firmware and logs)."""
    return dict(frame=int(frame), t_ms=int(t_ms), f0=_r(f0, 1), score=_r(score, 2), strength=_r(strength, 2),
                flat=_r(flat, 3), hits=int(hits), ml=_r(ml, 3), detected=int(bool(detected)), theta=_r(theta, 1),
                conf=_r(conf, 3), elev=_r(elev, 1), overhead=int(bool(overhead)),
                batt_v=_r(batt_v, 2) if batt_v else None,          # 0.00 = not read yet on the device -> None
                hap=[_r(h, 2) for h in hap], bstat=_i(bstat), mused=_i(mused), src=src)


def device_row_to_frame(row, src="serial"):
    """A TelemetryParser / read_log row (column name -> float) -> wire frame. Missing columns become None/0.
    Motors are matched by name (hap_F ... hap_FL), so an older 3- or 4-motor log (hap_L, hap_R, hap_B[, hap_F])
    lands on the same-named pads and the rest stay 0."""
    g = row.get
    return wire_frame(src, g("frame") or 0, g("t_ms") if g("t_ms") is not None else (g("host_s") or 0) * 1000,
                      g("f0"), g("score"), g("strength"), g("flat"), g("hits") or 0, g("ml"), g("detected"),
                      g("theta"), g("conf"), g("overhead"), [g("hap_" + m) or 0 for m in motor_names()],
                      elev=g("elev"), batt_v=g("batt_v"), bstat=g("bstat"), mused=g("mused"))


def pipeline_to_frames(results, src):
    """Pipeline.run() dicts -> wire frames. Mono input has no bearing: theta/conf/elev None, motors off.
    hap is what the board would drive (applied_haptics: overhead -> rotating triplet). bstat / mused come from the
    pipeline's bearing solve when it reports them (None otherwise)."""
    from droneears import config as C
    names = motor_names()
    out = []
    for i, r in enumerate(results):
        hap = r.get("haptics") or {}
        t_ms = i * C.HOP * 1000.0 / C.FS
        ovh = bool(r.get("overhead", False)) and bool(r["detected"])
        out.append(wire_frame(src, i, t_ms, r["f0"], r["score"], r["strength"], r["flat"],
                              r["hits"], r["ml"], r["detected"], r.get("theta"), r.get("conf"),
                              r.get("overhead", False), applied_haptics([hap.get(k, 0.0) for k in names], ovh, t_ms),
                              elev=r.get("elev"), bstat=r.get("bstat"), mused=r.get("mused")))
    return out


class Source:
    kind = "?"

    def __init__(self, emit, status):
        self.emit, self.status = emit, status
        self.stop_ev = threading.Event()
        self.thread = threading.Thread(target=self._guarded, name=f"source-{self.kind}", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_ev.set()

    def _guarded(self):
        try:
            self.run()
        except Exception as e:                               # never kill the server; show it on the page instead
            self.status(connected=False, error=f"{type(e).__name__}: {e}")
            raise

    def play_loop(self, frames):
        """Stream precomputed frames in real time (by t_ms), looping, with monotonic frame / t_ms."""
        if not frames:
            self.status(connected=False, error="no frames")
            return
        t_first = frames[0]["t_ms"]
        span = frames[-1]["t_ms"] - t_first + 1000.0 * 1024 / 48000
        n = len(frames)
        loop = 0
        start = time.monotonic()
        while not self.stop_ev.is_set():
            for i, f in enumerate(frames):
                target = start + (loop * span + f["t_ms"] - t_first) / 1000.0
                dt = target - time.monotonic()
                if dt > 0 and self.stop_ev.wait(dt):
                    return
                if dt < -0.5:                                # fell behind (laptop slept): skip ahead, don't burst
                    start += -dt
                g = dict(f)
                g["frame"] = loop * n + i
                g["t_ms"] = int(round(loop * span + f["t_ms"] - t_first))
                self.emit(g)
            loop += 1
            self.status(loops=loop)


class ReplaySource(Source):
    kind = "replay"

    def __init__(self, path, emit, status):
        self.path = path
        super().__init__(emit, status)

    def run(self):
        rows, bad = telemetry.read_log(self.path)
        frames = [device_row_to_frame(r, "replay") for r in rows]
        if frames and not all("t_ms" in r for r in rows):    # old log without t_ms: pace by host time
            for f, r in zip(frames, rows):
                f["t_ms"] = int(r["host_s"] * 1000)
        self.status(connected=True, port=os.path.basename(self.path), file=self.path,
                    firmware=f"replay of {len(rows)} logged frames ({bad} malformed skipped)")
        self.play_loop(frames)


class AudioSource(Source):
    """--simulate FILE (kind 'simulate') or --demo (kind 'demo', audio from demo_scene())."""

    def __init__(self, kind, emit, status, path=None, use_ml=True, model=DEFAULT_MODEL):
        self.kind, self.path, self.use_ml, self.model = kind, path, use_ml, model
        super().__init__(emit, status)

    def run(self):
        from droneears.pipeline import Pipeline
        from droneears import config as C
        FS = C.FS
        label = os.path.basename(self.path) if self.path else "scripted demo scene"
        self.status(connected=False, port=label, file=self.path, firmware="computing frames ...")
        if self.kind == "demo":
            y = demo_scene()
        else:
            from droneears.audio import load
            y = load(self.path, mono=False)
            if y.shape[0] not in (1, C.N_MICS):          # not the board's channel count: detection only
                y = y[:1]
        clf = None
        if self.use_ml and self.model and os.path.exists(self.model):
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            from droneears.classifier import load_tflite_classifier
            clf = load_tflite_classifier(self.model)
        t0 = time.time()
        frames = pipeline_to_frames(Pipeline(classifier=clf).run(y), self.kind)
        ch = y.shape[0]
        self.status(connected=True, port=label, file=self.path,
                    firmware=(f"{ch} ch \u00b7 {y.shape[1] / FS:.0f} s loop \u00b7 ML {'on' if clf else 'off'}"
                              + ("" if ch == C.N_MICS else " \u00b7 no bearing")),
                    channels=ch, compute_s=round(time.time() - t0, 2))
        self.play_loop(frames)


def demo_scene(seed=1):
    """Scripted 40 s scene on the board's mics (droneears/config.py MIC_POS, B, L, R, T): quiet, a drone approaches
    from the right low on the horizon, circles to behind while climbing, passes overhead (elevation ~80 deg,
    23-28 s), comes down to the front-left and fades out. Elevation is real 3D geometry (simulate.arrivals_moving
    with elev_fn), which the top mic T resolves."""
    from droneears import config as C
    from droneears.simulate import synth_drone, arrivals_moving, add_noise
    dur = 40.0
    fs = C.FS
    x = synth_drone(dur, f0=210.0, seed=seed)
    # (time s, bearing deg) key points, linearly interpolated (shortest way around the circle)
    keys = [(0, 90), (11, 90), (19, 180), (22, 180), (29, 200), (31, -60), (40, -60)]

    def theta_fn(t):
        for (ta, a), (tb, b) in zip(keys, keys[1:]):
            if t <= tb:
                d = (b - a + 180) % 360 - 180
                return ((a + d * (t - ta) / max(tb - ta, 1e-9)) + 180) % 360 - 180
        return keys[-1][1]

    # (time s, elevation deg): low on the horizon, climbing while it circles, ~80 deg overhead 23-28 s, then down
    ekeys = [(0, 5), (11, 10), (17, 25), (21, 50), (23, 80), (28, 80), (30, 35), (33, 15), (40, 10)]

    def elev_fn(t):
        return float(np.interp(t, [k[0] for k in ekeys], [k[1] for k in ekeys]))

    full = arrivals_moving(x, theta_fn, elev_fn=elev_fn)              # (N_MICS, n)
    t = np.arange(len(x)) / fs
    env_db = np.interp(t, [0, 3, 9, 30, 35, 37, 40], [-60, -25, 0, 0, -12, -60, -60])
    env = (10 ** (env_db / 20)).astype(np.float32)
    y = full * env
    noise = add_noise(full, 8.0, rng=np.random.default_rng(seed)) - full   # white noise 8 dB below the loud drone
    return (0.1 * (y + noise)).astype(np.float32)


class SerialSource(Source):
    """The real device (or a fake on a pty / TCP port). Transport-agnostic: USB serial (auto-detects the port like
    telemetry.py) or, when tcp is not None, Wi-Fi TCP ('' = headband.local:5000, else 192.168.4.1:5000). Sends 'T'
    on connect and 'S' every 2 s, and reconnects (re-detecting the port when auto) whenever the link drops.
    Bench panel: send(cmd) queues one command byte for the device ('M' motor test, 'S' status, ...), haptics()
    queues a manual per-motor command ('h<L>,<R>,<B>,<hz>' or 'A'), levels() keeps the per-slot mic level readout
    going ('L' 4x a second) for a few seconds, and the board's replies ('# levels', '# dsp', '# motor test',
    '# haptics') go to the message callback as {"type": "bench", "kind": ...} dicts.
    record() queues a raw-audio recording into data/team/<label>/ (what scripts/train.py trains on): the link
    thread switches the board to 'R', writes the packets to a WAV (like scripts/record_device.py), returns to
    'T', and reports progress / result / the rules-only detector check as {"kind": "record", "state": ...}.
    Calibration (webapp/calibration.py): cal_capture() streams 'W' (all six PDM DMA words, unmapped) for a few
    seconds and returns the samples, reporting {"kind": "cal_capture", "state": ...}; cal_apply() sends the mic map
    ('C') and motor angles ('Y') and waits for the board's '# cal' replies; every '# cal' line (also the 'K' reply sent
    on connect) updates cal_device and goes out as {"kind": "cal", "text", "device"}.
    Stealth mode: set_stealth() sends 'Q1' / 'Q0' (the board's detect LED stays off; kept in its NVS) and waits for
    the '# led' reply; every '# led stealth=on|off' line (also the reply to the 'Q' sent on connect) sets status
    led_stealth and goes out as {"kind": "led", "stealth", "text"}."""
    kind = "serial"
    STATUS_EVERY = 2.0
    LEVELS_EVERY = 0.25
    TCP_TIMEOUT = 2.0
    REC_STALL_S = 10.0                                       # a recording with no audio for this long is ended
    CAL_STALL_S = 4.0                                        # a calibration capture with no 6-word audio fails
    KV = re.compile(r"([A-Za-z_]+)=([^\s,]+)")
    BUDGET = re.compile(r"budget (\d+)")

    def __init__(self, port, emit, status, tcp=None, message=None):
        self.want = None if port in (None, "", "auto") else port
        self.tcp = tcp
        self.message = message or (lambda m: None)
        self.cmd_q = collections.deque()                     # command bytes from the page (deque: thread-safe)
        self.levels_until = 0.0                              # poll 'L' until this time.monotonic()
        self.rec_req = None                                  # pending recording (dict), taken by the link thread
        self.cal_req = None                                  # pending calibration capture (dict with a done Event)
        self.cal_wait = None                                 # cal_apply() waiting for the '# cal' replies
        self.cal_device = None                               # the board's calibration from its last '# cal' lines
        self.led_wait = None                                 # set_stealth() waiting for the '# led ... saved' reply
        self.recording = False
        self.connected = False
        super().__init__(emit, status)

    def send(self, cmd):
        """Queue one protocol command byte for the device (written by the link thread)."""
        b = str(cmd).encode("ascii", "ignore")[:1]
        if b:
            self.cmd_q.append(b)

    def haptics(self, duty, hz=0):
        """Queue a manual haptics command: duty = one PWM duty 0-255 per motor in motor_names() order (F, FR, R, BR,
        B, BL, L, FL; the board clamps them to its MOTOR_DUTY_MAX and keeps at most MOTOR_MAX_ON non-zero), hz = 0
        steady or 1-20 pulsed; duty None -> 'A', the detector drives the motors again. The board falls back to the
        detector 3 s after the last 'h', so the page repeats it while a toggle is on. ValueError for a malformed
        duty."""
        if duty is None:
            self.cmd_q.append(b"A")
            return
        d = [max(0, min(255, int(x))) for x in duty]
        if len(d) != len(motor_names()):
            raise ValueError(f"duty must be one value per motor ({', '.join(motor_names())})")
        h = max(0, min(20, int(hz or 0)))
        self.cmd_q.append(("h" + ",".join(str(v) for v in d) + f",{h}\n").encode("ascii"))

    def levels(self, seconds=2.5):
        """Keep the 'L' (per-slot mic levels) polling going for `seconds` more; the page calls this once a second
        while its tap-test display is on, so the polling stops by itself when the page goes away."""
        self.levels_until = time.monotonic() + seconds

    def record(self, label, seconds, holdout=False, tag=""):
        """Queue one raw recording of `seconds` into data/team/<label>/device_<time>[_<tag>][_holdout].wav (the
        naming of scripts/record_device.py; files with _holdout go entirely to the test split). Returns the file
        name. ValueError for bad arguments, RuntimeError when not connected or already recording."""
        if label not in ("drone", "background"):
            raise ValueError("label must be 'drone' or 'background'")
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            raise ValueError("seconds must be a number")
        if not 1.0 <= seconds <= 600.0:
            raise ValueError("seconds must be between 1 and 600")
        tag = re.sub(r"[^A-Za-z0-9_-]+", "_", str(tag or "")).strip("_")[:32]
        if self.recording or self.rec_req is not None:
            raise RuntimeError("a recording is already in progress")
        if not self.connected:
            raise RuntimeError("the device is not connected")
        name = ("device_" + time.strftime("%Y%m%d_%H%M%S") + (f"_{tag}" if tag else "")
                + ("_holdout" if holdout else "") + ".wav")
        path = os.path.abspath(os.path.join(TEAM_DIR, label, name))
        self.rec_req = dict(label=label, seconds=seconds, holdout=bool(holdout), tag=tag, name=name, path=path)
        return name

    def cal_capture(self, direction, seconds=3.0):
        """Blocking (call from a worker thread): have the board record `seconds` (1-8) of all six PDM words ('W<n>':
        it records into PSRAM, then sends every chunk as fast as the link allows) and return the int16 samples
        (n, 6), words w0..w5. ValueError for bad arguments, RuntimeError when not connected, busy, or the board sends
        no 6-word audio (its firmware does not know 'W')."""
        if direction not in calib.DIRECTIONS:
            raise ValueError(f"direction must be one of {calib.DIRECTIONS}")
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            raise ValueError("seconds must be a number")
        if not 1.0 <= seconds <= 8.0:
            raise ValueError("seconds must be between 1 and 8")
        if self.recording or self.rec_req is not None or self.cal_req is not None:
            raise RuntimeError("a recording or capture is already in progress")
        if not self.connected:
            raise RuntimeError("the device is not connected")
        req = dict(direction=direction, seconds=seconds, done=threading.Event(), result=None, error=None)
        self.cal_req = req
        if not req["done"].wait(3.0 * seconds + 20.0):          # recording + transfer at ~256 KB/s over USB
            if self.cal_req is req:
                self.cal_req = None
            raise RuntimeError("calibration capture timed out")
        if req["error"]:
            raise RuntimeError(req["error"])
        return req["result"]

    def cal_apply(self, c_arg, y_arg, timeout=3.0):
        """Blocking: send C<c_arg> and Y<y_arg> (webapp/calibration.firmware_args) and wait for both '# cal ... saved'
        replies; returns cal_device. RuntimeError when not connected / busy / the board rejects or does not answer."""
        if not self.connected:
            raise RuntimeError("the device is not connected")
        if self.recording or self.rec_req is not None or self.cal_req is not None:
            raise RuntimeError("a recording or capture is in progress")
        w = dict(ev=threading.Event(), mics=False, motors=False, error=None)
        self.cal_wait = w
        self.cmd_q.append(f"C{c_arg}\n".encode("ascii"))
        self.cmd_q.append(f"Y{y_arg}\n".encode("ascii"))
        ok = w["ev"].wait(timeout)
        self.cal_wait = None
        if w["error"]:
            raise RuntimeError(f"the board rejected it: {w['error']}")
        if not ok:
            raise RuntimeError(f"no '# cal ... saved' reply within {timeout:g} s (does the board's firmware know C / Y?)")
        return self.cal_device

    def set_stealth(self, on, timeout=3.0):
        """Blocking: send Q1 (stealth: the board's detect LED stays off) or Q0 and wait for its '# led stealth=...
        saved' reply; returns the board's stealth state (bool). RuntimeError when not connected / busy / the board
        rejects it; TimeoutError when it does not answer (firmware without 'Q')."""
        if not self.connected:
            raise RuntimeError("the device is not connected")
        if self.recording or self.rec_req is not None or self.cal_req is not None:
            raise RuntimeError("a recording or capture is in progress")
        w = dict(ev=threading.Event(), stealth=None, error=None)
        self.led_wait = w
        self.cmd_q.append(b"Q1\n" if on else b"Q0\n")
        ok = w["ev"].wait(timeout)
        self.led_wait = None
        if w["error"]:
            raise RuntimeError(f"the board rejected it: {w['error']}")
        if not ok:
            raise TimeoutError(f"no '# led ... saved' reply within {timeout:g} s (does the board's firmware know Q?)")
        return w["stealth"]

    def _cal_capture(self, ser, req, check):
        """One calibration capture on the open link (link thread): 'W<n>', 6-word packets -> memory, 'T'. The board
        records n chunks into PSRAM and sends them at link speed (6 words in real time are over twice what USB CDC
        carries), so the audio arrives complete but late. Messages of kind 'cal_capture': starting | recording (every
        0.5 s, phase "recording" while the board is still listening, then "transfer") | done | error. A broken link
        re-raises (the session reconnects); no audio just fails the request."""
        FS = recorder.FS
        target = int(round(req["seconds"] * FS))
        n_chunks = min(375, -(-target // 1024) + 1)            # the firmware's WCAP_MAX_CHUNKS is 375 (8 s)
        base = dict(type="bench", kind="cal_capture", direction=req["direction"], seconds=req["seconds"])
        self.recording = True
        self.message(dict(base, state="starting", wall=time.time()))
        p = recorder.RawParser()
        chunks, n, dropped, last_idx = [], 0, 0, None
        t0 = time.monotonic(); last_pkt = t0; last_msg = 0.0
        link_err = None
        try:
            ser.reset_input_buffer()
            ser.write(f"W{n_chunks}\n".encode("ascii")); ser.flush()
            t_rec_end = time.monotonic() + n_chunks * 1024 / FS
            while n < target and not self.stop_ev.is_set():
                data = ser.read(ser.in_waiting or 1)
                now = time.monotonic()
                pk = p.feed(data)
                if pk:
                    last_pkt = now
                elif now - last_pkt > 1.5:
                    check()                                       # cable out: raises
                    if now - last_pkt > self.CAL_STALL_S:
                        raise RuntimeError("no 6-word audio from the board for %.0f s after 'W': flash the firmware "
                                           "with calibration support" % (now - last_pkt))
                for idx, x in pk:
                    if x.shape[1] != calib.N_WORDS:
                        raise RuntimeError(f"the board sent {x.shape[1]}-channel packets, not the 6 words of 'W' "
                                           f"(firmware without calibration support?)")
                    if last_idx is not None:
                        gap = (idx - last_idx - 1) & 0xFFFFFFFF
                        if 0 < gap <= recorder.MAX_GAP_FILL:      # dropped chunks: hold the line, keep the timing
                            chunks.append(np.repeat(x[:1], gap * len(x), axis=0)); n += gap * len(x); dropped += gap
                    last_idx = idx
                    chunks.append(x); n += len(x)
                if now - last_msg >= 0.5:
                    last_msg = now
                    self.message(dict(base, state="recording", wall=time.time(), elapsed=round(n / FS, 2),
                                      dropped=dropped, phase="recording" if now < t_rec_end else "transfer"))
            x = np.concatenate(chunks)[:target] if chunks else np.zeros((0, calib.N_WORDS), np.int16)
            if len(x) < target:
                raise RuntimeError(f"capture ended after {len(x) / FS:.1f} s")
            req["result"] = x
            self.message(dict(base, state="done", wall=time.time(), seconds_recorded=round(len(x) / FS, 2),
                              dropped=dropped))
        except RuntimeError as e:
            req["error"] = str(e)
            self.message(dict(base, state="error", wall=time.time(), error=str(e)))
        except Exception as e:                                    # the link itself failed: end the session too
            req["error"] = f"{type(e).__name__}: {e}"
            self.message(dict(base, state="error", wall=time.time(), error=req["error"]))
            link_err = e
        finally:
            self.recording = False
            try:
                ser.write(b"T"); ser.flush()
            except Exception:
                pass
            req["done"].set()
        if link_err is not None:
            raise link_err

    def _record(self, ser, req, check):
        """One raw recording on the open link (link thread): 'R', packets -> WAV, 'T'. Bench messages of kind
        'record' carry state starting | recording (every 0.5 s: elapsed, chunks, dropped, per-channel levels of the
        last half second) | done (totals, clipping / DC verdict) | error; then a background thread runs the
        rules-only pipeline over the clip (first CHECK_MAX_S s) and sends state 'check'. The clip ends early, and
        what was recorded is kept, when the link goes (check() raises: the USB port vanished) or no audio arrives
        for REC_STALL_S; a clip with no audio at all is not left behind."""
        import soundfile as sf
        FS = recorder.FS
        nch = names = None                                   # from the first packet: 4 (B, L, R, T), or 3 (old firmware)
        path, name, seconds = req["path"], req["name"], req["seconds"]
        target = int(round(seconds * FS))
        base = dict(type="bench", kind="record", name=name, label=req["label"], seconds=seconds,
                    path=os.path.relpath(path, ROOT))
        self.recording = True
        self.message(dict(base, state="starting", wall=time.time()))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = total = sec = None                             # the WAV opens on the first packet (its channel count)
        p = recorder.RawParser()
        n = chunks = dropped = restarts = 0; last_idx = None; levels = None
        t0 = time.monotonic(); last_pkt = last_r = t0; last_msg = 0.0; stalled = None
        try:
            ser.reset_input_buffer()
            ser.write(b"R"); ser.flush()
            while n < target and not self.stop_ev.is_set():
                data = ser.read(ser.in_waiting or 1)
                now = time.monotonic()
                pk = p.feed(data)
                if pk:
                    last_pkt = now
                elif now - last_pkt > 3.0 and now - last_r > 3.0:   # no packets: wrong port or stalled link
                    check()                                         # cable out: raises, the clip so far is kept
                    if now - last_pkt > self.REC_STALL_S:           # board there but silent: end it, keep it
                        stalled = now - last_pkt
                        break
                    ser.write(b"R"); ser.flush(); last_r = now
                for idx, x in pk:
                    if out is None:
                        nch = x.shape[1]; names = recorder.ch_names(nch)
                        out = sf.SoundFile(path, "w", samplerate=FS, channels=nch, subtype="PCM_16")
                        total = recorder.Stats(nch); sec = recorder.Stats(nch)
                    elif x.shape[1] != nch:                        # mixed channel counts: not one recording
                        continue
                    if last_idx is not None:
                        gap = (idx - last_idx - 1) & 0xFFFFFFFF
                        if 0 < gap <= recorder.MAX_GAP_FILL:      # dropped chunks: zero-fill, like record_device
                            z = np.zeros((gap * len(x), nch), np.int16)[: max(0, target - n)]
                            out.write(z); total.add(z); n += len(z); dropped += gap
                        elif gap > recorder.MAX_GAP_FILL:
                            restarts += 1
                    last_idx = idx; chunks += 1
                    x = x[: max(0, target - n)]
                    if len(x):
                        out.write(x); total.add(x); sec.add(x); n += len(x)
                if now - last_msg >= 0.5:
                    last_msg = now
                    if sec is not None and sec.n:
                        rms, dc = sec.rms_ac(), sec.dc()
                        levels = [dict(ch=names[c], rms=recorder.dbfs(rms[c]), peak=recorder.dbfs(sec.pk[c]),
                                       dc=float(dc[c])) for c in range(nch)]
                        sec = recorder.Stats(nch)
                    self.message(dict(base, state="recording", wall=time.time(), elapsed=n / FS, chunks=chunks,
                                      dropped=dropped, no_packets=now - last_pkt > 3.0, levels=levels))
        except Exception as e:
            self.message(dict(base, state="error", wall=time.time(), error=f"{type(e).__name__}: {e}",
                              seconds_recorded=n / FS))
            raise
        finally:
            self.recording = False
            if out is not None:
                out.close()                                  # writes the WAV header's lengths: never skip it
            if n == 0 and out is not None:
                try:
                    os.remove(path)                          # header only, no audio: not a training clip
                except OSError:
                    pass
            try:
                ser.write(b"T"); ser.flush()
            except Exception:
                pass
        if n == 0:
            self.message(dict(base, state="error", wall=time.time(), seconds_recorded=0.0,
                              error=f"no audio from the board for {stalled or 0:.0f} s, nothing saved"))
            return
        rms, dc = total.rms_ac(), total.dc()
        lv = [dict(ch=names[c], rms=recorder.dbfs(rms[c]), peak=recorder.dbfs(total.pk[c]), dc=float(dc[c]))
              for c in range(nch)]
        clip = [names[c] for c in range(nch) if recorder.dbfs(total.pk[c]) > -1.0]
        dcb = [names[c] for c in range(nch) if abs(dc[c]) > 0.01]
        dead = [names[c] for c in range(nch) if rms[c] < 1e-5]
        verdict = [("CLIPPING on " + ",".join(clip)) if clip else "no clipping",
                   ("DC offset > 1 % on " + ",".join(dcb)) if dcb else "DC ok"]
        if dead:
            verdict.append("SILENT channel(s) " + ",".join(dead))
        if stalled:
            verdict.append(f"STOPPED EARLY: no audio from the board for {stalled:.0f} s")
        self.message(dict(base, state="done", wall=time.time(), seconds_recorded=n / FS, chunks=chunks,
                          dropped=dropped, restarts=restarts, skipped=p.skipped, levels=lv,
                          verdict="; ".join(verdict), wall_s=round(time.monotonic() - t0, 1)))
        if n:
            threading.Thread(target=self._check, args=(path, base), name="record-check", daemon=True).start()

    def _check(self, path, base):
        """Rules-only detector over the clip just recorded (what `record_device.py --mono-check` prints), so the
        bench shows at once whether a drone clip is detectable and a background clip is quiet."""
        try:
            import soundfile as sf
            if ROOT not in sys.path:
                sys.path.insert(0, ROOT)
            from droneears.pipeline import Pipeline, summarise
            y, fs = sf.read(path, dtype="float32", always_2d=True, frames=int(CHECK_MAX_S * recorder.FS))
            t1 = time.monotonic()
            res = Pipeline().run(y.T)
            s = summarise(res)
            self.message(dict(base, state="check", wall=time.time(), frames=s["frames"],
                              checked_s=len(y) / fs, detected_frac=s["detected_frac"],
                              first_detect_s=s.get("first_detect_s"), theta_median=s.get("theta_median_when_detected"),
                              max_hits=max((r["hits"] for r in res), default=0), check_s=round(time.monotonic() - t1, 1)))
        except Exception as e:
            self.message(dict(base, state="check", wall=time.time(), error=f"{type(e).__name__}: {e}"))

    def _pick(self):
        if self.want:
            return self.want if os.path.exists(self.want) else None
        c = telemetry.candidate_ports()
        return c[0] if c else None

    def _comment(self, line):
        kv = {k: v for k, v in self.KV.findall(line)}
        body = line.lstrip("# ").strip()
        upd = dict(last_comment=body)
        if body.startswith("wifi"):                         # '# wifi mode=AP ssid=HEADBAND ip=... client=...'
            for k in ("mode", "ssid", "ip", "port", "mdns", "client", "sta"):
                if k in kv:
                    upd["wifi_" + k] = kv[k]
            self.status(**upd)
            return
        if body.startswith("levels"):                       # '# levels w0 rms=.. peak=.. dc=.. ... w5 ... | B=w0 L=w3 R=w2 T=w5'
            words, chmap, hints = telemetry.levels_parse(line)
            chmap = {k: v for k, v in chmap.items() if k in "FBLRT"}  # channel 0 is F= when it is the front mic
            self.message(dict(type="bench", kind="levels", wall=time.time(), map=chmap, hints=hints,
                              words=[dict(w=w, slot=telemetry.LEVELS_SLOT.get(w, f"w{w}"), rms=words[w][0],
                                          peak=words[w][1], dc=words[w][2]) for w in sorted(words)]))
            return
        if body.startswith("dsp "):                          # '# dsp frames=.. us/frame wait=.. ... total=.. dma_ovf=.. (budget 21333)'
            d = {}
            for k in ("frames", "wait", "spec", "mel", "det", "bear", "total", "dma_ovf"):
                if k in kv:
                    try:
                        d[k] = int(float(kv[k]))
                    except ValueError:
                        pass
            m = self.BUDGET.search(body)
            d["budget"] = int(m.group(1)) if m else None
            self.message(dict(type="bench", kind="dsp", wall=time.time(), **d))
            self.status(dsp_total_us=d.get("total"), dsp_budget_us=d.get("budget"), dma_ovf=d.get("dma_ovf"))
            return
        if body.startswith("cal"):                           # '# cal mics B=w0 (or F=w0) ... motors M1=0.0 ... src=nvs' / '# cal: bad ...'
            d = calib.parse_cal_line(body)
            w = self.cal_wait
            if "error" in d:
                if w is not None:
                    w["error"] = d["error"]; w["ev"].set()
                self.message(dict(type="bench", kind="cal", wall=time.time(), text=body, error=d["error"],
                                  device=self.cal_device))
                return
            dev = dict(self.cal_device or {})
            for k in ("mics", "motors", "src"):
                if k in d:
                    dev[k] = d[k]
            self.cal_device = dev
            if w is not None and d.get("saved"):
                w["mics"] = w["mics"] or "mics" in d
                w["motors"] = w["motors"] or "motors" in d
                if w["mics"] and w["motors"]:
                    w["ev"].set()
            self.status(cal_device=dev)
            self.message(dict(type="bench", kind="cal", wall=time.time(), text=body, device=dev))
            return
        if body.startswith("led"):                           # '# led stealth=on|off [saved]' / '# led: bad ...'
            w = self.led_wait
            if body.startswith("led:"):
                if w is not None:
                    w["error"] = body[4:].strip(); w["ev"].set()
                self.message(dict(type="bench", kind="led", wall=time.time(), text=body, error=body[4:].strip()))
                return
            stealth = {"on": True, "off": False}.get(kv.get("stealth"))
            if stealth is None:
                return
            self.status(led_stealth=stealth)                 # before waking set_stealth: /api/status agrees at once
            self.message(dict(type="bench", kind="led", wall=time.time(), text=body, stealth=stealth))
            if w is not None and body.endswith("saved"):
                w["stealth"] = stealth; w["ev"].set()
            return
        if body.startswith("motor test"):
            self.message(dict(type="bench", kind="motor_test", wall=time.time(), text=body))
            return
        if body.startswith("haptics"):                       # '# haptics manual F=0 FR=0 R=190 ... pulse_hz=0 max=190 ...'
            self.message(dict(type="bench", kind="haptics", wall=time.time(), text=body,
                              manual=body.startswith("haptics manual"), max=kv.get("max")))
            return
        if body.startswith("boot "):                         # '# boot free_heap=.. psram_free=.. psram_bufs=3/3 tasks_failed=0x00'
            self.status(boot_line=body, psram_bufs=kv.get("psram_bufs"), tasks_failed=kv.get("tasks_failed"))
            return
        if body.startswith("status"):
            for k in ("frame", "dropped", "wifi_dropped", "free_heap", "psram_free", "dsp_stack_min", "pdm_us"):
                if k in kv:
                    try:
                        upd["dev_" + k if k == "frame" else k] = int(float(kv[k].rstrip("s")))
                    except ValueError:
                        pass
            if "batt_v" in kv:                               # '... batt_v=3.92' (pins.h firmware on)
                try:
                    v = float(kv["batt_v"])
                    upd["batt_v"] = round(v, 2) if v > 0 else None
                except ValueError:
                    pass
            if "mode" in kv:
                upd["mode"] = kv["mode"]
            upd["status_line"] = body
        elif "chunk_us" in kv:
            try:
                upd["chunk_us"] = int(kv["chunk_us"])
            except ValueError:
                pass
        elif not body.startswith("host:"):
            upd["firmware"] = body                           # '# headband fw 1 fs=48000 ...' (or any banner)
        self.status(**upd)

    def _run_tcp(self):
        host, port = telemetry.parse_tcp(self.tcp)
        label = f"{host or telemetry.TCP_HOST}:{port}"
        while not self.stop_ev.is_set():
            try:
                link = telemetry.open_tcp(host, port, timeout=self.TCP_TIMEOUT)
            except OSError as e:
                self.status(connected=False, transport="tcp", port=label, host=host or telemetry.TCP_HOST,
                            error=f"connect failed: {e}")
                self.stop_ev.wait(1.0)
                continue
            self.status(connected=True, transport="tcp", port=link.name, host=link.host, peer=link.peer, baud=None,
                        error=None)
            self._session(link, lambda: None)
            self.stop_ev.wait(1.0)

    def _session(self, ser, check):
        """One connection: 'T', then parse until the link fails; check() raises if the link vanished silently."""
        p = telemetry.TelemetryParser()
        try:
            ser.reset_input_buffer()
            ser.write(b"T"); ser.write(b"K"); ser.write(b"Q\n"); ser.flush()   # telemetry, calibration, stealth
            last_s = last_t = last_l = time.monotonic()
            self.cmd_q.clear()
            self.connected = True
            while not self.stop_ev.is_set():
                if self.rec_req is not None:                 # bench panel: record a clip, then back to telemetry
                    req, self.rec_req = self.rec_req, None
                    while self.cmd_q:                        # the page's queued commands first ('A': motors off
                        ser.write(self.cmd_q.popleft())      # before the clip, not 3 s into it)
                    self._record(ser, req, check)
                    p = telemetry.TelemetryParser()          # the raw bytes are not lines; start clean on 'T'
                    last_s = last_t = time.monotonic()
                if self.cal_req is not None:                 # calibration tab: a 'W' capture of all six words
                    req, self.cal_req = self.cal_req, None
                    while self.cmd_q:
                        ser.write(self.cmd_q.popleft())
                    self._cal_capture(ser, req, check)
                    p = telemetry.TelemetryParser()
                    last_s = last_t = time.monotonic()
                data = ser.read(ser.in_waiting or 1)
                for kind, x in p.feed(data):
                    if kind == "row":
                        self.emit(device_row_to_frame(x))
                    elif kind == "comment":
                        self._comment(x)
                now = time.monotonic()
                while self.cmd_q:                            # bench panel: 'M', 'S', ...
                    ser.write(self.cmd_q.popleft()); ser.flush()
                if now < self.levels_until and now - last_l >= self.LEVELS_EVERY:
                    ser.write(b"L"); ser.flush(); last_l = now
                if p.need_header and now - last_t > 2.0:
                    ser.write(b"T"); ser.flush(); last_t = now
                if now - last_s >= self.STATUS_EVERY:
                    ser.write(b"S"); ser.flush(); last_s = now
                    self.status(bad_lines=p.bad, headers=p.headers)
                if not data:
                    check()
        except Exception as e:                               # SerialException, OSError/ConnectionError: unplugged,
            self.status(connected=False, error=f"{type(e).__name__}: {e}")   # reset, Wi-Fi dropped
        finally:
            self.connected = False
            self.rec_req = None
            req, self.cal_req = self.cal_req, None            # a capture queued as the link went: fail it now
            if req is not None:
                req["error"] = "the device disconnected"; req["done"].set()
            if self.cal_wait is not None:
                self.cal_wait["error"] = "the device disconnected"; self.cal_wait["ev"].set()
            if self.led_wait is not None:
                self.led_wait["error"] = "the device disconnected"; self.led_wait["ev"].set()
            try:
                ser.close()
            except Exception:
                pass

    def run(self):
        if self.tcp is not None:
            return self._run_tcp()
        while not self.stop_ev.is_set():
            port = self._pick()
            if port is None:
                self.status(connected=False, port=self.want or "searching for /dev/cu.usbmodem* ...")
                self.stop_ev.wait(1.0)
                continue
            try:
                ser = telemetry.open_serial(port)
            except Exception as e:
                self.status(connected=False, port=port, error=f"open failed: {e}")
                self.stop_ev.wait(1.0)
                continue
            self.status(connected=True, transport="serial", port=port, error=None, baud=ser.baudrate)

            def check(port=port):
                if not os.path.exists(port):
                    raise OSError("port vanished")
            self._session(ser, check)
            self.stop_ev.wait(1.0)
