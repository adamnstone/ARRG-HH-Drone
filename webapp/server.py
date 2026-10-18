"""Live web dashboard for the headband drone detector: one aiohttp process serving the page and a WebSocket.

Sources (pick one; with none: the device over USB if a serial port is found, else over Wi-Fi if headband.local:5000
answers within 2 s, else --demo):
  --port [PORT]        the headband over USB serial (auto-detect like scripts/telemetry.py); sends 'T' on connect,
                       'S' every 2 s for the status line, reconnects if the cable drops
  --tcp [HOST[:PORT]]  the headband over Wi-Fi (same protocol on TCP; default headband.local:5000, falling back to
                       192.168.4.1:5000, the board's own access point); reconnects if the link drops
  --replay LOG.csv     a log from `scripts/telemetry.py --log`, replayed in real time (by t_ms), looping
  --simulate AUDIO     droneears.pipeline.Pipeline over a 1- or 4-channel (B, L, R, T) audio file, real time, looping
  --demo               the same pipeline over a scripted synthetic scene (right -> behind, climbing -> overhead ->
                       down to the front-left)

Endpoints:
  GET /              the dashboard (webapp/static/)
  GET /ws            WebSocket: {"type":"hello",...} once, then one frame object per pipeline frame
                     ({frame,t_ms,f0,score,strength,flat,hits,ml,detected,theta,conf,elev,overhead,batt_v,
                     hap:[F,FR,R,BR,B,BL,L,FL] (applied weights; overhead = the rotating triplet),bstat,mused,src};
                     theta / elev null when the frame has no azimuth / elevation; bstat = bearing status 0 none or
                     unreliable, 1 ambiguous mirror, 2 three mics (azimuth only, or elevation assumed above), 3 full
                     3D; mused = mics in this frame's fit, B=1 L=2 R=4 T=8; both null from older sources),
                     {"type":"status",...} every second, {"type":"event",kind,t,theta,...} on state changes,
                     {"type":"bench",kind:"levels"|"dsp"|"motor_test"|"haptics"|"cal"|"cal_capture"|"led",...} for
                     the board's bench replies and the calibration tab's captures.
                     The page may send {"type":"cmd","cmd":"M"|"S"|"L"|"T"} (one protocol byte to the device:
                     motor test, status, levels, telemetry), {"type":"levels"} (keep the per-slot mic level
                     polling going for 2.5 s more) and {"type":"haptics","duty":[F,FR,R,BR,B,BL,L,FL],"hz":N} (manual motors:
                     PWM duty 0-255 each, steady or pulsed at N Hz; duty null = back to the detector; the board
                     reverts by itself 3 s after the last one, so the page repeats it); all are ignored unless the
                     source is the device.
  GET /api/status    the current status object (same as the WebSocket's status message; led_stealth = the board's
                     stealth mode from its last '# led' line, null until one arrives)
  POST /api/led      {"stealth": true|false}: stealth mode, the board's detect LED stays off ('Q1' / 'Q0', kept in its
                     NVS); waits for the board's '# led ... saved' reply -> {"ok": true, "stealth": bool}. 409 unless
                     the source is the device and it is connected, 504 when the board does not answer
  GET /api/config    thresholds and geometry from droneears/config.py (the page draws its threshold lines from it)
  POST /api/record   {"label":"drone"|"background","seconds":N,"holdout":bool,"tag":str}: record a raw 4-channel
                     clip from the device into data/team/<label>/ (what scripts/train.py trains on); progress and
                     the detector check arrive as {"type":"bench","kind":"record",...} WebSocket messages
  GET /api/recordings  the clips in data/team/ (per file and per label: count, minutes, holdout share)
  GET /api/clip?label=drone&name=<file>&mix=stereo|mono|B|L|R|T   a 16-bit WAV of that clip for listening:
                     stereo = channels L and R, mono = the mean of all channels, or one channel (named by the
                     file's channel count: 4 = B, L, R, T; 3 = the old firmware's L, R, B); peak-normalised
                     to -3 dBFS (the stored file is untouched). Finder / QuickTime will not play the 3-channel
                     originals, browsers will play these.
  Calibration (webapp/calibration.py; which DMA word is which mic position, which hole each motor sits in):
  GET /api/geometry  the headband geometry (webapp/static/headband_geometry.json from the STL: outline, holes, mic
                     sites), or an 8-hole placeholder until that file exists
  GET /api/calibration  {"calibration": data/calibration.json or the defaults, "device": the board's calibration from
                     its last '# cal' lines (or null), "device_matches": bool, "captures": [directions recorded],
                     "error": why the file could not be used (or null)}
  POST /api/calibration  body = the calibration object (or {"calibration": ...}): validated and saved
  POST /api/calibration/apply  send the saved calibration to the board (C<b>,<l>,<r>,<t>: the DMA words of the
                     back, left, right and top mics, -1 = not connected, or C<f>,<l>,<r>,<t>,F with a front mic in
                     channel 0 instead of the back one; Y<m1..m8>; kept in its NVS) and wait for its '# cal ... saved'
                     replies
  POST /api/calibration/capture {"direction":"front"|"right"|"left"|"back"|"top","seconds":3}  record all six
                     PDM words ('W<n>', 1-8 s; the board buffers them and sends them complete, which takes ~2.5x the
                     recording time over USB) while someone claps from that side; kept for solve and written to
                     data/calibration/captures/<direction>.wav; returns per-word levels / mirror / live flags and
                     whether a clap was found ({"type":"bench","kind":"cal_capture",...} carries the progress)
  DELETE /api/calibration/capture  forget the captures
  POST /api/calibration/solve  [{"slot_order": "left_first"|"right_first", "mic_words": {...}}]  assign words to
                     the positions (front or back, left, right, top) from the front / right / left (+ back, top)
                     captures: {"mic_words", "confidence", "notes", "live_words", "mirrors", "per_direction",
                     "slot_order"}; a line with one mic reads it in both words, and its position goes to a word with
                     a header under slot_order, preferring a header connected in mic_words (both default to the
                     saved calibration's)
"""
import argparse, asyncio, collections, glob, json, os, sys, threading, time, webbrowser

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from aiohttp import web, WSMsgType  # noqa: E402
from droneears import config as C  # noqa: E402
from webapp import sources  # noqa: E402
from webapp import calibration  # noqa: E402

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
SRC_LABEL = {"serial": "serial", "replay": "replay", "simulate": "simulate", "demo": "demo"}
CLIENT_QUEUE = 94            # ~2 s of frames per client; a slower client loses its oldest frames (coalescing)
HISTORY = 20 * 47            # frames replayed to a client on connect, so its charts are full at once
BATT_EMPTY_V, BATT_FULL_V = 3.30, 4.20   # firmware/esp32s3/pins.h BATT_EMPTY_V / BATT_FULL_V (percent is linear)


class Hub:
    """Owns the latest state, derives events from frame transitions, and fans out to WebSocket clients.
    All methods run on the asyncio loop; source threads reach it through emit_threadsafe/status_threadsafe."""

    def __init__(self, kind, loop):
        self.kind, self.loop = kind, loop
        self.t_start = time.time()
        self.clients = set()
        self.history = collections.deque(maxlen=HISTORY)
        self.events = collections.deque(maxlen=20)
        self.stamps = collections.deque()
        self.n_frames = 0
        self.dropped_ws = 0
        self.last = None
        self.det = False; self.ovh = False
        self.det_t = None; self.ovh_t = None
        self.ev_start = None          # t_ms of the start of the current run of evidence (hits > 0) while idle
        self.src = dict(source=SRC_LABEL[kind], transport=None, connected=False, port=None, file=None, firmware=None,
                        chunk_us=None, free_heap=None, dropped=None, dev_frame=None, error=None, led_stealth=None)

    # ---- thread-safe entry points
    def emit_threadsafe(self, frame):
        self.loop.call_soon_threadsafe(self.publish, frame)

    def status_threadsafe(self, **kw):
        self.loop.call_soon_threadsafe(self.src.update, kw)

    def message_threadsafe(self, msg):
        """A one-off message for every client (bench replies); droppable like frames when a client lags."""
        self.loop.call_soon_threadsafe(self._broadcast, msg, True)

    # ---- state
    def status(self):
        now = time.time()
        while self.stamps and now - self.stamps[0] > 1.0:
            self.stamps.popleft()
        s = dict(self.src)
        last_age = None if self.last is None else round(now - self._last_wall, 2)
        s.update(type="status", frames_per_s=len(self.stamps), uptime_s=round(now - self.t_start, 1),
                 frames=self.n_frames, clients=len(self.clients), ws_dropped=self.dropped_ws,
                 last_frame_age_s=last_age, frame=None if self.last is None else self.last["frame"],
                 device_uptime_s=None if self.last is None else round(self.last["t_ms"] / 1000.0, 1))
        if self.kind == "serial":
            s["connected"] = bool(self.src.get("connected"))
        return s

    def _event(self, kind, t_ms, theta, **extra):
        ev = dict(type="event", kind=kind, t=round(t_ms / 1000.0, 2), wall=time.time(),
                  theta=None if theta is None else round(theta, 1), **extra)
        self.events.append(ev)
        self._broadcast(ev, droppable=False)

    def publish(self, f):
        now = time.time()
        if self.last is not None and f["t_ms"] < self.last["t_ms"] - 1000:   # device reset: start over
            self.det = self.ovh = False; self.det_t = self.ovh_t = self.ev_start = None
            self.history.clear()
        self.last = f; self._last_wall = now
        self.n_frames += 1
        self.stamps.append(now)
        t = f["t_ms"]
        if not self.det:
            if f["hits"] <= 0:
                self.ev_start = None
            elif self.ev_start is None:
                self.ev_start = t
        det, ovh = bool(f["detected"]), bool(f["overhead"]) and bool(f["detected"])
        if det and not self.det:
            lat = None if self.ev_start in (None, t) else round((t - self.ev_start) / 1000.0, 2)   # None: alarm already on at the first frame
            self.det_t = t
            self._event("detect_on", t, f["theta"], latency_s=lat, elev=f.get("elev"))
        if ovh and not self.ovh:
            self.ovh_t = t
            self._event("overhead_on", t, f["theta"], elev=f.get("elev"))
        if self.ovh and not ovh:
            self._event("overhead_off", t, f["theta"],
                        duration_s=None if self.ovh_t is None else round((t - self.ovh_t) / 1000.0, 2))
        if self.det and not det:
            self._event("detect_off", t, self.last_theta,
                        duration_s=None if self.det_t is None else round((t - self.det_t) / 1000.0, 2))
            self.ev_start = None
        if det and f["theta"] is not None and not ovh and f["hits"] >= C.PERSIST_ON:
            self.last_theta = f["theta"]          # last bearing while the evidence was strong (not the fade-out)
        self.det, self.ovh = det, ovh
        self.history.append(f)
        self._broadcast(f, droppable=True)

    last_theta = None
    _last_wall = 0.0

    def hello(self):
        return dict(type="hello", source=SRC_LABEL[self.kind], config=config_json(),
                    history=list(self.history), events=list(self.events), status=self.status(),
                    det_t=self.det_t, ovh_t=self.ovh_t)

    def _broadcast(self, msg, droppable):
        data = json.dumps(msg, separators=(",", ":"))
        for c in list(self.clients):
            c.push(data, droppable)
            if c.dropped_now:
                self.dropped_ws += c.dropped_now; c.dropped_now = 0


class Client:
    """Per-WebSocket outbound queue. Frames are droppable (oldest first when the client lags); events/status not."""

    def __init__(self, ws):
        self.ws = ws
        self.q = collections.deque()
        self.n_frames = 0
        self.dropped_now = 0
        self.wake = asyncio.Event()

    def push(self, data, droppable):
        if droppable:
            if self.n_frames >= CLIENT_QUEUE:
                for i, (d, drop) in enumerate(self.q):
                    if drop:
                        del self.q[i]; self.n_frames -= 1; self.dropped_now += 1
                        break
            self.n_frames += 1
        self.q.append((data, droppable))
        self.wake.set()

    async def pump(self):
        while not self.ws.closed:
            await self.wake.wait()
            self.wake.clear()
            while self.q and not self.ws.closed:
                data, drop = self.q.popleft()
                if drop:
                    self.n_frames -= 1
                await self.ws.send_str(data)


def config_json():
    return dict(fs=C.FS, hop=C.HOP, frames_per_s=C.FRAMES_PER_SEC, harm_thresh_db=C.HARM_THRESH_DB,
                persist_on=C.PERSIST_ON, persist_off=C.PERSIST_OFF, persist_frames=C.PERSIST_FRAMES,
                ml_thresh=C.ML_THRESH, f0_min=C.F0_MIN, f0_max=C.F0_MAX, flat_min=C.FLAT_MIN,
                flat_max=C.FLAT_THRESH, overhead_elev_on=C.OVERHEAD_ELEV_ON, overhead_elev_off=C.OVERHEAD_ELEV_OFF,
                motor_angles=C.MOTOR_ANGLES, motor_names=list(C.MOTOR_ANGLES), motor_max_on=sources.MOTOR_MAX_ON,
                mic_pos=C.MIC_POS.tolist(), mic_names=C.MIC_NAMES, n_mics=len(C.MIC_NAMES),
                front_pos=C.FRONT_POS.tolist() if hasattr(C, "FRONT_POS") else None,    # channel 0 when it is the front mic
                batt_empty_v=BATT_EMPTY_V, batt_full_v=BATT_FULL_V,
                chunk_us_expect=int(round(1e6 * C.HOP / C.FS)))


# ---- HTTP handlers
async def index(request):
    return web.FileResponse(os.path.join(STATIC, "index.html"), headers={"Cache-Control": "no-cache"})


async def api_status(request):
    hub = request.app["hub"]
    s = hub.status()
    s["last"] = hub.last
    s["events"] = list(hub.events)[-10:]
    return web.json_response(s)


async def api_config(request):
    return web.json_response(config_json())


async def api_record(request):
    src = request.app.get("source")
    if not isinstance(src, sources.SerialSource):
        return web.json_response(dict(ok=False, error="recording needs the device: start with --port or --tcp"),
                                 status=409)
    try:
        d = await request.json()
        if not isinstance(d, dict):
            raise ValueError
    except Exception:
        return web.json_response(dict(ok=False, error="bad JSON body"), status=400)
    try:
        name = src.record(d.get("label"), d.get("seconds", 30), bool(d.get("holdout")), d.get("tag", ""))
    except ValueError as e:
        return web.json_response(dict(ok=False, error=str(e)), status=400)
    except RuntimeError as e:
        return web.json_response(dict(ok=False, error=str(e)), status=409)
    return web.json_response(dict(ok=True, name=name))


def recordings_summary():
    """The clips under data/team/{drone,background}: per file (name, label, holdout, seconds, mtime; newest first,
    at most 50) and per label (clips, seconds, holdout clips / seconds). Lengths come from the file header
    (soundfile): WAV / FLAC / OGG; compressed phone clips (m4a) count but have no length here."""
    import soundfile as sf
    team = sources.TEAM_DIR
    pats = ("*.wav", "*.m4a", "*.mp3", "*.aac", "*.flac", "*.ogg", "*.mp4", "*.WAV", "*.M4A")
    files, summary = [], {}
    for label in ("drone", "background"):
        paths = sorted(set(p for pat in pats for p in glob.glob(os.path.join(team, label, pat))))
        tot = dict(clips=0, seconds=0.0, holdout_clips=0, holdout_seconds=0.0, unknown_length=0)
        for p in paths:
            secs = None
            try:
                info = sf.info(p); secs = info.frames / info.samplerate
            except Exception:
                tot["unknown_length"] += 1
            hold = "_holdout" in os.path.basename(p)
            files.append(dict(name=os.path.basename(p), label=label, holdout=hold, seconds=secs,
                              mtime=os.path.getmtime(p)))
            tot["clips"] += 1; tot["seconds"] += secs or 0.0
            if hold:
                tot["holdout_clips"] += 1; tot["holdout_seconds"] += secs or 0.0
        summary[label] = tot
    files.sort(key=lambda f: -f["mtime"])
    return dict(files=files[:50], summary=summary, folder=os.path.relpath(team, ROOT))


async def api_recordings(request):
    return web.json_response(await asyncio.get_running_loop().run_in_executor(None, recordings_summary))


def clip_mix_wav(label, name, mix):
    """Bytes of a 16-bit WAV listening mix of data/team/<label>/<name> (see the module docstring). Only files
    inside the two training folders are served (label whitelisted, name must be a bare file name)."""
    import io
    import numpy as np
    import soundfile as sf
    if label not in ("drone", "background") or os.path.basename(name) != name or not name:
        raise ValueError("bad label or name")
    path = os.path.join(sources.TEAM_DIR, label, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(name)
    y, fs = sf.read(path, dtype="float32", always_2d=True)        # (n, ch)
    ch = {n: i for i, n in enumerate(sources.recorder.ch_names(y.shape[1]))}   # 4 ch: B, L, R, T; 3 ch: L, R, B
    if mix == "mono" or y.shape[1] == 1:
        out = y.mean(axis=1, keepdims=True)
    elif mix in ch and y.shape[1] > ch[mix]:
        out = y[:, ch[mix]:ch[mix] + 1]
    elif "L" in ch and "R" in ch:                                  # stereo: L, R
        out = y[:, [ch["L"], ch["R"]]]
    else:                                                          # unknown layout: the first two channels
        out = y[:, :2]
    out = out - out.mean(axis=0, keepdims=True)                    # drop any DC before normalising
    peak = float(np.abs(out).max()) if out.size else 0.0
    if peak > 0:
        out = out * (10 ** (-3 / 20) / peak)                       # -3 dBFS peak, so quiet rooms are audible
    buf = io.BytesIO()
    sf.write(buf, out, fs, format="WAV", subtype="PCM_16")
    return buf.getvalue()


async def api_clip(request):
    q = request.query
    try:
        data = await asyncio.get_running_loop().run_in_executor(
            None, clip_mix_wav, q.get("label", ""), q.get("name", ""), q.get("mix", "stereo"))
    except ValueError as e:
        return web.json_response(dict(ok=False, error=str(e)), status=400)
    except FileNotFoundError:
        return web.json_response(dict(ok=False, error="no such clip"), status=404)
    return web.Response(body=data, content_type="audio/wav", headers={"Cache-Control": "no-cache"})


# ---- calibration
def _json_err(msg, status):
    return web.json_response(dict(ok=False, error=msg), status=status)


async def _body(request):
    try:
        d = await request.json()
    except Exception:
        d = None
    return d if isinstance(d, dict) else None


def _device_src(app):
    src = app.get("source")
    return src if isinstance(src, sources.SerialSource) else None


async def api_led(request):
    src = _device_src(request.app)
    if src is None:
        return _json_err("stealth mode needs the device: start with --port or --tcp", 409)
    d = await _body(request)
    if d is None or not isinstance(d.get("stealth"), bool):
        return _json_err('want {"stealth": true|false}', 400)
    try:
        on = await asyncio.get_running_loop().run_in_executor(None, src.set_stealth, d["stealth"])
    except TimeoutError as e:
        return _json_err(str(e), 504)
    except RuntimeError as e:
        return _json_err(str(e), 409)
    return web.json_response(dict(ok=True, stealth=on))


async def api_geometry(request):
    return web.json_response(calibration.load_geometry())


async def api_calibration_get(request):
    cal, err = calibration.load_calibration()
    src = _device_src(request.app)
    dev = src.cal_device if src is not None else None
    return web.json_response(dict(calibration=cal, device=dev, device_matches=calibration.device_matches(dev, cal),
                                  captures=sorted(request.app["cal_captures"]), error=err))


async def api_calibration_post(request):
    d = await _body(request)
    if d is None:
        return _json_err("bad JSON body", 400)
    if isinstance(d.get("calibration"), dict):
        d = d["calibration"]
    try:
        cal = calibration.save_calibration(d)
    except ValueError as e:
        return _json_err(str(e), 400)
    return web.json_response(dict(ok=True, calibration=cal))


async def api_calibration_apply(request):
    src = _device_src(request.app)
    if src is None:
        return _json_err("applying needs the device: start with --port or --tcp", 409)
    cal, err = calibration.load_calibration()
    if err:
        return _json_err(err, 409)
    c_arg, y_arg = calibration.firmware_args(cal)
    try:
        dev = await asyncio.get_running_loop().run_in_executor(None, src.cal_apply, c_arg, y_arg)
    except RuntimeError as e:
        return _json_err(str(e), 504 if "reply" in str(e) else 409)
    return web.json_response(dict(ok=True, device=dev, sent=dict(C=c_arg, Y=y_arg),
                                  device_matches=calibration.device_matches(dev, cal)))


async def api_calibration_capture(request):
    src = _device_src(request.app)
    if src is None:
        return _json_err("capturing needs the device: start with --port or --tcp", 409)
    d = await _body(request)
    if d is None:
        return _json_err("bad JSON body", 400)
    direction, seconds = d.get("direction"), d.get("seconds", 3)
    loop = asyncio.get_running_loop()
    try:
        x = await loop.run_in_executor(None, src.cal_capture, direction, seconds)
    except ValueError as e:
        return _json_err(str(e), 400)
    except RuntimeError as e:
        return _json_err(str(e), 409 if ("progress" in str(e) or "not connected" in str(e)) else 504)

    def keep():
        calibration.save_capture_wav(direction, x)
        return calibration.capture_summary(x)
    summary = await loop.run_in_executor(None, keep)
    request.app["cal_captures"][direction] = x
    return web.json_response(dict(ok=True, direction=direction, seconds=round(len(x) / calibration.FS, 2),
                                  words=summary["words"], event=summary["event"],
                                  captures=sorted(request.app["cal_captures"])))


async def api_calibration_capture_delete(request):
    request.app["cal_captures"].clear()
    for d in calibration.DIRECTIONS:
        try:
            os.remove(os.path.join(calibration.CAPTURE_DIR, f"{d}.wav"))
        except OSError:
            pass
    return web.json_response(dict(ok=True, captures=[]))


async def api_calibration_solve(request):
    caps = dict(request.app["cal_captures"])
    d = (await _body(request) if request.can_read_body else None) or {}
    saved = calibration.load_calibration()[0]             # the page sends its current map; else the saved file's
    order = d.get("slot_order") if d.get("slot_order") in calibration.SLOT_ORDERS else saved["slot_order"]
    current = d.get("mic_words") if isinstance(d.get("mic_words"), dict) else saved["mic_words"]
    try:
        res = await asyncio.get_running_loop().run_in_executor(None, calibration.solve, caps, order, current)
    except ValueError as e:
        return _json_err(str(e), 400)
    return web.json_response(res)


def handle_client_msg(app, text):
    """Bench panel commands from the page (see the module docstring); no-ops unless the source is the device."""
    try:
        d = json.loads(text)
    except ValueError:
        return
    src = app.get("source")
    if not isinstance(src, sources.SerialSource) or not isinstance(d, dict):
        return
    if d.get("type") == "cmd" and d.get("cmd") in ("M", "S", "L", "T"):
        src.send(d["cmd"])
    elif d.get("type") == "levels":
        src.levels(2.5)
    elif d.get("type") == "haptics":
        try:
            src.haptics(d.get("duty"), d.get("hz", 0))
        except (TypeError, ValueError):
            pass


async def ws_handler(request):
    hub = request.app["hub"]
    ws = web.WebSocketResponse(heartbeat=15, max_msg_size=1 << 16)
    await ws.prepare(request)
    await ws.send_str(json.dumps(hub.hello(), separators=(",", ":")))
    c = Client(ws)
    hub.clients.add(c)
    pump = asyncio.create_task(c.pump())
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                handle_client_msg(request.app, msg.data)
            elif msg.type == WSMsgType.ERROR:
                break
    finally:
        hub.clients.discard(c)
        pump.cancel()
    return ws


async def status_ticker(app):
    hub = app["hub"]
    while True:
        await asyncio.sleep(1.0)
        hub._broadcast(hub.status(), droppable=False)


@web.middleware
async def no_cache(request, handler):
    resp = await handler(request)
    if request.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


def make_app(args):
    app = web.Application(middlewares=[no_cache])
    app["args"] = args
    app["cal_captures"] = {}                 # direction -> int16 (n, 6) from POST /api/calibration/capture
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/api/status", api_status)
    app.router.add_get("/api/config", api_config)
    app.router.add_post("/api/record", api_record)
    app.router.add_get("/api/recordings", api_recordings)
    app.router.add_get("/api/clip", api_clip)
    app.router.add_post("/api/led", api_led)
    app.router.add_get("/api/geometry", api_geometry)
    app.router.add_get("/api/calibration", api_calibration_get)
    app.router.add_post("/api/calibration", api_calibration_post)
    app.router.add_post("/api/calibration/apply", api_calibration_apply)
    app.router.add_post("/api/calibration/capture", api_calibration_capture)
    app.router.add_delete("/api/calibration/capture", api_calibration_capture_delete)
    app.router.add_post("/api/calibration/solve", api_calibration_solve)
    app.router.add_static("/static/", STATIC)

    async def on_startup(app):
        loop = asyncio.get_running_loop()
        kind = args.kind
        hub = Hub(kind, loop)
        app["hub"] = hub
        emit, status = hub.emit_threadsafe, hub.status_threadsafe
        if kind == "serial":
            src = sources.SerialSource(args.port, emit, status, tcp=args.tcp, message=hub.message_threadsafe)
            hub.src.update(transport="serial" if args.tcp is None else "tcp")
        elif kind == "replay":
            src = sources.ReplaySource(args.replay, emit, status)
        elif kind == "simulate":
            src = sources.AudioSource("simulate", emit, status, path=args.simulate, use_ml=not args.no_ml,
                                      model=args.model)
        else:
            src = sources.AudioSource("demo", emit, status, use_ml=not args.no_ml, model=args.model)
        hub.src.update(port=args.replay or args.simulate or None)
        app["source"] = src
        src.start()
        app["ticker"] = asyncio.create_task(status_ticker(app))

    async def on_cleanup(app):
        app["source"].stop()
        app["ticker"].cancel()
        for c in list(app["hub"].clients):
            await c.ws.close()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="dashboard.py", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--port", nargs="?", const="auto", metavar="PORT",
                   help="serial port of the headband (no value: auto-detect)")
    g.add_argument("--tcp", nargs="?", const="", metavar="HOST[:PORT]",
                   help="the headband over Wi-Fi (no value: headband.local:5000, else 192.168.4.1:5000)")
    g.add_argument("--replay", metavar="LOG.csv", help="replay a scripts/telemetry.py --log CSV in real time")
    g.add_argument("--simulate", metavar="AUDIO", help="run the Python pipeline over an audio file (1 or 4 ch, B L R T)")
    g.add_argument("--demo", action="store_true", help="scripted synthetic scene through the Python pipeline")
    ap.add_argument("--http-port", type=int, default=8765, help="HTTP port (default 8765)")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 to show it on another machine)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    ap.add_argument("--no-ml", action="store_true", help="simulate/demo: rules only, no classifier")
    ap.add_argument("--model", default=sources.DEFAULT_MODEL, help="simulate/demo: TFLite classifier")
    args = ap.parse_args(argv)
    if args.port or args.tcp is not None:
        args.kind = "serial"
    elif args.replay:
        args.kind = "replay"
    elif args.simulate:
        args.kind = "simulate"
    elif args.demo:
        args.kind = "demo"
    elif sources.telemetry.candidate_ports():
        args.kind, args.port = "serial", "auto"
    elif _probe_tcp():
        args.kind, args.tcp = "serial", f"{sources.telemetry.TCP_HOST}:{sources.telemetry.TCP_PORT}"
    else:
        print("# dashboard: no mode given, no serial port and no headband on Wi-Fi: running --demo", flush=True)
        args.kind = "demo"
    for p in (args.replay, args.simulate):
        if p and not os.path.exists(p):
            ap.error(f"{p}: no such file")
    return args


def _probe_tcp(timeout=2.0):
    """Auto mode: is a headband answering at headband.local:5000 (mDNS) within `timeout` s?"""
    t = sources.telemetry
    try:
        t.open_tcp(t.TCP_HOST, t.TCP_PORT, timeout=timeout).close()
        print(f"# dashboard: found the headband at {t.TCP_HOST}:{t.TCP_PORT} (Wi-Fi)", flush=True)
        return True
    except OSError:
        return False


def main(argv=None):
    args = parse_args(argv)
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.http_port}/"
    via = "" if args.kind != "serial" else (" via tcp " + (args.tcp or "headband.local:5000") if args.tcp is not None
                                            else " via usb serial")
    print(f"# dashboard: source={args.kind}{via} -> {url}", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    web.run_app(make_app(args), host=args.host, port=args.http_port, print=None, access_log=None)


if __name__ == "__main__":
    main()
