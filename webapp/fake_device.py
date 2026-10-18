"""Dev tool: a fake headband on a pseudo-terminal (and optionally a TCP port, like the board's Wi-Fi server), speaking
the firmware's protocol, so the dashboard's device path, scripts/telemetry.py and scripts/record_device.py can be
exercised with no board.

It prints the pty path, then streams the Python pipeline's frames for the demo scene (or an audio file) as
'T,...' lines at 46.9/s in the firmware's printf format, looping. Like the sketch it prints a '# headband fw' banner,
the 'H,...' header (again on every 'T' command), '# pdm chunk_us=...' once, and answers 'S' with
'# status mode=T frame=N dropped=N wifi_dropped=N free_heap=N batt_v=V' plus a '# wifi ...' line. 'R' switches that
connection to raw mode: binary packets with the firmware's 12-byte header (A5 5A, uint8 n_ch=4, uint8 flags=0,
uint16 n=1024, uint32 chunk_index, uint16 reserved=0) and 1024 x 4 int16 B,L,R,T, carrying the same scene's
4-channel audio, one per chunk, with chunk_index counting every chunk since start (as on the board). T lines carry
elev, a slowly discharging batt_v, the eight applied motor weights (overhead: the rotating triplet) and the bearing
status columns bstat / mused, worked out from the mics the fake's calibration has connected (bearing_fields: four
mics give a 3D fix while detected, with one mic dropping out of the fit for a second every ~10 s; three or fewer
follow the robust solver's rules, a front mic doing what a back one does; theta / elev are empty when the frame has
none, as on the board).
--old-format emits the 3-mic firmware's 8-byte header and 3 channels instead (host backward-compatibility test).

It also takes the manual haptics commands: 'h<F>,<FR>,<R>,<BR>,<B>,<BL>,<L>,<FL>[,<hz>]\n' (PWM duties 0-255, clamped
to 190, at most 3 non-zero like the firmware) puts hap_* = duty / 190 into the T lines, with one '# haptics manual ...'
reply per new state, until 'A' or 3 s without another 'h' (the firmware's HAP_MANUAL_MS).

Calibration commands, like the firmware: 'C<w0>,<l>,<r>,<t>\n' (DMA word of DSP channel 0 = the back mic, then the
left, right and top mic, -1 = not connected) or 'C<w0>,<l>,<r>,<t>,F\n' (channel 0 is the front mic; ',B' = back),
'Y<m1>,...,<m8>\n' (motor angles, >= 900 = not fitted), 'K' (report), each answered with a '# cal ...' line (also sent
at start and on TCP connect; channel 0 printed as F=.. or B=..; the compiled default is B=w0 L=w3 R=w2 T=w5, the PCB
headers MK2..MK5 under the board's right-first slot order); 'L' answers one '# levels w0 rms=.. peak=.. dc=.. ... w5
... | B=w0 L=w3 R=w2 T=w5' line (room noise on the --wiring words). 'W' streams all six PDM words (12-byte header,
n_ch = 6) of a headset wired as --wiring (default back=0,left=3,right=2,top=5, like the firmware's default map; e.g.
front=0,left=3,right=2,top=5 for a front mic instead of the back one: which word carries which mic; on a line with
one mic the empty slot copies it, a line with none reads -0.944 FS). To test auto-calibration without hardware, each
'W' capture can carry two claps ~1 s in from a direction: the next one given by 'Z<theta>[,<elev>]\n' (degrees; also
'Zfront', 'Zright', ...), else the next entry of --cal-script front,right,left (in order, once each), else none (room
noise only).

Stealth mode, like the firmware but kept in memory: 'Q1\n' / 'Q0\n' answered with '# led stealth=on|off saved' (anything
else after Q: '# led: bad ...'), 'Q\n' alone with '# led stealth=on|off'; the '# led' line also follows the '# cal' line
at start and on TCP connect.

With --tcp PORT it also listens on that TCP port: one client at a time, a new connection replaces the old one, and
on connect the client gets '# headband fw', '# wifi', '# pdm' and the H header, exactly like the firmware. The pty
and the TCP client each have their own mode ('T' / 'R'). Writes never block: like the firmware's TCP sink, a
connection whose socket buffer is full drops whole packets/lines (counted as wifi_dropped). --raw-drop-every N
drops every Nth raw packet on purpose, to exercise the host's chunk-gap handling.

Usage:
  python -m webapp.fake_device [--audio FILE] [--no-ml] [--malformed-every N] [--tcp PORT] [--raw-drop-every N]
  python scripts/dashboard.py --port /dev/ttysNNN          # in another terminal, with the printed path
  python scripts/dashboard.py --tcp 127.0.0.1:PORT         # or over TCP
"""
import argparse, errno, os, select, socket, sys, time, tty
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

MOTORS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]       # droneears/config.py MOTOR_ANGLES order
HEADER = ("H,frame,t_ms,f0,score,strength,flat,hits,ml,detected,theta,conf,elev,overhead,batt_v,"
          + ",".join("hap_" + m for m in MOTORS) + ",bstat,mused\n")
MIC_BITS = {"0": 1, "L": 2, "R": 4, "T": 8}      # mused: the mics in a frame's fit ("0": channel 0, front or back)
CLEARLY_ABOVE = 30.0              # deg: with three mics an elevation (assumed above) only beyond this
BANNER = "# headband fw 1 fs=48000 nfft=2048 hop=1024 (fake_device.py)\n"
HOP = 1024
N_CH = 4                          # B, L, R, T
RAW_BYTES = 12 + HOP * N_CH * 2   # 8204
N_WORDS = 6                       # 'W': all six PDM DMA words
W_BYTES = 12 + HOP * N_WORDS * 2  # 12300
MAX_PENDING = 4 * W_BYTES         # like the firmware: never queue more than ~4 packets for a slow reader
CAL_MOTORS = [f"M{k}" for k in range(1, 9)]
MOT_DUTY_MAX = 190                # firmware/esp32s3/pins.h MOTOR_DUTY_MAX
MOT_MAX_ON = 3                    # pins.h MOTOR_MAX_ON: the firmware never drives more than 3 motors at once
HAP_MANUAL_S = 3.0                # the firmware's HAP_MANUAL_MS


def frames_for(audio, use_ml):
    """(wire frames, int16 audio (n, N_CH)) for the demo scene or an audio file."""
    from droneears.pipeline import Pipeline
    from webapp.sources import demo_scene, pipeline_to_frames, DEFAULT_MODEL
    if audio:
        from droneears.audio import load
        y = load(audio, mono=False)
        y = y if y.shape[0] in (1, N_CH) else y[:1]
    else:
        y = demo_scene()
    clf = None
    if use_ml and os.path.exists(DEFAULT_MODEL):
        from droneears.classifier import load_tflite_classifier
        clf = load_tflite_classifier(DEFAULT_MODEL)
    frames = pipeline_to_frames(Pipeline(classifier=clf).run(y), "serial")
    yn = np.repeat(y, N_CH, axis=0) if y.shape[0] == 1 else y
    pcm = (np.clip(yn.T, -1.0, 32767 / 32768) * 32768).astype("<i2")      # (n, 4) B, L, R, T
    return frames, pcm


def batt_v(t_s):
    """A fake battery: 4.05 V, discharging ~0.1 V per 10 minutes, with 10 mV of ADC jitter."""
    return max(3.3, 4.05 - t_s / 6000.0 + 0.01 * np.sin(t_s * 7.3))


def bearing_fields(f, connected, n):
    """(theta, elev, bstat, mused) of frame number n as the four-mic firmware prints them, for the connected mics
    ('BLRT' or 'FLRT' letters, from the fake's calibration: channel 0 is the back or the front mic, and either does
    the same, the geometry being mirror-symmetric). A frame from a pipeline that reports bstat / mused keeps them when
    all four are connected; otherwise: no bearing (bstat 0) for mono input, room noise (not detected) and two mics or
    fewer; four mics: a 3D fix (3, mused 15), except that for 1 s in every ~10 s one of them drops out of the fit
    and the best three carry on; three mics, like the robust solver: B L R (F L R) -> azimuth (2), plus an elevation
    assumed above when the source is clearly above; B L T / B R T (F L T / F R T) -> both when clearly above, else
    ambiguous (1); L R T -> front and back cannot be told apart, ambiguous (1)."""
    theta, elev = f.get("theta"), f.get("elev")
    connected = "".join("0" if c in "BF" else c for c in connected)
    use = [c for c in "0LRT" if c in connected]
    if len(use) == 4 and f.get("bstat") is not None:
        return theta, elev, f["bstat"], f.get("mused")
    if theta is None or not f["detected"] or len(use) < 3:
        return None, None, 0, 0
    if len(use) == 4 and (n // 47) % 10 == 7:
        use.remove("0LRT"[(n // 470) % 4])
    mused = sum(MIC_BITS[c] for c in use)
    if len(use) == 4:
        return theta, elev, 3, mused
    above = elev is not None and elev >= CLEARLY_ABOVE
    missing = next(c for c in "0LRT" if c not in use)
    if missing == "T":
        return theta, elev if above else None, 2, mused
    if missing in "LR" and above:
        return theta, elev, 2, mused
    return None, None, 1, mused


def t_line(frame, t_ms, f, batt, connected="BLRT"):
    z = lambda v: 0.0 if v is None else v
    o = lambda v: "" if v is None else "%.1f" % v                # empty = no azimuth / elevation this frame
    theta, elev, bstat, mused = bearing_fields(f, connected, frame)
    return (("T,%d,%d,%.1f,%.2f,%.2f,%.3f,%d,%.3f,%d,%s,%.3f,%s,%d,%.2f," + ",".join(["%.2f"] * len(MOTORS))
             + ",%d,%d\n") %
            (frame, t_ms, z(f["f0"]), z(f["score"]), z(f["strength"]), z(f["flat"]), f["hits"], z(f["ml"]),
             f["detected"], o(theta), z(f["conf"]), o(elev), f["overhead"], batt,
             *[z(h) for h in f["hap"]], bstat, mused or 0))


def raw_packet(pcm, idx, old=False):
    n = len(pcm) // HOP
    k = idx % n
    x = pcm[k * HOP:(k + 1) * HOP]
    if old:                                                   # 3-mic firmware: 8-byte header, 3 channels
        return b"\xA5\x5A" + HOP.to_bytes(2, "little") + (idx & 0xFFFFFFFF).to_bytes(4, "little") + \
            np.ascontiguousarray(x[:, :3]).tobytes()
    return (b"\xA5\x5A" + bytes([x.shape[1], 0]) + HOP.to_bytes(2, "little") + (idx & 0xFFFFFFFF).to_bytes(4, "little")
            + b"\x00\x00" + x.tobytes())


def words_packet(x, idx):
    """One 'W' packet: 12-byte header with n_ch = 6, then HOP x 6 int16."""
    return (b"\xA5\x5A" + bytes([x.shape[1], 0]) + HOP.to_bytes(2, "little") + (idx & 0xFFFFFFFF).to_bytes(4, "little")
            + b"\x00\x00" + np.ascontiguousarray(x, "<i2").tobytes())


class FakeCal:
    """The firmware's runtime calibration: DMA word per DSP channel (channel 0 at the back or at the front) and motor
    angles, with its '# cal' replies."""

    def __init__(self):
        self.mics = [0, 3, 2, 5]                        # channel 0, L, R, T (the compiled default map, right first)
        self.front = False                              # channel 0 is the front mic (C...,F), else the back one
        self.motors = [0.0, 45.0, 90.0, 135.0, 180.0, -135.0, -90.0, -45.0]
        self.src = "default"

    def letters(self):
        return ("F" if self.front else "B") + "LRT"

    def mics_text(self):
        return " ".join(f"{c}={'-' if w < 0 else 'w%d' % w}" for c, w in zip(self.letters(), self.mics))

    def connected(self):
        return "".join(c for c, w in zip(self.letters(), self.mics) if w >= 0)

    def motors_text(self):
        return " ".join(f"{m}={'-' if a is None else '%.1f' % a}" for m, a in zip(CAL_MOTORS, self.motors))

    def report(self):
        return f"# cal mics {self.mics_text()} motors {self.motors_text()} src={self.src}\n"

    def set_mics(self, arg):
        parts = arg.split(",")
        side = parts.pop() if len(parts) == 5 and parts[-1] in ("F", "B") else "B"
        try:
            w = [int(v) for v in parts]
        except ValueError:
            w = []
        used = [v for v in w if v >= 0]
        if len(w) != 4 or any(not -1 <= v <= 5 for v in w) or len(used) != len(set(used)):
            return (f"# cal: bad 'C{arg}', want C<w0>,<l>,<r>,<t>[,F] (channel 0 at the back, or ,F at the front) with "
                    f"words 0-5 or -1, none twice\n")
        self.mics, self.front, self.src = w, side == "F", "nvs"
        return f"# cal mics {self.mics_text()} saved\n"

    def set_motors(self, arg):
        try:
            a = [float(v) for v in arg.split(",")]
        except ValueError:
            a = []
        if len(a) != 8 or any(v != v or (v < 900 and not -360 <= v <= 360) for v in a):
            return f"# cal: bad 'Y{arg}', want Y<m1>,...,<m8> in degrees (>= 900 = not fitted)\n"
        self.motors, self.src = [None if v >= 900 else v for v in a], "nvs"
        return f"# cal motors {self.motors_text()} saved\n"


def levels_line(wiring, cal, seed=0):
    """The firmware's reply to 'L': per DMA word rms / peak (dBFS, about its DC) and dc of a short stretch of room
    noise on a headset wired as `wiring` ({position: word}), then the words the DSP reads, as `cal` has them."""
    from webapp import calibration
    x = calibration.to_float(calibration.synth_capture(wiring, None, seconds=0.05, seed=seed))
    out = "# levels"
    for w in range(x.shape[1]):
        v = x[:, w] - float(np.mean(x[:, w]))
        rms, pk = float(np.sqrt(np.mean(v ** 2))), float(np.max(np.abs(v)))
        out += (f" w{w} rms={20 * np.log10(max(rms, 1e-6)):.1f} peak={20 * np.log10(max(pk, 1e-6)):.1f} "
                f"dc={float(np.mean(x[:, w])):+.3f}")
    return out + f" | {cal.mics_text()}\n"


class FakeLed:
    """The firmware's stealth mode ('Q': its detect LED stays off), in memory, with its '# led' replies."""

    def __init__(self):
        self.stealth = False

    def report(self, saved=False):
        return f"# led stealth={'on' if self.stealth else 'off'}{' saved' if saved else ''}\n"

    def command(self, arg):
        if arg not in ("", "0", "1"):
            return f"# led: bad 'Q{arg}', want Q0 or Q1\n"
        if arg:
            self.stealth = arg == "1"
        return self.report(saved=bool(arg))


class Sink:
    """One output (pty master fd or TCP socket) with its own mode and a bounded, non-blocking outbound buffer.
    send(unit) queues a whole line/packet or drops it (pending buffer full); pump() writes what the OS takes."""

    def __init__(self, name, fd_write, closer=None):
        self.name, self.fd_write, self.closer = name, fd_write, closer
        self.mode = "T"
        self.out = bytearray()
        self.dropped = 0
        self.alive = True
        self.harg = None              # the argument of a multi-byte command being collected (None: not in one)
        self.hcmd = None              # ... and its letter ('h', 'C', 'Y', 'Z', 'W', 'Q')
        self.wbuf = None              # 'W': the six-word capture being streamed (int16 (n, 6)) and the position in it
        self.wpos = 0
        self.wn = 0                   # chunks in the current capture ('W<n>')

    def send(self, data, droppable=True):
        if droppable and len(self.out) + len(data) > MAX_PENDING:
            self.dropped += 1
            return
        self.out += data
        self.pump()

    def pump(self):
        while self.out and self.alive:
            try:
                n = self.fd_write(bytes(self.out[:65536]))
            except (BlockingIOError, InterruptedError):
                return
            except OSError as e:
                if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    return
                self.alive = False
                return
            del self.out[:n]

    def close(self):
        self.alive = False
        if self.closer:
            try:
                self.closer()
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--audio", help="audio file instead of the scripted demo scene")
    ap.add_argument("--no-ml", action="store_true")
    ap.add_argument("--malformed-every", type=int, default=500, help="truncate one T line every N (0: never)")
    ap.add_argument("--tcp", type=int, metavar="PORT", help="also serve the protocol on this TCP port "
                    "(like the board's Wi-Fi server on 5000)")
    ap.add_argument("--tcp-host", default="127.0.0.1", help="bind address for --tcp (default 127.0.0.1)")
    ap.add_argument("--raw-drop-every", type=int, default=0, metavar="N",
                    help="drop every Nth raw packet on purpose (host chunk-gap test; 0: never)")
    ap.add_argument("--old-format", action="store_true",
                    help="raw packets in the 3-mic firmware's 8-byte, 3-channel format (backward-compat test)")
    ap.add_argument("--wiring", default="back=0,left=3,right=2,top=5",
                    help="which PDM word carries which mic position (front or back, left, right, top) in 'W' "
                         "captures and 'L' levels (default: the PCB headers MK2..MK5 under the board's right-first "
                         "slot order, as the firmware's default map; front=.. instead of back=.. for a front mic)")
    ap.add_argument("--cal-script", default="",
                    help="comma list of clap directions for successive 'W' captures, e.g. front,right,left")
    a = ap.parse_args()
    from webapp import calibration
    try:
        wiring = calibration.parse_wiring(a.wiring)
    except ValueError as e:
        ap.error(f"--wiring: {e}")
    cal_script = [d.strip() for d in a.cal_script.split(",") if d.strip()]
    for d in cal_script:
        if d not in calibration.DIRECTION_ANGLES:
            ap.error(f"--cal-script: unknown direction {d!r}")
    cal = FakeCal()
    led = FakeLed()
    cal_next = dict(z=None, k=0, seed=0)          # Z override; next --cal-script entry; capture counter

    def start_w(sk, arg=""):
        """'W<n>': like the firmware, n chunks (default 141 = 3 s, at most 375) of all six words, chunk_index 0 .. n-1,
        then nothing until 'T'; with the next clap direction, if any (Z override, else the next --cal-script entry)."""
        try:
            nch = int(arg) if arg else 141
        except ValueError:
            nch = 0
        if not 1 <= nch <= 375:
            if sk.mode == "T":
                sk.send(f"# cal: bad 'W{arg}', want W<chunks> 1-375\n".encode(), False)
            return
        if cal_next["z"] is not None:
            direction, cal_next["z"] = cal_next["z"], None
        elif cal_next["k"] < len(cal_script):
            direction = cal_script[cal_next["k"]]; cal_next["k"] += 1
        else:
            direction = None
        cal_next["seed"] += 1
        sk.wbuf = calibration.synth_capture(wiring, direction, seconds=nch * HOP / 48000.0 + 0.05, click_t=1.0,
                                            seed=cal_next["seed"])
        sk.wpos = 0; sk.wn = nch
        sk.mode = "W"
        print(f"# fake_device: 'W' capture on {sk.name}, clap from {direction or 'nowhere'}", flush=True)

    def set_z(arg):
        if arg in calibration.DIRECTION_ANGLES:
            cal_next["z"] = arg
            return
        try:
            v = [float(x) for x in arg.split(",")]
            cal_next["z"] = (v[0], v[1] if len(v) > 1 else 0.0)
        except (ValueError, IndexError):
            pass
    frames, pcm = frames_for(a.audio, not a.no_ml)
    m, s = os.openpty(); tty.setraw(s)
    os.set_blocking(m, False)
    print(os.ttyname(s), flush=True)

    pty = Sink("pty", lambda b: os.write(m, b))
    sinks = {"pty": pty}
    lsock = None
    if a.tcp:
        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind((a.tcp_host, a.tcp)); lsock.listen(1); lsock.setblocking(False)
        print(f"# fake_device: tcp {a.tcp_host}:{a.tcp}", flush=True)
    tcp_sock = None
    wifi_dropped_base = 0
    pdm_known = False
    man = dict(on=False, duty=[0] * len(MOTORS), hz=0, t=0.0)   # manual haptics ('h'), like the firmware's man_hap

    def manual_haptics(sk, arg):
        try:
            d = [int(x) for x in arg.split(",")]
        except ValueError:
            d = []
        nm = len(MOTORS)
        if len(d) < nm:
            if sk.mode == "T":
                sk.send(f"# haptics: bad 'h{arg}', want h<{'>,<'.join(MOTORS)}>[,<hz>]\n".encode(), False)
            return
        duty = [max(0, min(MOT_DUTY_MAX, x)) for x in d[:nm]]
        while sum(1 for x in duty if x > 0) > MOT_MAX_ON:    # the firmware's limit_motors(): drop the weakest
            duty[min((i for i in range(nm) if duty[i] > 0), key=lambda i: duty[i])] = 0
        hz = max(0, min(20, d[nm])) if len(d) > nm else 0
        changed = not man["on"] or man["duty"] != duty or man["hz"] != hz
        man.update(on=True, duty=duty, hz=hz, t=time.time())
        if changed and sk.mode == "T":
            sk.send(("# haptics manual " + " ".join(f"{k}={v}" for k, v in zip(MOTORS, duty))
                     + f" pulse_hz={hz} max={MOT_DUTY_MAX} max_on={MOT_MAX_ON} (detector again "
                     f"{int(HAP_MANUAL_S * 1000)} ms after the last h)\n").encode(), False)

    def wifi_dropped():
        return wifi_dropped_base + (sinks["tcp"].dropped if "tcp" in sinks else 0)

    tcp_peer = ["none"]                              # remembered at accept: getpeername() fails once it closes

    def wifi_line():
        peer = "none" if tcp_sock is None else tcp_peer[0]
        return (f"# wifi mode=AP ssid=HEADBAND ip={a.tcp_host if a.tcp else '0.0.0.0'} port={a.tcp or 5000} "
                f"mdns=headband.local client={peer}\n")

    pty.send(BANNER.encode(), False); pty.send(cal.report().encode(), False); pty.send(led.report().encode(), False)
    pty.send(HEADER.encode(), False)
    fps = 48000 / 1024
    n, t0 = 0, time.time()
    nxt = t0
    while True:
        rlist = [m] + ([lsock] if lsock else []) + ([tcp_sock] if tcp_sock else [])
        wlist = [x for x, k in ((m, "pty"), (tcp_sock, "tcp")) if x is not None and k in sinks and sinks[k].out]
        r, w, _ = select.select(rlist, wlist, [], max(0.0, nxt - time.time()))
        for k in ("pty", "tcp"):
            if k in sinks:
                sinks[k].pump()
        if lsock in r:
            try:
                c, addr = lsock.accept()
            except BlockingIOError:
                c = None
            if c is not None:
                if tcp_sock is not None:                     # a new connection replaces the old one
                    wifi_dropped_base += sinks["tcp"].dropped
                    sinks.pop("tcp").close()
                    print("# fake_device: tcp client replaced", flush=True)
                c.setblocking(False); c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                tcp_sock = c
                tcp_peer[0] = addr[0]
                sinks["tcp"] = Sink("tcp", c.send, c.close)
                print(f"# fake_device: tcp client {addr[0]}:{addr[1]} connected", flush=True)
                t = sinks["tcp"]
                t.send(BANNER.encode(), False); t.send(wifi_line().encode(), False); t.send(cal.report().encode(), False)
                t.send(led.report().encode(), False)
                if pdm_known:
                    t.send(b"# pdm chunk_us=21333 (expect 21333)\n", False)
                t.send(HEADER.encode(), False)
        for k, fd in (("pty", m), ("tcp", tcp_sock)):
            if fd is None or fd not in r or k not in sinks:
                continue
            try:
                data = os.read(m, 64) if k == "pty" else fd.recv(64)
            except (BlockingIOError, InterruptedError):
                continue
            except OSError:
                data = b"" if k == "tcp" else None
            if k == "tcp" and not data:                      # client closed (or reset)
                wifi_dropped_base += sinks["tcp"].dropped
                sinks.pop("tcp").close(); tcp_sock = None
                print("# fake_device: tcp client disconnected", flush=True)
                continue
            sk = sinks[k]
            for ch in (data or b"").decode(errors="ignore"):
                if sk.harg is not None:                      # inside "h..." / "C..." / "Y..." / "Z...": up to the newline
                    if ch in "\r\n":
                        arg, cmd, sk.harg, sk.hcmd = sk.harg, sk.hcmd, None, None
                        if cmd == "h":
                            manual_haptics(sk, arg)
                        elif cmd == "C" and sk.mode == "T":
                            sk.send(cal.set_mics(arg).encode(), False)
                        elif cmd == "Y" and sk.mode == "T":
                            sk.send(cal.set_motors(arg).encode(), False)
                        elif cmd == "C":
                            cal.set_mics(arg)
                        elif cmd == "Y":
                            cal.set_motors(arg)
                        elif cmd == "Z":
                            set_z(arg)
                        elif cmd == "W":
                            start_w(sk, arg)
                        elif cmd == "Q":
                            r = led.command(arg)
                            if sk.mode == "T":
                                sk.send(r.encode(), False)
                        continue
                    ok = ch.isdigit() or ch == "," or (sk.hcmd in "CYZQ" and (ch in "-." or (sk.hcmd == "Z" and ch.isalpha())
                                                                    or (sk.hcmd == "C" and ch in "FB")))
                    if ok and len(sk.harg) < 63:
                        sk.harg += ch
                        continue
                    sk.harg = sk.hcmd = None                 # anything else drops it; ch is handled below
                if ch in "hCYZWQ":
                    sk.harg, sk.hcmd = "", ch
                elif ch == "K" and sk.mode == "T":
                    sk.send(cal.report().encode(), False)
                elif ch == "L" and sk.mode == "T":
                    cal_next["seed"] += 1
                    sk.send(levels_line(wiring, cal, cal_next["seed"]).encode(), False)
                elif ch == "A":
                    man["on"] = False
                    if sk.mode == "T":
                        sk.send(b"# haptics auto (the detector drives the motors)\n", False)
                elif ch == "T":
                    sk.mode = "T"; sk.wbuf = None; sk.send(HEADER.encode(), False)
                elif ch == "R":
                    sk.mode = "R"
                elif ch == "S" and sk.mode == "T":
                    sk.send((f"# status mode=T frame={n} dropped=0 wifi_dropped={wifi_dropped()} "
                             f"free_heap={147000 - (n % 50) * 16} batt_v={batt_v(time.time() - t0):.2f}\n").encode(),
                            False)
                    sk.send(wifi_line().encode(), False)
        if "tcp" in sinks and not sinks["tcp"].alive:
            wifi_dropped_base += sinks["tcp"].dropped
            sinks.pop("tcp").close(); tcp_sock = None
            print("# fake_device: tcp client gone (write failed)", flush=True)
        if time.time() < nxt:
            continue
        nxt = max(nxt + 1 / fps, time.time() - 0.1)
        if n == 10:
            pdm_known = True
        f = frames[n % len(frames)]
        if man["on"] and time.time() - man["t"] > HAP_MANUAL_S:
            man["on"] = False                                # host went quiet: back to the detector
        if man["on"]:
            f = dict(f, hap=[x / MOT_DUTY_MAX for x in man["duty"]])
        line = t_line(n, int((time.time() - t0) * 1000), f, batt_v(time.time() - t0), cal.connected())
        if a.malformed_every and n % a.malformed_every == a.malformed_every // 2:
            line = line[: len(line) // 2] + "\n"
        pkt = None
        for sk in list(sinks.values()):
            if sk.mode == "T":
                if n == 10:
                    sk.send(b"# pdm chunk_us=21333 (expect 21333)\n")
                sk.send(line.encode())
            elif sk.mode == "W" and sk.wbuf is not None:
                i = sk.wpos                                   # like the firmware: chunk_index 0 .. n-1, then idle
                sk.send(words_packet(sk.wbuf[i * HOP:(i + 1) * HOP], i))
                sk.wpos += 1
                if sk.wpos >= sk.wn:
                    sk.wbuf = None
            elif sk.mode == "R":
                if a.raw_drop_every and n % a.raw_drop_every == a.raw_drop_every - 1:
                    if sk.name == "tcp":
                        sk.dropped += 1
                    continue
                pkt = pkt or raw_packet(pcm, n, a.old_format)
                sk.send(pkt)
        n += 1


if __name__ == "__main__":
    main()
