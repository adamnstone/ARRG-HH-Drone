"""Live monitor and logger for the headband's serial telemetry (firmware/esp32s3/headband.ino).

Protocol (device -> host, 921600 baud; USB CDC ignores the baud rate; the same bytes over Wi-Fi/TCP port 5000):
  '#...'                     comment / status lines, printed verbatim
  'H,frame,t_ms,f0,...'      header: names the columns of the T lines that follow
  'T,<frame>,<t_ms>,...'     one line per frame (~47 lines/s)
Commands (host -> device, single bytes): 'T' telemetry mode, 'R' raw audio mode, 'S' status line.

T lines are parsed by the header's column names, not by position, so new columns do not break this script (the
firmware appends bstat, the bearing status, and mused, the mics in the frame's fit). theta and elev are empty when the
frame has no azimuth / elevation; they parse as None (blank in a log, NaN in a plot). Malformed lines are skipped; a T line whose field count does not match the header is dropped, and after a run
of them the header is discarded and requested again ('T') so the parser resyncs on the next header.

Usage:
  python scripts/telemetry.py [--port PORT] [--log run.csv] [--seconds N]     live monitor (sends 'T' on connect)
  python scripts/telemetry.py --tcp [HOST[:PORT]] [--log run.csv]            the same over Wi-Fi (default
                                                                              headband.local:5000, else 192.168.4.1)
  python scripts/telemetry.py --list-ports
  python scripts/telemetry.py --plot run.csv [--out run.png]                  offline figure from a log

The status line and the plot are generic over the log's columns: elev and batt_v are shown when present, and every
hap_* column is a motor (8 on the current board, F FR R BR B BL L FL; older logs have 3 or 4), in header order.
"""
import argparse, collections, csv, glob, os, re, select, socket, sys, threading, time

BAUD = 921600
PORT_PATTERNS = [["/dev/cu.usbmodem*"], ["/dev/cu.usbserial*", "/dev/cu.SLAB*"], ["/dev/ttyACM*", "/dev/ttyUSB*"]]
ML_GATE = 0.7          # ML_THRESH in droneears/config.py (only drawn on the plot)
ELEV_ON = 60.0         # OVERHEAD_ELEV_ON in droneears/config.py (only drawn on the plot)
TCP_PORT = 5000        # WIFI_TCP_PORT in the sketch
TCP_HOST = "headband.local"
TCP_FALLBACK = "192.168.4.1"   # the board's soft-AP address


def candidate_ports():
    """Serial ports in preference order: native USB (usbmodem) first, then USB-UART bridges, then Linux names."""
    out = []
    for group in PORT_PATTERNS:
        for pat in group:
            out += sorted(glob.glob(pat))
    return out


def resolve_port(port):
    if port:
        return port
    c = candidate_ports()
    if not c:
        sys.exit("no serial port found (looked for " + ", ".join(p for g in PORT_PATTERNS for p in g) +
                 "); pass --port")
    return c[0]


def open_serial(port):
    """Open at BAUD. macOS sets 921600 via the IOSSIOSPEED ioctl, which ptys and some drivers reject; USB CDC
    ignores the baud rate anyway, so fall back to a standard rate rather than fail."""
    import serial
    try:
        return serial.Serial(port, BAUD, timeout=0.05)
    except OSError as e:
        print(f"# host: {port}: cannot set {BAUD} baud ({e}); opening at 115200 (USB CDC ignores baud)", flush=True)
        return serial.Serial(port, 115200, timeout=0.05)


def parse_tcp(spec):
    """'HOST[:PORT]' | '' | None -> (host or None, port). host None = the default (mDNS, then the AP address)."""
    if not spec:
        return None, TCP_PORT
    host, sep, port = spec.rpartition(":")
    if sep and port.isdigit() and host and "]" not in port:
        return host.strip("[]"), int(port)
    return spec.strip("[]"), TCP_PORT


def _resolve(host, port, timeout):
    """getaddrinfo with a timeout (mDNS lookups of a missing .local name can hang for several seconds)."""
    box = {}

    def work():
        try:
            box["r"] = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
        except OSError as e:
            box["e"] = e
    th = threading.Thread(target=work, daemon=True)
    th.start(); th.join(timeout)
    if "r" in box:
        return box["r"][0][4]
    raise box.get("e") or socket.timeout(f"resolving {host} timed out after {timeout:g} s")


class TcpLink:
    """A connected TCP socket with the small pyserial-like interface the tools use: read(n) (returns b"" after a
    short timeout, like serial), in_waiting, write(b), flush(), reset_input_buffer(), close(). A closed or reset
    connection raises ConnectionError from read()/write(), as a pulled cable raises SerialException."""

    READ_TIMEOUT = 0.05          # same as open_serial's timeout
    transport = "tcp"
    baudrate = None

    def __init__(self, sock, host, port):
        self.sock, self.host, self.port = sock, host, port
        self.peer = "%s:%d" % sock.getpeername()[:2]
        self.name = f"{host}:{port}"
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)   # raw audio is ~384 KB/s
        sock.settimeout(self.READ_TIMEOUT)

    @property
    def in_waiting(self):
        """Bytes readable without blocking (0 or the receive-buffer size hint; exact counts are not needed)."""
        r, _, _ = select.select([self.sock], [], [], 0)
        return 65536 if r else 0

    def read(self, n=1):
        try:
            data = self.sock.recv(max(int(n), 1))
        except socket.timeout:
            return b""
        except OSError as e:
            raise ConnectionError(f"{self.name}: {e}") from e
        if not data:
            raise ConnectionError(f"{self.name}: connection closed by the device")
        return data

    def read_available(self):
        return self.read(self.in_waiting or 1)

    def write(self, b):
        try:
            self.sock.sendall(b)
        except OSError as e:
            raise ConnectionError(f"{self.name}: {e}") from e
        return len(b)

    def flush(self):
        pass

    def reset_input_buffer(self):
        self.sock.setblocking(False)
        try:
            while self.sock.recv(65536):
                pass
        except (BlockingIOError, OSError):
            pass
        finally:
            self.sock.settimeout(self.READ_TIMEOUT)

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def open_tcp(host=None, port=TCP_PORT, timeout=3.0):
    """Connect to the headband over Wi-Fi. host None: headband.local (mDNS), then the soft-AP address 192.168.4.1.
    Returns a TcpLink; raises OSError (the last failure) when no candidate answers."""
    cands = [host] if host else [TCP_HOST, TCP_FALLBACK]
    err = None
    for h in cands:
        try:
            addr = _resolve(h, port, timeout)
            return TcpLink(socket.create_connection(addr[:2], timeout=timeout), h, port)
        except OSError as e:
            err = e
            if len(cands) > 1:
                print(f"# host: {h}:{port}: {e}", flush=True)
    raise err


def link_name(link):
    return getattr(link, "name", None) or getattr(link, "port", "?")


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


NULLABLE = ("theta", "elev")   # empty in a T line when the frame has no azimuth / elevation


def _row_nums(cols, vals):
    """Field strings of one T line / log row -> numbers by column, or None if any field is not a number (an empty
    field is allowed only in the NULLABLE columns, where it means None)."""
    nums = [None if (v == "" and c in NULLABLE) else _num(v) for c, v in zip(cols, vals)]
    if any(n is None and not (v == "" and c in NULLABLE) for c, v, n in zip(cols, vals, nums)):
        return None
    return nums


class TelemetryParser:
    """Line-oriented parser. feed(bytes) -> list of events: ('comment', str) | ('header', [cols]) | ('row', dict)."""

    MAX_BAD_RUN = 10

    def __init__(self):
        self.buf = b""
        self.cols = None
        self.bad = 0              # malformed lines in total
        self.bad_run = 0          # consecutive mismatched T lines
        self.headers = 0
        self.need_header = False  # set when the caller should send 'T' to get a fresh header

    def feed(self, data):
        self.buf += data
        ev = []
        *lines, self.buf = self.buf.split(b"\n")
        if len(self.buf) > 4096:  # binary garbage without newlines (e.g. leftover raw packets): drop it
            self.buf = b""; self.bad += 1
        for raw in lines:
            line = raw.decode("ascii", errors="replace").strip("\r\x00 ")
            if not line:
                continue
            if line.startswith("#"):
                ev.append(("comment", line))
            elif line.startswith("H,"):
                cols = [c.strip() for c in line.split(",")[1:]]
                if cols and all(c and c.replace("_", "").isalnum() for c in cols):
                    self.cols = cols; self.headers += 1; self.bad_run = 0; self.need_header = False
                    ev.append(("header", cols))
                else:
                    self.bad += 1
            elif line.startswith("T,"):
                if self.cols is None:
                    self.need_header = True
                    continue
                vals = line.split(",")[1:]
                nums = _row_nums(self.cols, vals) if len(vals) == len(self.cols) else None
                if nums is None:
                    self.bad += 1; self.bad_run += 1
                    if self.bad_run >= self.MAX_BAD_RUN:
                        self.cols = None; self.need_header = True; self.bad_run = 0
                    continue
                self.bad_run = 0
                row = dict(zip(self.cols, nums))
                row["_raw"] = vals
                ev.append(("row", row))
            else:
                self.bad += 1
        return ev


LEVELS_RE = re.compile(r"w(\d) rms=(-?[\d.]+) peak=(-?[\d.]+) dc=([-+][\d.]+)")
LEVELS_MAP_RE = re.compile(r"\b([A-Z])=w(\d)")
# the board's DMA words: the v3 board reads the SEL 3V3 phase of each data line first (right-first slot order, verified
# by tapping), so w0 is DIN0's SEL 3V3 slot (header MK2), w1 its SEL GND slot (MK1), ...; DIN2's SEL 3V3 slot has no header
LEVELS_SLOT = {0: "DIN0 SEL 3V3 (MK2)", 1: "DIN0 SEL GND (MK1)", 2: "DIN1 SEL 3V3 (MK4)", 3: "DIN1 SEL GND (MK3)",
               4: "DIN2 SEL 3V3 (no header)", 5: "DIN2 SEL GND (MK5)"}


def levels_parse(line):
    """A '# levels ...' line from the board -> (words, chmap, hints). words: {word: (rms_dBFS, peak_dBFS, dc)} for
    the PDM DMA words (LEVELS_SLOT); chmap: {'B' or 'F': word, 'L': word, 'R': word, 'T': word}, the words the DSP
    reads (channel 0 is 'F' when it is the front mic; older 3-mic firmware: L, R, B); hints: what
    the numbers mean for the tap test. A word pair of one line that is identical to the last digit means only one
    PDM phase is being driven on that line (one mic, or two mics strapped to the same SEL), because the floating
    phase mirrors the driven one; a DC offset far from 0 means the line is stuck or two mics are fighting on it."""
    words = {int(w): (float(r), float(pk), float(dc)) for w, r, pk, dc in LEVELS_RE.findall(line)}
    chmap = {ch: int(w) for ch, w in LEVELS_MAP_RE.findall(line.split("|", 1)[1])} if "|" in line else {}
    hints = []
    for din, (a, b) in ((0, (0, 1)), (1, (2, 3)), (2, (4, 5))):
        if a in words and b in words:
            (ra, pa, da), (rb, pb, db) = words[a], words[b]
            same = abs(ra - rb) <= 0.1 and abs(pa - pb) <= 0.1 and abs(da - db) <= 0.002
            bad_dc = max(abs(da), abs(db)) > 0.2
            if bad_dc:
                hints.append(f"DIN{din}: dc {da:+.2f} = stuck line or two mics on the same SEL (one SEL to GND, the other to 3V3)")
            elif same and max(pa, pb) < -1.0:
                hints.append(f"DIN{din}: w{a}=w{b}, one mic (its mirror shows on the empty phase)")
            elif not same:
                hints.append(f"DIN{din}: two different signals")
    return words, chmap, hints


def levels_hints(line):
    """The '# levels' line with levels_parse()'s hints appended, for the terminal."""
    hints = levels_parse(line)[2]
    return line + ("   <- " + "; ".join(hints) if hints else "")


def bar(w, n=5):
    w = 0.0 if w is None else max(0.0, min(1.0, w))
    k = int(round(w * n))
    return "#" * k + "." * (n - k)


def hap_cols(keys):
    """The motor columns of a row / header, in header order: every 'hap_<name>' (3, 4 or 8 of them)."""
    return [k for k in keys if k.startswith("hap_")]


def status_text(r, fps):
    if r is None:
        return f"waiting for telemetry ... {fps:4.1f} fr/s"
    g = r.get
    state = "OVERHEAD" if g("overhead") else ("DRONE" if g("detected") else "idle")
    def f(k, fmt, d="-"):
        v = g(k)
        return d if v is None else format(v, fmt)
    hc = hap_cols(r)
    n = 3 if len(hc) > 4 else 5                      # shorter bars when there are 8 motors
    hap = " ".join(f"{k[4:]}[{bar(g(k), n)}]" for k in hc)
    elev = f"  elev {f('elev', '4.0f')}" if "elev" in r else ""
    bv = g("batt_v")                     # 0 = not read yet
    batt = f"  batt {bv:4.2f} V" if bv is not None and bv > 0 else ""
    return (f"{state:8s} f0 {f('f0', '5.0f')} Hz  score {f('score', '4.1f')}  hits {f('hits', '3.0f')}  "
            f"ml {f('ml', '4.2f')}  theta {f('theta', '6.1f')}{elev}  conf {f('conf', '4.2f')}  {hap}{batt}  "
            f"{fps:4.1f} fr/s")


def monitor(args):
    tcp = args.tcp is not None
    if tcp:
        thost, tport = parse_tcp(args.tcp)
        ser = None
        port = None
    else:
        port = resolve_port(args.port)
        ser = open_serial(port)
    tty = sys.stdout.isatty()

    def connect():
        """TCP: (re)connect until it works or --seconds runs out; returns the link or None."""
        while not (args.seconds and time.time() - t0 >= args.seconds):
            try:
                return open_tcp(thost, tport, timeout=3.0)
            except OSError as e:
                print(f"# host: tcp {thost or TCP_HOST + ' or ' + TCP_FALLBACK}:{tport}: {e}; retrying in 1 s", flush=True)
                time.sleep(1.0)
        return None

    t0 = time.time()
    if tcp:
        ser = connect()
        if ser is None:
            sys.exit("# host: no connection")
        print(f"# host: connected to {ser.name} (tcp, peer {ser.peer}); sending 'T'", flush=True)
    else:
        print(f"# host: connected to {port} at {ser.baudrate} baud; sending 'T'", flush=True)
    ser.reset_input_buffer()
    ser.write(b"T"); ser.flush()
    if args.motor_test:
        ser.write(b"M"); ser.flush()
    p = TelemetryParser()
    last_levels = last_motor = time.time()
    reconnects = 0
    logf = writer = None
    if args.log:
        os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
        logf = open(args.log, "w", newline="")
        writer = csv.writer(logf)
    log_cols = None
    last_status = 0.0; last_req = t0
    stamps = collections.deque()
    last_row = None; n_rows = 0
    status_period = 0.25 if tty else 1.0

    def emit(text):
        if tty:
            sys.stdout.write("\r\x1b[K" + text + "\n")
        else:
            sys.stdout.write(text + "\n")
        sys.stdout.flush()

    try:
        while True:
            now = time.time()
            if args.seconds and now - t0 >= args.seconds:
                break
            try:
                data = ser.read(ser.in_waiting or 1)
            except ConnectionError as e:             # TCP only (serial errors propagate as before)
                print(f"# host: {e}; reconnecting", flush=True)
                ser.close()
                ser = connect()
                if ser is None:
                    break
                reconnects += 1
                print(f"# host: reconnected to {ser.name}; sending 'T'", flush=True)
                p = TelemetryParser(); ser.write(b"T")
                continue
            now = time.time()
            for kind, x in p.feed(data):
                if kind == "comment":
                    emit(levels_hints(x) if args.levels and x.startswith("# levels") else x)
                elif kind == "header":
                    if writer and x != log_cols:
                        writer.writerow(["host_s"] + x)   # a new header row in the log = resync point for --plot
                        log_cols = x
                elif kind == "row":
                    n_rows += 1; last_row = x; stamps.append(now)
                    if writer:
                        writer.writerow([f"{now - t0:.3f}"] + x["_raw"])
            if args.levels and now - last_levels >= 0.25:  # tap test: one "# levels" line per poll
                try:
                    ser.write(b"L"); ser.flush()
                except ConnectionError:
                    pass
                last_levels = now
            if args.motor_test and now - last_motor >= 7.0:  # 8 motors x 0.8 s = 6.4 s per sequence; loop it
                try:
                    ser.write(b"M"); ser.flush()
                except ConnectionError:
                    pass
                last_motor = now
            if p.need_header and now - last_req > 2.0:      # T lines without a usable header: ask for one
                try:
                    ser.write(b"T"); ser.flush()
                except ConnectionError:
                    pass                                  # the next read() notices and reconnects
                last_req = now
            while stamps and now - stamps[0] > 1.0:
                stamps.popleft()
            if now - last_status >= status_period:
                last_status = now
                txt = status_text(last_row, len(stamps))
                if tty:
                    sys.stdout.write("\r\x1b[K" + txt[: os.get_terminal_size().columns - 1]); sys.stdout.flush()
                else:
                    print(f"[{now - t0:5.1f}s] {txt}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if tty:
            sys.stdout.write("\n")
        el = time.time() - t0
        print(f"# host: {n_rows} frames in {el:.1f} s ({n_rows / max(el, 1e-9):.1f} fr/s), "
              f"{p.headers} header(s), {p.bad} malformed line(s) skipped"
              + (f", {reconnects} reconnect(s)" if tcp else "")
              + (f"; log -> {args.log}" if args.log else ""), flush=True)
        if logf:
            logf.close()
        if ser is not None:
            ser.close()


def read_log(path):
    """Tolerant CSV reader: rows under the most recent 'host_s,...' header; malformed rows skipped."""
    cols, rows, bad = None, [], 0
    with open(path, newline="") as f:
        for rec in csv.reader(f):
            if not rec:
                continue
            if rec[0] == "host_s":
                cols = rec; continue
            if cols is None or len(rec) != len(cols):
                bad += 1; continue
            nums = _row_nums(cols, rec)
            if nums is None:
                bad += 1; continue
            rows.append(dict(zip(cols, nums)))
    return rows, bad


def plot(args):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, bad = read_log(args.plot)
    if not rows:
        sys.exit(f"{args.plot}: no telemetry rows")
    def col(k):
        return np.array([r.get(k, np.nan) for r in rows], dtype=float)
    if all("t_ms" in r for r in rows):
        t = col("t_ms") / 1000.0; t -= t[0]; xlabel = "device time (s)"
    else:
        t = col("host_s"); xlabel = "host time (s)"
    det = col("detected") > 0.5
    ovh = col("overhead") > 0.5 if any("overhead" in r for r in rows) else np.zeros(len(t), bool)
    blue, orange, aqua, ink, muted = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8984"
    has_elev = any("elev" in r for r in rows)
    hc = hap_cols(rows[-1])
    # panels: score | classifier | bearing | conf strip | elevation (if logged) | haptics (one row per motor)
    names = ["score", "ml", "theta", "conf"] + (["elev"] if has_elev else []) + ["hap"]
    ratios = {"score": 3, "ml": 2, "theta": 3, "conf": 1, "elev": 2, "hap": max(2, len(hc) * 0.45)}
    fig, axs = plt.subplots(len(names), 1, figsize=(11, 0.68 * sum(ratios[k] for k in names) + 1.0),
                            sharex=True, gridspec_kw=dict(height_ratios=[ratios[k] for k in names], hspace=0.62))
    fig.subplots_adjust(top=0.93)
    A = dict(zip(names, axs))
    def shade(a):
        a.fill_between(t, 0, 1, where=det, transform=a.get_xaxis_transform(), color=muted, alpha=0.18,
                       linewidth=0, label="DETECTED")
        if ovh.any():
            a.fill_between(t, 0, 1, where=ovh, transform=a.get_xaxis_transform(), facecolor="none",
                           edgecolor=muted, hatch="//", linewidth=0, label="overhead")
    for k in names:
        if k != "hap":
            shade(A[k])
    a = A["score"]
    a.plot(t, col("score"), color=blue, lw=1.5, label="score")
    a.plot(t, col("strength"), color=orange, lw=1.5, label="strength")
    a.set_ylabel("dB"); a.set_title("harmonic score and strength", loc="left", fontsize=10)
    a = A["ml"]
    a.plot(t, col("ml"), color=blue, lw=1.5, label="ml p(drone)")
    a.axhline(ML_GATE, color=ink, lw=1, ls="--", label=f"gate {ML_GATE}")
    a.set_ylim(-0.05, 1.05); a.set_ylabel("p"); a.set_title("classifier", loc="left", fontsize=10)
    a = A["theta"]
    a.plot(t, col("theta"), color=blue, lw=1.5, label="theta")
    a.set_ylabel("theta (deg)"); a.set_ylim(-185, 185); a.set_yticks([-180, -90, 0, 90, 180])
    a.set_title("bearing (azimuth)", loc="left", fontsize=10)
    a = A["conf"]
    a.plot(t, col("conf"), color=blue, lw=1.2, label="conf")
    a.set_ylim(-0.05, 1.05); a.set_yticks([0, 1]); a.set_ylabel("conf")
    a.set_title("bearing confidence", loc="left", fontsize=9)
    if has_elev:
        a = A["elev"]
        a.plot(t, col("elev"), color=blue, lw=1.5, label="elev")
        a.axhline(ELEV_ON, color=ink, lw=1, ls="--", label=f"overhead on {ELEV_ON:g}\u00b0")
        a.set_ylim(-5, 95); a.set_yticks([0, 30, 60, 90]); a.set_ylabel("elev (deg)")
        a.set_title("elevation (0 horizon, 90 overhead)", loc="left", fontsize=10)
    a = A["hap"]
    if hc:
        H = np.array([col(k) for k in hc])                     # (motors, frames), header order
        H = np.nan_to_num(H, nan=0.0)
        edges = np.concatenate([t, [t[-1] + (t[-1] - t[-2] if len(t) > 1 else 0.02)]])
        a.pcolormesh(edges, np.arange(len(hc) + 1), H, cmap="Oranges", vmin=0, vmax=1, shading="flat")
        a.set_yticks(np.arange(len(hc)) + 0.5); a.set_yticklabels([k[4:] for k in hc], fontsize=8)
        a.set_ylim(len(hc), 0)
    a.set_ylabel("motor"); a.set_xlabel(xlabel)
    a.set_title(f"haptic weights applied ({len(hc)} motors; dark = 1)", loc="left", fontsize=10)
    for k in names:
        if k != "hap":
            A[k].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False, ncol=4,
                        borderaxespad=0.1)
    for a in axs:
        a.grid(alpha=0.25, lw=0.5)
        a.spines["top"].set_visible(False)
    fig.suptitle(f"{os.path.basename(args.plot)}: {len(rows)} frames, {t[-1]:.1f} s, "
                 f"DETECTED {det.mean():.0%} of frames", fontsize=11, y=0.985)
    out = args.out or os.path.splitext(args.plot)[0] + ".png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"{args.plot}: {len(rows)} rows ({bad} malformed skipped) -> wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port (default: auto-detect, see --list-ports)")
    ap.add_argument("--tcp", nargs="?", const="", metavar="HOST[:PORT]",
                    help="connect over Wi-Fi instead of USB (no value: headband.local:5000, else 192.168.4.1:5000); "
                         "reconnects if the link drops")
    ap.add_argument("--list-ports", action="store_true", help="print candidate serial ports and exit")
    ap.add_argument("--log", metavar="FILE.csv", help="write every T line, with a host_s column prepended")
    ap.add_argument("--levels", action="store_true",
                    help="tap test: poll the board's 'L' command 4x a second and print its '# levels' lines "
                         "(RMS / peak / DC of every PDM DMA word, so a mic on an unused slot is seen too)")
    ap.add_argument("--motor-test", action="store_true",
                    help="send 'M' every 7 s: the board buzzes F, FR, R, BR, B, BL, L, FL in turn (0.5 s each); the hap_* bars in the "
                         "status line show which one is meant to be on")
    ap.add_argument("--seconds", type=float, default=None, help="stop after N seconds (default: until Ctrl-C)")
    ap.add_argument("--plot", metavar="LOG.csv", help="offline: render a figure from a log (no device)")
    ap.add_argument("--out", metavar="PNG", help="output image for --plot (default: LOG.png)")
    args = ap.parse_args()
    if args.list_ports:
        c = candidate_ports()
        print("\n".join(c) if c else "no candidate serial ports")
        return
    if args.plot:
        plot(args)
        return
    monitor(args)


if __name__ == "__main__":
    main()
