"""Record 4-channel (B, L, R, T) 48 kHz audio from the headband over serial, for the held-out field session.

Sends 'R' (raw audio mode) and reads binary packets:
  12-byte header: magic 0xA5 0x5A, uint8 n_ch (4), uint8 flags (0), uint16 LE n_frames (1024),
  uint32 LE chunk_index (+1 per chunk), uint16 LE reserved (0);
  then n_frames * n_ch int16 LE samples interleaved B, L, R, T (the board's mic order, droneears/config.py MIC_NAMES).
The WAV gets as many channels as the packets carry, in that order. The old 3-mic firmware's 8-byte header (magic,
uint16 n_frames, uint32 chunk_index, then L, R, B) is still recognised: a new header has 1 <= n_ch <= 16 and
flags 0 in bytes 2-3, an old one has n_frames = 1024 there (bytes 00 04), so the two cannot be confused.
The parser resyncs on the magic bytes (a candidate packet is accepted when its chunk_index is the expected next
one, or when the magic of the following packet is where it should be). Gaps in chunk_index are filled with zeros
and counted. ~384 KB/s: use the ESP32-S3's native USB port (/dev/cu.usbmodem*), a USB-UART bridge cannot keep up.
--tcp records the same packets over Wi-Fi (TCP port 5000). That is best effort: when the link stalls the board drops
whole packets, which show up here as zero-filled chunk gaps; if the connection drops it reconnects, resends 'R' and
zero-fills the gap (gaps over ~10 s are treated as a restart). Use USB for the held-out session when you can.

Every second it prints per-channel RMS (dBFS), peak (dBFS) and DC offset (% of full scale), so the tap test (tap
each mic and watch which channel jumps) and the DC check can be done live. On exit it sends 'T' (back to
telemetry) and prints a summary with a clipping / DC verdict.

Usage:
  python scripts/record_device.py [--port PORT] [--seconds 60] [--out FILE.wav | --drone] [--holdout] [--mono-check]
  python scripts/record_device.py --tcp [HOST[:PORT]] ...    # over Wi-Fi (default headband.local, else 192.168.4.1)
  python scripts/record_device.py --monitor                 # live levels only, no file
  python scripts/record_device.py --list-ports
"""
import argparse, glob, os, struct, sys, time
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BAUD = 921600
FS = 48000
MAGIC = b"\xA5\x5A"
HDR = struct.Struct("<2sBBHIH")       # magic, n_ch, flags, n_frames, chunk_index, reserved (12 bytes)
HDR_OLD = struct.Struct("<2sHI")      # 3-mic firmware: magic, n_frames, chunk_index (8 bytes), always 3 channels
MAX_CH = 16
# channel names by channel count (the board's order); anything else is named ch0, ch1, ...
CH_NAMES_BY_N = {4: ["B", "L", "R", "T"], 3: ["L", "R", "B"], 5: ["F", "B", "L", "R", "T"]}


def ch_names(n):
    return CH_NAMES_BY_N.get(n) or [f"ch{i}" for i in range(n)]
MAX_FRAMES = 8192                     # sanity bound on n_frames in a header
MAX_GAP_FILL = 10 * FS // 1024        # never zero-fill more than ~10 s for one gap (index jump = restart instead)
PORT_PATTERNS = [["/dev/cu.usbmodem*"], ["/dev/cu.usbserial*", "/dev/cu.SLAB*"], ["/dev/ttyACM*", "/dev/ttyUSB*"]]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from telemetry import open_tcp, parse_tcp, TCP_HOST, TCP_FALLBACK  # noqa: E402  (the shared Wi-Fi transport)


def candidate_ports():
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


class RawParser:
    """feed(bytes) -> list of (chunk_index, int16 array (n_frames, n_ch)). n_ch comes from each packet's header
    (12-byte format), or is 3 for the old 8-byte format."""

    def __init__(self):
        self.buf = bytearray()
        self.expect = None
        self.skipped = 0          # bytes discarded while searching for sync
        self.old_format = 0       # packets parsed in the old 8-byte format

    def _header(self):
        """(n_ch, n_frames, chunk_index, header size) of the candidate at buf[0], or None if implausible.
        Needs len(buf) >= HDR.size."""
        _, nch, flags, n, idx, _r = HDR.unpack_from(self.buf, 0)
        if 1 <= nch <= MAX_CH and flags == 0 and 1 <= n <= MAX_FRAMES:
            return nch, n, idx, HDR.size
        _, n, idx = HDR_OLD.unpack_from(self.buf, 0)
        if 1 <= n <= MAX_FRAMES and self.buf[2] == 0:   # old firmware: n_frames 1024 -> low byte 0, never n_ch
            return 3, n, idx, HDR_OLD.size
        return None

    def feed(self, data, final=False):
        self.buf += data
        out = []
        while True:
            i = self.buf.find(MAGIC)
            if i < 0:
                keep = 1 if self.buf.endswith(MAGIC[:1]) else 0
                self.skipped += len(self.buf) - keep
                del self.buf[: len(self.buf) - keep]
                break
            if i:
                self.skipped += i; del self.buf[:i]
            if len(self.buf) < HDR.size:
                break
            h = self._header()
            if h is None:
                self.skipped += 1; del self.buf[:1]; continue
            nch, n, idx, hsz = h
            end = hsz + n * nch * 2
            if len(self.buf) < end:
                break
            # A packet counts only if the next magic lines up right after it: USB CDC occasionally loses bytes inside
            # a packet (seen on the S3: a 'W' packet 128 bytes short), and the expected chunk index alone would let
            # such a packet through, its samples shifted into the wrong channels. So each packet is released when the
            # next one's magic arrives (or at final, if it is the expected index or the very first packet).
            if len(self.buf) >= end + 2:
                ok = self.buf[end:end + 2] == MAGIC
            elif final:
                ok = self.expect is None or idx == self.expect
            else:
                break                                    # wait for more bytes before deciding
            if not ok:
                self.skipped += 1; del self.buf[:1]; continue
            x = np.frombuffer(bytes(self.buf[hsz:end]), dtype="<i2").reshape(n, nch)
            del self.buf[:end]
            if hsz == HDR_OLD.size:
                self.old_format += 1
            out.append((idx, x))
            self.expect = (idx + 1) & 0xFFFFFFFF
        return out


def dbfs(v):
    return 20.0 * np.log10(max(float(v), 1e-9))


class Stats:
    """Running per-channel sums over int16 samples."""

    def __init__(self, nch):
        self.nch = nch; self.names = ch_names(nch)
        self.n = 0
        self.s = np.zeros(nch); self.ss = np.zeros(nch); self.pk = np.zeros(nch)

    def add(self, x):
        xf = x.astype(np.float64) / 32768.0
        self.n += len(x); self.s += xf.sum(0); self.ss += (xf ** 2).sum(0)
        if len(x):
            self.pk = np.maximum(self.pk, np.abs(xf).max(0))

    def dc(self):
        return self.s / max(self.n, 1)

    def rms_ac(self):           # RMS with the DC offset removed (the signal level)
        return np.sqrt(np.maximum(self.ss / max(self.n, 1) - self.dc() ** 2, 0.0))

    def line(self):
        rms, dc = self.rms_ac(), self.dc()
        return "  ".join(f"{self.names[c]} rms {dbfs(rms[c]):6.1f} pk {dbfs(self.pk[c]):6.1f} dc {100 * dc[c]:+5.2f}%"
                         for c in range(self.nch))


def default_out(args):
    folder = os.path.join(ROOT, "data", "team", "drone" if args.drone else "background")
    name = "device_" + time.strftime("%Y%m%d_%H%M%S") + ("_holdout" if args.holdout else "") + ".wav"
    return os.path.join(folder, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port (default: auto-detect, see --list-ports)")
    ap.add_argument("--tcp", nargs="?", const="", metavar="HOST[:PORT]",
                    help="record over Wi-Fi instead of USB (no value: headband.local:5000, else 192.168.4.1:5000); "
                         "best effort, reconnects if the link drops")
    ap.add_argument("--list-ports", action="store_true", help="print candidate serial ports and exit")
    ap.add_argument("--seconds", type=float, default=60.0, help="recording length in seconds (default 60)")
    ap.add_argument("--out", metavar="FILE.wav",
                    help="output WAV (default data/team/background/device_<timestamp>.wav, or data/team/drone/ "
                         "with --drone)")
    ap.add_argument("--drone", action="store_true", help="default output folder data/team/drone/")
    ap.add_argument("--holdout", action="store_true", help="append _holdout to the default file name "
                    "(held-out test session, see TRAINING_PLAN.md section 4)")
    ap.add_argument("--monitor", action="store_true", help="live level display only, write no file "
                    "(runs until Ctrl-C unless --seconds is given explicitly)")
    ap.add_argument("--mono-check", action="store_true", help="after recording, run the rules-only pipeline on "
                    "all channels and print the detection summary and median bearing / elevation")
    args = ap.parse_args()
    if args.list_ports:
        c = candidate_ports()
        print("\n".join(c) if c else "no candidate serial ports")
        return
    if args.monitor and "--seconds" not in sys.argv:
        args.seconds = None
    tcp = args.tcp is not None
    out = None if args.monitor else (args.out or default_out(args))
    target = None if args.seconds is None else int(round(args.seconds * FS))
    if tcp:
        thost, tport = parse_tcp(args.tcp)

        def connect(deadline=None):
            while deadline is None or time.time() < deadline:
                try:
                    return open_tcp(thost, tport, timeout=3.0)
                except OSError as e:
                    print(f"# host: tcp {thost or TCP_HOST + ' or ' + TCP_FALLBACK}:{tport}: {e}; retrying in 1 s", flush=True)
                    time.sleep(1.0)
            return None
        ser = connect(time.time() + 30.0)
        if ser is None:
            sys.exit("# host: no Wi-Fi connection to the headband")
        port = f"{ser.name} (tcp)"
        ser.reset_input_buffer()          # drop the connect banner (comments + H line) before switching to raw
    else:
        port = resolve_port(args.port)
        ser = open_serial(port)
    sf_out = None                         # opened on the first packet, with that packet's channel count
    if out:
        import soundfile as sf
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    print(f"# host: {port} -> {'monitor only' if out is None else out}; "
          f"{'until Ctrl-C' if target is None else f'{args.seconds:g} s'}; sending 'R'", flush=True)
    ser.write(b"R"); ser.flush()

    p = RawParser()
    total = sec = None                    # Stats, sized on the first packet
    nch = None
    keep = [] if args.mono_check else None
    n_written = 0; chunks = 0; dropped_chunks = 0; dropped_samples = 0; restarts = 0; last_idx = None
    reconnects = 0
    t0 = time.time(); last_data = t0; last_warn = t0

    def put(x):
        nonlocal n_written, total, sec, sf_out, nch
        if nch is None:
            nch = x.shape[1]
            total, sec = Stats(nch), Stats(nch)
            print(f"# host: {nch}-channel packets ({', '.join(ch_names(nch))})"
                  + (" in the old 8-byte format" if p.old_format else ""), flush=True)
            if out:
                sf_out = sf.SoundFile(out, "w", samplerate=FS, channels=nch, subtype="PCM_16")
        if x.shape[1] != nch:
            print(f"# host: packet with {x.shape[1]} channels in a {nch}-channel recording: dropped", flush=True)
            return
        if target is not None:
            x = x[: max(0, target - n_written)]
        if not len(x):
            return
        if sf_out is not None:
            sf_out.write(x)
        if keep is not None:
            keep.append(x.copy())
        total.add(x)
        # per-second display, by sample count (device time), split exactly at 1 s boundaries
        while len(x):
            room = FS - sec.n
            a, x = x[:room], x[room:]
            sec.add(a)
            if sec.n >= FS:
                print(f"{(n_written + len(a)) / FS:6.1f}s  {sec.line()}  drop {dropped_chunks}", flush=True)
                sec.__init__(nch)
            n_written += len(a)

    try:
        while target is None or n_written < target:
            try:
                data = ser.read(ser.in_waiting or 1)
            except ConnectionError as e:             # TCP only: reconnect, resend 'R'; the gap is zero-filled below
                print(f"# host: {e}; reconnecting", flush=True)
                ser.close()
                ser = connect(time.time() + 30.0)
                if ser is None:
                    print("# host: could not reconnect within 30 s; stopping", flush=True)
                    break
                reconnects += 1
                p.expect = None                      # resync from scratch; chunk_index continues on the device
                ser.write(b"R"); last_warn = time.time()
                print(f"# host: reconnected to {ser.name}; sent 'R'", flush=True)
                continue
            now = time.time()
            if data:
                last_data = now
            pk = p.feed(data)
            if pk:
                last_warn = now
            elif now - last_warn > 3.0:
                print(f"# host: no raw packets for {now - last_warn:.0f} s (received {now - last_data:.1f} s ago); "
                      + ("Wi-Fi link stalled? " if tcp else "is this the S3's native USB port? ")
                      + "resending 'R'", flush=True)
                try:
                    ser.write(b"R"); ser.flush()
                except ConnectionError:
                    pass                                 # the next read() reconnects
                last_warn = now
            for idx, x in pk:
                if last_idx is not None:
                    gap = (idx - last_idx - 1) & 0xFFFFFFFF
                    if 0 < gap <= MAX_GAP_FILL:
                        dropped_chunks += gap; dropped_samples += gap * len(x)
                        print(f"# host: chunk gap {last_idx} -> {idx}: {gap} chunk(s) zero-filled", flush=True)
                        put(np.zeros((gap * len(x), x.shape[1]), np.int16))
                    elif gap > MAX_GAP_FILL:
                        restarts += 1
                        print(f"# host: chunk index jumped {last_idx} -> {idx}; treated as a stream restart "
                              "(not zero-filled)", flush=True)
                last_idx = idx; chunks += 1
                put(x)
    except KeyboardInterrupt:
        print("\n# host: interrupted", flush=True)
    finally:
        if ser is not None:
            try:
                ser.write(b"T"); ser.flush()
            except Exception:
                pass
            ser.close()
        if sf_out is not None:
            sf_out.close()

    dur = n_written / FS
    print(f"\nsummary: {dur:.2f} s, {n_written} samples per channel, {chunks} chunks received, "
          f"{dropped_chunks} dropped chunk(s) zero-filled ({dropped_samples} samples), {restarts} restart(s), "
          f"{p.skipped} byte(s) skipped resyncing, " + (f"{reconnects} reconnect(s), " if tcp else "")
          + f"wall {time.time() - t0:.1f} s")
    if total is not None and total.n:
        rms, dc = total.rms_ac(), total.dc()
        names = total.names
        for c in range(nch):
            print(f"  {names[c]}: rms {dbfs(rms[c]):6.1f} dBFS  dc {100 * dc[c]:+6.2f}% FS  "
                  f"peak {dbfs(total.pk[c]):6.1f} dBFS")
        clip = [names[c] for c in range(nch) if dbfs(total.pk[c]) > -1.0]
        dcb = [names[c] for c in range(nch) if abs(dc[c]) > 0.01]
        dead = [names[c] for c in range(nch) if rms[c] < 1e-5]
        verdict = []
        verdict.append("CLIPPING on " + ",".join(clip) + " (peak > -1 dBFS)" if clip else "no clipping")
        verdict.append("DC offset > 1 % FS on " + ",".join(dcb) if dcb else "DC ok")
        if dead:
            verdict.append("SILENT channel(s) " + ",".join(dead))
        print("verdict: " + "; ".join(verdict))
    if out and sf_out is not None:
        print(f"wrote {out} ({nch} channels: {', '.join(ch_names(nch))})")
    elif out:
        print(f"no packets received: {out} not written")

    if args.mono_check:
        if not keep:
            print("mono-check: no audio recorded"); return
        sys.path.insert(0, ROOT)
        from droneears.pipeline import Pipeline, summarise
        y = (np.concatenate(keep).astype(np.float32) / 32768.0).T          # (n_ch, n)
        t1 = time.time()
        try:
            res = Pipeline().run(y)                                        # rules only (no classifier)
        except Exception as e:                                             # e.g. an old 3-ch file vs a 4-mic config
            print(f"mono-check: pipeline failed on {y.shape[0]} channels ({type(e).__name__}: {e}); "
                  "retrying on the channel mean (detection only)")
            res = Pipeline().run(y.mean(0, keepdims=True))
        s = summarise(res)
        hits = max((r["hits"] for r in res), default=0)
        print(f"mono-check (rules only, {y.shape[0]} ch, {len(res)} frames, {time.time() - t1:.1f} s): {s}")
        th = s.get("theta_median_when_detected")
        el = [r["elev"] for r in res if r.get("detected") and r.get("elev") is not None]
        print(f"  max hits {hits}; median theta while detected: "
              + ("n/a (never detected)" if th is None else f"{th:.1f} deg")
              + "; median elev while detected: " + (f"{float(np.median(el)):.1f} deg" if el else "n/a"))


if __name__ == "__main__":
    main()
