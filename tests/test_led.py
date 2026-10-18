"""pytest: stealth mode (the board's detect LED stays off): the fake device's Q, SerialSource's '# led' handling,
POST /api/led's status codes, and the whole path end to end (fake device -> dashboard)."""
import asyncio, functools, os, socket, subprocess, sys, threading, time
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from webapp import sources
from webapp.fake_device import FakeLed


def test_fake_device_q():
    led = FakeLed()
    assert led.report() == "# led stealth=off\n"
    assert led.command("1") == "# led stealth=on saved\n" and led.stealth
    assert led.command("") == "# led stealth=on\n"                       # 'Q\n' alone: report
    assert led.command("0") == "# led stealth=off saved\n" and not led.stealth
    assert led.command("2") == "# led: bad 'Q2', want Q0 or Q1\n" and not led.stealth


def _source():
    st, msgs = {}, []
    src = sources.SerialSource("/nonexistent", lambda f: None, lambda **kw: st.update(kw), message=msgs.append)
    return src, st, msgs


def test_led_lines_update_status_and_bench():
    src, st, msgs = _source()
    src._comment("# led stealth=on")
    assert st["led_stealth"] is True
    assert msgs[-1]["type"] == "bench" and msgs[-1]["kind"] == "led" and msgs[-1]["stealth"] is True
    src._comment("# led stealth=off saved")
    assert st["led_stealth"] is False and msgs[-1]["stealth"] is False
    src._comment("# led: bad 'Q2', want Q0 or Q1")
    assert st["led_stealth"] is False and msgs[-1]["kind"] == "led" and "Q2" in msgs[-1]["error"]
    src._comment("# cal mics B=w1 L=w2 R=w3 T=w4 motors M1=0.0 src=nvs")    # not an led line
    assert st["led_stealth"] is False


def test_set_stealth_waits_for_the_board():
    src, st, msgs = _source()
    with pytest.raises(RuntimeError, match="not connected"):
        src.set_stealth(True)
    src.connected = True
    out = {}
    t = threading.Thread(target=lambda: out.update(r=src.set_stealth(True)))
    t.start()
    for _ in range(200):
        if src.cmd_q:
            break
        time.sleep(0.01)
    assert src.cmd_q.popleft() == b"Q1\n"
    src._comment("# led stealth=on")                       # a report is not the reply
    time.sleep(0.05)
    assert t.is_alive()
    src._comment("# led stealth=on saved")
    t.join(2)
    assert out["r"] is True and src.led_wait is None
    err = {}

    def rejected():
        try:
            src.set_stealth(False)
        except RuntimeError as e:
            err["e"] = str(e)
    t = threading.Thread(target=rejected)
    t.start()
    for _ in range(200):
        if src.cmd_q:
            break
        time.sleep(0.01)
    assert src.cmd_q.popleft() == b"Q0\n"
    src._comment("# led: bad 'Q0', want Q0 or Q1")
    t.join(2)
    assert "rejected" in err["e"]
    with pytest.raises(TimeoutError):
        src.set_stealth(False, timeout=0.1)


def _post_led(source, **kw):
    """POST /api/led on an app with just that route and app["source"] = source; returns (status, json)."""
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp.test_utils import TestClient, TestServer
    from webapp import server

    async def run():
        app = aiohttp.web.Application()
        app["source"] = source
        app.router.add_post("/api/led", server.api_led)
        async with TestClient(TestServer(app)) as c:
            r = await c.post("/api/led", **kw)
            return r.status, await r.json()
    return asyncio.run(run())


def test_api_led_status_codes():
    assert _post_led(None, json={"stealth": True})[0] == 409                 # demo / replay / simulate: no device
    assert _post_led(object(), json={"stealth": True})[0] == 409
    src, st, msgs = _source()
    assert _post_led(src, json={"stealth": True})[0] == 409                  # the device, not connected
    src.connected = True
    for bad in (dict(json={"stealth": "yes"}), dict(json={}), dict(json=[True]), dict(data="not json")):
        code, j = _post_led(src, **bad)
        assert code == 400 and not j["ok"], bad
    src.set_stealth = functools.partial(sources.SerialSource.set_stealth, src, timeout=0.2)
    code, j = _post_led(src, json={"stealth": True})                        # connected, but nothing answers
    assert code == 504 and "no '# led" in j["error"]


# ---------------------------------------------------------------- end to end: fake device -> dashboard API
def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


def _greeting_led(tcp):
    """Connect to the fake device as a new client: its '# led' greeting line, then its replies to 'Q\n' and 'Q0\n'."""
    with socket.create_connection(("127.0.0.1", tcp), timeout=5) as c:
        f = c.makefile("rb")
        lines = []

        def led_line():
            while True:
                ln = f.readline().decode(errors="ignore").strip()
                if not ln:
                    raise AssertionError("connection closed")
                if ln.startswith("# led"):
                    return ln
        lines.append(led_line())
        c.sendall(b"Q\n"); lines.append(led_line())
        c.sendall(b"Q0\n"); lines.append(led_line())
        return lines


@pytest.mark.skipif(os.environ.get("HH_FAST_TESTS") == "1", reason="HH_FAST_TESTS=1: skip the end-to-end test")
def test_end_to_end_stealth(tmp_path):
    aiohttp = pytest.importorskip("aiohttp")
    tcp, http = _free_port(), _free_port()
    env = dict(os.environ, PYTHONWARNINGS="ignore", TF_CPP_MIN_LOG_LEVEL="3",
               HH_CALIBRATION=str(tmp_path / "calibration.json"), HH_CAL_CAPTURES=str(tmp_path / "captures"))
    py = sys.executable
    fake = subprocess.Popen([py, "-m", "webapp.fake_device", "--tcp", str(tcp), "--no-ml"], cwd=ROOT, env=env,
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

        async def status(s):
            async with s.get(base + "/api/status") as r:
                return await r.json()

        async def post(s, body):
            async with s.post(base + "/api/led", json=body) as r:
                return r.status, await r.json()

        async def run():
            async with aiohttp.ClientSession() as s:
                st = {}
                for _ in range(100):
                    try:
                        st = await status(s)
                        if st.get("connected") and st.get("led_stealth") is not None:
                            break
                    except Exception:
                        pass
                    await asyncio.sleep(0.3)
                assert st["led_stealth"] is False, st                 # the '# led' greeting / the 'Q' reply
                assert await post(s, {"stealth": True}) == (200, {"ok": True, "stealth": True})
                assert (await status(s))["led_stealth"] is True
                assert await post(s, {"stealth": False}) == (200, {"ok": True, "stealth": False})
                assert (await status(s))["led_stealth"] is False
                code, j = await post(s, {"stealth": 1})
                assert code == 400 and not j["ok"]
                assert (await post(s, {"stealth": True}))[0] == 200
        asyncio.run(asyncio.wait_for(run(), 60))
        dash.terminate(); dash.wait(5); dash = None
        # The fake board kept it (the firmware keeps it in NVS): a new client is greeted with it.
        assert _greeting_led(tcp) == ["# led stealth=on", "# led stealth=on", "# led stealth=off saved"]
    finally:
        for p in (dash, fake):
            if p is not None:
                p.terminate()
                try:
                    p.wait(5)
                except subprocess.TimeoutExpired:
                    p.kill()
