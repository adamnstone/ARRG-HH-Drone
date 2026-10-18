"""Live web dashboard for the headband: what the device hears and decides, in a browser tab.

Modes (one of; with none: the device over USB, else over Wi-Fi at headband.local:5000, else the demo):
  python scripts/dashboard.py --port [PORT]        the headband over USB serial (auto-detect when PORT is omitted)
  python scripts/dashboard.py --tcp [HOST[:PORT]]  the headband over Wi-Fi (default headband.local:5000)
  python scripts/dashboard.py --replay run1.csv    replay a `scripts/telemetry.py --log` CSV in real time, looping
  python scripts/dashboard.py --simulate FILE      Python pipeline over a 1- or 4-channel (B L R T) file, looping
  python scripts/dashboard.py --demo               Python pipeline over a scripted synthetic drone pass (no files)
Options: --http-port 8765, --host 127.0.0.1, --no-browser, --no-ml, --model PATH. Implementation: webapp/server.py.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from webapp.server import main  # noqa: E402

if __name__ == "__main__":
    main()
