# ARRG-HH-Drone

Acoustic drone detector with 360° haptic bearing and elevation, for the headband device. `main` merges the team's
branches (Oct 2026): `gabriel` (pipeline, C++ core, firmware, dashboard, board bring-up), `gabriel-4mic` (four mics
with elevation and eight motors: the design this tree builds), `richard-interp` (what the classifier listens to, and
the local-flatness test + DSP gate the board now runs: `interp/`) and `adam-1` (Adam's model comparison and
triangulation experiments, below). `docs/` holds the overview written for the three-mic version. The repo is software
only (Python pipeline, training, C++ core, ESP32-S3 sketch); the board itself is not part of it. Contents:

* `droneears/` — the signal-processing pipeline in NumPy: FFT front end, log-mel features, harmonic-comb detector,
  four-mic masked GCC-PHAT bearing (azimuth + elevation), fusion, haptic panning. Runs on a Mac now; written to be ported line for line.
* `firmware/core/` — the same pipeline in portable C++ (no dependencies), with a laptop harness that is verified
  frame-by-frame against the Python (`scripts/compare_core.py`: f0, score, state and bearing identical to <0.001°).
* `firmware/esp32s3/headband.ino` — ESP32-S3 firmware: four-mic PDM capture (B, L, R, T on I2S0 PDM RX, three data
  lines, one shared clock), DSP on core 1, classifier + eight-motor haptics + battery sense + serial / Wi-Fi protocol
  on core 0 (CSV telemetry per frame with bearing, elevation, battery and the applied motor weights, or raw 4-channel
  audio packets on the host's `R` command, for `scripts/telemetry.py` / `scripts/record_device.py`; raw needs the
  native USB port), plus the bench commands (tap test `L`, motor test `M`, manual motors `h`, status `S`). Every
  GPIO is in `firmware/esp32s3/pins.h`. The 3-mic / 3-motor version ran on the bench board (Oct 2026: real-time at
  47 frames/s, Wi-Fi AP up, every command verified over USB); this 4-mic / 8-motor version compiles with
  `flash.sh --build` but has not run on hardware, and its DSP load is estimated above the frame budget (see
  "Porting status").
* `scripts/train.py` — trains the drone/background classifier and exports an int8 TFLite model + C array.
* `TRAINING_PLAN.md` — data, splits, augmentation, and the fine-tune procedure after the field session.
* Adam's experiments (`adam-1`): `train_compare.py` (standard, tiny and depthwise-tiny CNNs compared on the team
  clips and the public ITU-ARIS drone set; results in `results/`), `hf_public.py`, `triangulation.py` /
  `direction_triangulation.py` (four-mic B/L/R/T direction of arrival on simulated delays; `results_direction/`)
  and `esp32s3_drone_direction_espidf.zip` (an ESP-IDF project). See `HUGGINGFACE_README.md` and
  `DIRECTION_README.md`; they read `config.json` / `direction_config.json` and use the public dataset committed
  under `data/huggingface_itu_aris/` (CC BY 4.0, ~240 MB). The model comparison needs PyTorch and librosa;
  the direction simulation uses NumPy, matplotlib and FFmpeg (`requirements.txt`). The team's
  phone recordings live in `data/drone/` and `data/non_drone/` (formerly `richard_data/`).
* `sig_proc_tri.py` — Rachel's all-pairs GCC-PHAT solver adapted to four B/L/R/T microphones. Returns a 3D unit
  direction vector using NumPy; its optional dataset experiment also needs datasets, soundfile and scipy.
  Shared geometry and usage are documented in `DIRECTION_README.md`.
* `interp/` (`richard-interp`) — what the classifier listens to (synthetic probes, edits to real recordings), and the
  DSP replacement for its veto that the board now runs: a local-flatness tonality test and a six-feature gate.
  Findings, numbers and how to reproduce: `interp/README.md`.

## Quick start on a Mac

```
brew install ffmpeg
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                   # 193 tests: bearing + elevation, overhead, 8-motor pan, 9 real clips + ambient (rules and fused), mic tilt, DSP gate, C++ == Python, calibration, LED, battery badge, four-mic triangulation (sig_proc_tri*)

python scripts/demo_direction.py            # drone circles the head -> artifacts/direction_demo.png
python scripts/run_file.py some.m4a         # timeline of detector + classifier on any recording
python scripts/live_mac.py                  # live on the laptop mic (detection only; bearing needs 4 channels)
python scripts/record_mac.py --drone --seconds 60 --check   # record training/holdout audio from the laptop or iPhone mic into data/team/
python scripts/eval_esc50.py                # false-alarm rate over all 2000 ESC-50 clips (--no-ml for rules only)
python scripts/get_dads.py                  # ~1.7 GB: public DADS drone audio, 6 parquet shards -> data/hf-drone/
python scripts/train.py --epochs 30 --n-synth 60   # retrain + export; uses data/drone, data/non_drone, data/team/*, DADS, ESC-50
```

ESC-50 is expected at `data/ESC-50` (`git clone --depth 1 https://github.com/karolpiczak/ESC-50 data/ESC-50`) or set
`ESC50_DIR`. DADS shards go in `data/hf-drone` (or `DADS_DIR`); the loader skips the public data if it is absent.

## Device bring-up, plug and play

1. **Wire / check the board**: the mics, motors, LED and battery sense per `firmware/esp32s3/pins.h` (the single source of truth for
   every GPIO; the tables below are copied from it). ESP32-S3-DevKitC-1, Adafruit PDM MEMS breakouts (ST
   MP34DT01-M) on I2S0 PDM RX at 48 kHz with a 3.072 MHz PDM clock; each data line carries two mics, selected by the
   breakout's SEL strap (GND = left slot, 3V3 = right slot). This build has **four** mics (B, L, R, T); the map's
   FRONT position is not populated and its slot is ignored. Mics: 3V to 3V3, GND to GND.

   | Mic | Position | CLK | DAT | SEL | PDM slot | WAV / ring channel |
   |---|---|---|---|---|---|---|
   | F | forehead (**not populated**) | GPIO 4 | GPIO 5 (DIN0) | GND | line 0 left | (slot 0, ignored) |
   | B | back of the head | GPIO 4 | GPIO 5 (DIN0) | 3V3 | line 0 right | 0 |
   | L | left temple | GPIO 4 | GPIO 6 (DIN1) | GND | line 1 left | 1 |
   | R | right temple | GPIO 4 | GPIO 6 (DIN1) | 3V3 | line 1 right | 2 |
   | T | crown (top of the head) | GPIO 4 | GPIO 7 (DIN2) | GND | line 2 left | 3 |

   Eight coin motors on a ULN2803A (low side, active-high inputs), one LEDC channel each, 200 Hz, 8-bit, duty capped
   at 190/255, **never more than 3 energised at once** (the PowerBoost supplies 1 A):

   | Motor | F | FR | R | BR | B | BL | L | FL |
   |---|---|---|---|---|---|---|---|---|
   | ULN2803A | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 |
   | GPIO | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 21 |
   | Bearing | 0° | 45° | 90° | 135° | 180° | −135° | −90° | −45° |

   | Other | Pins |
   |---|---|
   | IMU BNO085 (I2C 0x4A, pull-ups on the breakout; not used yet, `USE_IMU 0`) | SCL GPIO 8, SDA GPIO 9, INT GPIO 10 (active low), RST GPIO 11 (active low) |
   | Detect LED | GPIO 2 (DevKitC header pin 40), high = on: solid while DETECTED, a short blink every 3 s when idle, other states blink their own patterns (table at the top of `headband.ino`); the dashboard's *Stealth* button (`Q1`/`Q0`, kept in NVS) keeps it off |
   | Battery sense | GPIO 1 = ADC1 channel 0, 1:2 divider (V_batt = 2 × V_adc), 4.20 V full, 3.30 V empty; read once a second |
   | Reserved | GPIO 0, 3, 45, 46 (strapping), 19/20 (native USB), 43/44 (UART0), 26–37 (flash / PSRAM) |
   | Free spares | 3, 47, 48 (PowerBoost LBO not wired) |

   There is no buzzer on this board (GPIO 9 is now IMU SDA, and the eight motors use all eight LEDC channels).
   The bench board of Oct 2026 was wired for the 3-mic firmware (mics on DIN0 SEL GND, DIN0 SEL 3V3 and DIN1; motors
   on M1-M3): under this map its DIN0 SEL GND mic is the ignored F position, so rewire it before trusting bearing or
   elevation (detection works with any subset).
2. **Build and flash**: `firmware/esp32s3/flash.sh` compiles with arduino-cli (ESP32 core 3.x, board
   `esp32s3`, USB CDC on boot, huge_app partition, octal PSRAM: the sample rings, spectra and raw packet pool live
   there, otherwise Wi-Fi has no heap left) and uploads to the S3's native USB port (the Espressif "USB JTAG/serial"
   device, `/dev/cu.usbmodem*`); `flash.sh --build` only compiles. The board's second USB serial device (a CH343
   UART bridge, `/dev/cu.usbmodem5B...`) is the UART0 console: boot log and crash output, telemetry too, but it
   drops raw audio. (`USE_TFLM 1` additionally needs esp-tflite-micro.)
3. **Watch it run**: `python scripts/telemetry.py --log run1.csv` shows state, f0, bearing, elevation, classifier,
   battery voltage and the eight applied motor weights live (47 lines/s from the board; each T line ends with `bstat`,
   0 no bearing / 1 ambiguous / 2 three mics / 3 full 3D, and `mused`, the mics in this frame's fit as a bitmask
   B=1 L=2 R=4 T=8; theta and elev are empty when unknown); `python
   scripts/telemetry.py --plot run1.csv --out run1.png` plots a log afterwards (score, classifier, azimuth,
   confidence, elevation against the 60° overhead line, and a motor-by-time heat map). Both read the columns by name,
   so logs from the 3-mic firmware still work. The first `# pdm chunk_us=` line should read ~21333 (another value
   means the 6-word DMA frame assumption is wrong; also `pdm_us=` in the `# status` reply to `S`, with free heap /
   PSRAM, the DSP task's stack headroom and, in the `# dsp` line, the per-stage DSP time against the 21.3 ms
   budget; the `# boot` line after the banner says whether every task started and how many bulk buffers landed in
   PSRAM).
4. **Tap test**: `python scripts/telemetry.py --levels` prints a `# levels` line 4x a second with the RMS, peak
   and DC of all six PDM DMA words (`w0`/`w1` DIN0 SEL GND / 3V3, `w2`/`w3` DIN1, `w4`/`w5` DIN2) and which word
   each DSP channel reads. Tap each mic with a fingernail: one word's `peak` jumps per mic. If B, L, R, T are not on
   words 1, 2, 3, 4, fix `CH_SLOT[]` in the sketch (the one table that maps DMA words to channels; its comment gives
   the most likely alternative) and re-flash. `python scripts/record_device.py --monitor` then shows the four DSP
   channels (B, L, R, T) with DC and clipping.
5. **Motor test**: `python scripts/telemetry.py --motor-test` buzzes F, FR, R, BR, B, BL, L, FL in turn (0.5 s each,
   looping); the `hap_*` bars in the status line say which one is meant to be on. Swap the `PIN_MOT_*` in `pins.h` if
   the felt position is wrong.
6. **Record the held-out session** through the real mics: `python scripts/record_device.py --drone --holdout
   --seconds 120 --mono-check` for each drone pass, `--holdout --seconds 300` (no `--drone`) for backgrounds,
   including your own motors buzzing and, if you can find one, a lawn mower. Files land in `data/team/` and are
   never committed. Without the board, `python scripts/record_mac.py --drone --holdout --seconds 60 --check`
   (`--device iphone` for the phone over Continuity) records mono training/holdout clips with the same naming.
7. **Retune on that session only**: `python scripts/retune_holdout.py` sweeps the frame thresholds and persistence,
   reports drone clips detected vs false alarms on your backgrounds and ESC-50, and `--apply` writes the winner
   into `droneears/config.py` and regenerates `firmware/core/dsp_config.h`. If the classifier is what fails, drop
   the clips into `data/team/` and `python scripts/train.py --epochs 30 --n-synth 60`, then copy
   `artifacts/model_data.h` to `firmware/esp32s3/`. Re-flash.

**Wireless.** With `USE_WIFI 1` (default) the board also serves the same protocol over Wi-Fi on TCP port 5000, so
it can be worn on a USB power bank (Wi-Fi adds ~100 mA) while a laptop watches. Join the board's own network
`HEADBAND` (WPA2 password `dronedetect` unless changed; the board is 192.168.4.1), or copy
`firmware/esp32s3/wifi_config.h.example` to `wifi_config.h` (gitignored) and set `WIFI_STA_SSID` / `WIFI_STA_PASS`
to put it on an existing network (it falls back to the AP after 15 s). Then add `--tcp` to any tool:
`python scripts/telemetry.py --tcp`, `python scripts/record_device.py --tcp --seconds 60`,
`python scripts/dashboard.py --tcp` (default `headband.local:5000`, falling back to `192.168.4.1:5000`; or
`--tcp HOST[:PORT]`); all of them reconnect if the link drops. One TCP client at a time: a new connection replaces
the old one. Raw audio over Wi-Fi is best effort (a slow link drops whole chunks, reported as zero-filled gaps);
USB is the reliable path for the held-out recording (raw 4-channel audio is ~384 KB/s; packets carry a 12-byte
header with the channel count, and `record_device.py` also still reads the 3-mic firmware's 8-byte format).
`python -m webapp.fake_device --tcp 5050` fakes the board on a TCP port for testing.

Haptics: eight motors 45° apart. The board pans between the two neighbours bracketing the bearing, so at most two
run for a side bearing: a drone straight ahead buzzes F, 20° right buzzes F and FR, left buzzes L, back-left BL.
Overhead (elevation above 60° for ~0.5 s, measured directly thanks to the top mic T; released below 45°) plays a
ring instead of buzzing everything: every 100 ms three motors 135° apart (k, k+3, k+6 of 8) are on and the triplet
steps one motor round, so it feels like a spinning ring and never exceeds the 3-motor limit (the firmware also
enforces that limit on any weights). Pulse rate rises with harmonic strength; the overhead ring runs continuously.
The `hap_*` telemetry columns report what is actually applied, so during the overhead chase they show the current
triplet.

## Live dashboard

`python scripts/dashboard.py` opens a browser tab (http://localhost:8765) showing what the headband hears and
decides, live: a state banner (IDLE / DRONE DETECTED / DRONE OVERHEAD, time since the alarm, evidence-to-alarm
latency, bearing and elevation), a bearing radar around a top-down head (nose up, +90° = right, 180° = behind; the
wedge shortens toward the centre as the drone climbs, since that is where it projects in a top view, and a side-view
gauge in the corner shows the elevation in degrees against the 60° overhead line; overhead shows an ABOVE disc), the
eight haptic motors around the head lit by the weights the board applies (overhead: the spinning triplet, 3 of 3
max), 20 s charts of harmonic score, rule hits, classifier p(drone) and f0 against their thresholds, device health
(including battery volts and percent, linear 3.30–4.20 V) and an event log. The header's battery badge (`static/battery.js`) shows the charge from a Li-ion discharge curve with a 5-minute trend and time left, and warns once below 15 % and 5 % ("charging?" is a guess from rising voltage: the PowerBoost's charge pin is not wired). One process (`webapp/server.py`, needs `aiohttp`), no build step, works offline. Sources:

```
python scripts/dashboard.py --port                     # the board over USB (auto-detect; or --port /dev/cu.usbmodemXXXX)
python scripts/dashboard.py --tcp                      # the board over Wi-Fi (headband.local:5000; or --tcp HOST[:PORT])
python scripts/dashboard.py --replay run1.csv          # a scripts/telemetry.py --log CSV, real time, looping
python scripts/dashboard.py --simulate artifacts/example_drone_4ch.wav   # Python pipeline over a 1- or 4-ch (B L R T) file
python scripts/dashboard.py --demo                     # synthetic pass: right, low -> behind, climbing -> overhead -> front-left
```

With no flag it uses the board over USB if a serial port is found, else over Wi-Fi if `headband.local:5000`
answers within 2 s, else the demo; the device panel shows the transport (USB / Wi-Fi) and host. Serial mode sends `T` on connect and `S`
every 2 s (free heap, dropped chunks, battery) and reconnects when the cable is replugged. `--http-port`, `--host 0.0.0.0`,
`--no-browser`, `--no-ml` are available. `python -m webapp.fake_device` puts a fake board (the demo scene, in the
firmware's line format) on a pty for testing the serial path without hardware. The page also reads
`GET /api/status` and a WebSocket at `/ws` (one JSON object per frame plus status, event and bench messages).

**Bench panel** (the `Bench` button in the header, or open http://localhost:8765/#bench): the board tests from
steps 4 and 5 of the bring-up as buttons, with the device over USB or Wi-Fi. *Start tap test* polls the board's
`L` command 4x a second and shows one bar per PDM slot (w0/w1 = DIN0 SEL GND / 3V3, w2/w3 = DIN1, w4/w5 = DIN2),
peak and RMS in dBFS, DC offset, the DSP channel (B / L / R / T) that reads each slot, and the same plain-language
hints as
`telemetry.py --levels` (a slot pair that is identical means one mic per line; a DC offset near 50 % means two
mics fighting on the same SEL). Tap each mic and note which slot jumps, then fix `CH_SLOT[]` in the sketch.
*Buzz each in turn* sends `M`: the pads under Haptics show which of the eight motors should be buzzing while you
feel for it; swap the `PIN_MOT_*` in `pins.h` if they disagree. The motor toggles drive any motor by hand (at most
three at once, like the board). *Status now* sends `S`; the Board column shows the DSP load against the
21.3 ms frame budget (per-stage microseconds, DMA overflows), chunk period, free heap and PSRAM, the DSP task's
stack headroom, the boot line (every task started, bulk buffers in PSRAM) and the Wi-Fi state.
*Record* collects training audio through the board's own mics: pick `drone` or `background`, a length (10 s to
5 min), optionally `holdout` (the whole clip goes to the test split) and a tag for the file name, press Record.
The board streams raw 4-channel audio (the detector display pauses) into `data/team/<label>/device_<time>_<tag>.wav`,
exactly as `scripts/record_device.py` would; the panel shows progress, dropped chunks, per-channel peak / RMS,
the clipping / DC verdict and then the rules-only detector over the clip (a drone clip should come out mostly
DETECTED, a background clip at 0 %), plus a running count of clips and minutes per label and a player: Finder
and QuickTime will not play the multi-channel originals, so the panel plays a listening mix (L + R, mono, or one
channel, peak-normalised; `GET /api/clip`) of the newest clip or any of the last eight. Then retrain with
`python scripts/train.py --epochs 30 --n-synth 60` and copy `artifacts/model_data.h` to `firmware/esp32s3/`.
USB is the reliable transport for recording; over Wi-Fi a slow link shows up as dropped (zero-filled) chunks.
Measured on the v3 board, the S3's USB CDC delivers ~256 KB/s, less than the 4-mic raw stream's ~384 KB/s, so a
long *Record* over USB also loses some chunks (reported as dropped).

**Calibrate tab** (the `Calibrate` button, or http://localhost:8765/#calibrate): tells the headset which mics and
motors are connected, which mic is where on the head and which motor sits in which hole of the band, without
reflashing. The headset has four mics: left, right, top, and one of back or front (B or F: they share DSP channel 0,
so a front mic replaces the back one).
- *Mics*: one row per PCB header MK1–MK5 with a *Connected* tick and its position (front, back, left, right or top;
  at most four headers connected, one per position, front and back exclusive: picking one unticks the header that had
  the other). Defaults: MK2 back, MK3 left, MK4 right, MK5 top, MK1 unticked but assignable to any position. Each row
  has a live level bar and badges: "no signal" (ticked but flat), "hears sound" (unticked but live), "= MKn" (on a
  data line with one mic, the empty SEL phase mirrors it). Which DMA word a header lands in depends on the *slot
  order*: on this board the right (SEL 3V3) slot comes first on each line (verified on the v3 board with MK1–MK4
  fitted, the default), so MK1–MK5 are w1, w0, w3, w2, w5 and w4 has no header; "left first" is still selectable.
  Under the rows the tab says what the board can do with the ticked mics (front behaves exactly like back):

  | Ticked mics | What the headset gives |
  |---|---|
  | B or F, L, R, T | bearing and elevation; a mic that is weak or drops out for a moment is left out and the best three carry on |
  | B or F, L, R | bearing; elevation only when the drone is clearly above the head (assumed above) |
  | B or F, with L or R missing | bearing and elevation when the drone is clearly above the head, otherwise ambiguous |
  | L, R, T | front and back cannot be told apart: no reliable bearing (ambiguous) |
  | two or fewer | detection only (the motors pulse the overhead ring) |

  The DSP uses exactly the ticked mics (detection from their mean; unticked ones skip their FFT). Set positions by
  hand, or *Auto-calibrate*: stand ~1 m in front of the headset and clap sharply twice while it records 3 s, then to
  its right, then to its left (back and top optional), and *Solve*. The headset records each 3 s of all six words
  into its PSRAM and then sends them (six words in real time are over twice what its USB carries, so a capture
  arrives ~7 s after the click, complete). The solver (`webapp/calibration.py`) finds the clap, times its arrival at
  every live word by GCC-PHAT and picks: right = the first word to hear the right clap, left = the first for the
  left clap, front = the word clearly first on the front clap, else back = the word clearly last on it (or the first
  for a back clap), top = the first for a top clap or the one live word left over. Every pick must lead (or lag) all
  other mics by at least 0.08 ms, else that position stays empty with a note (so a headset with a missing mic is not
  guessed); a line with one mic reads it in both words and the position goes to the word with a header. *Use this*
  ticks the mics it found.
- *Motors*: rows M1–M8 (ULN outputs; defaults F, FR, R, BR, B, BL, L, FL) with a *Connected* tick (the detector
  never drives an unticked one; ticking it again puts it back in its hole) and the hole each sits in. The holes and
  the band outline come from `HH_CAD_V4_loops.stl` via `scripts/stl_geometry.py` -> `webapp/static/headband_geometry.json`
  (8 round ~11 mm holes at 0, ±36.7, ±82.9, ±144 and 180 deg, so not 45 deg apart; the front is the STL's -y end,
  `--front +y` flips it; render: `docs/headband_geometry.png`). *Buzz* runs one output for 0.6 s; *Find by feel*
  buzzes the connected outputs in turn and you click the hole where you feel it. A motor takes its hole's measured
  angle.
- *Save* writes `data/calibration.json` (tracked: it describes this headset); *Apply to headset* sends it to the
  board, which keeps it in NVS: `C<w0>,<l>,<r>,<t>` (the PDM word of the back, left, right and top mic; -1 = not
  connected), or `C<w0>,<l>,<r>,<t>,F` when channel 0's mic is at the front, and `Y<m1>,...,<m8>` (motor angles in
  degrees, >= 900 = not connected; the detector pans between the two connected motors bracketing the bearing). `K`
  prints the board's current calibration (`# cal mics B=w0 L=w3 R=w2 T=w5 motors ... src=nvs`, `F=` for a front
  mic; that is also the compiled default map), and `W<n>` is the capture command. Without hardware: `python -m
  webapp.fake_device --tcp 5050 --wiring front=0,left=3,right=2,top=5 --cal-script front,right,left` fakes a headset
  with a front mic and claps from those sides (default wiring `back=0,left=3,right=2,top=5`).

## How the pipeline works (one frame = 2048 samples at 48 kHz, every 1024 samples)

1. **Spectra.** Each mic channel is Hann-windowed and FFT'd. The mono power spectrum feeds the detector and the
   classifier; the four complex spectra feed the bearing. One FFT per channel per frame is the whole compute budget.
2. **Harmonic detector** (`detector.py`). The dB spectrum minus a smooth, peak-ignoring spectral baseline gives
   *prominence*, averaged over frames (EMA, weight 0.5 on the new frame). A comb scores every candidate f0 from
   120 to 400 Hz in 2 Hz steps as the mean prominence of five consecutive harmonics starting from the first one
   ≥ 300 Hz (below that, wind and handling rumble dominate real recordings). The best f0 wins, with hysteresis: a
   candidate within 6 % of the previous f0 is kept if it scores within 0.5 dB of the best, so the track does not
   hop between near-equal comb fits (sub-harmonics, two motors at slightly different RPM). A frame counts as drone
   evidence if score > 5.5 dB, ≥ 3 of the 8 harmonics from that first one are > 6 dB proud, the *local* flatness
   in 300–3000 Hz (flatness of the prominence spectrum, i.e. with the smooth spectral shape divided out, so a
   different microphone response cancels) is above 0.3 (a broadband rotor/airflow floor between the harmonics,
   which rules out pure tones such as sirens, alarms and bells), and the f0 track is steady (≥ 35 % of the last
   ~1 s of f0 estimates within 6 % of their median, checked once 0.5 s of history exists). DETECTED needs 70 of the
   last 94 frames (~1.5 s of evidence in a 2 s window) **and the DSP gate open**, and releases after 2 s with ≤ 20.
   The gate is a weighted sum of six ~1 s rolling measurements that separate rotors from the rules' false alarms
   (prominence at the shaft rate f0/2 and below 300 Hz up; f0 jumps, spectral flux, level wobble and off-comb peaks
   down; `GATE_*` in `config.py`); it does the classifier's veto on the board, where the classifier is off
   (`USE_TFLM 0`). This replaced global flatness between 0.17 and 0.45, which rejected 97 % of the public DADS drone
   recordings and lost 7 of 9 phone drones under a −3 dB/octave mic tilt (`interp/README.md`; `FLAT_MODE = "global"`,
   `GATE_ON = False` restore it). Measured on all 2000 ESC-50 clips: 0.62 % of frames / 31 clips false-positive
   (washing machine, chainsaw, insects, airplane, engine), against 2.2 % / 112 for the old rules alone; DADS shard 36
   drone recordings 131 of 150 detected (old rules: 5). On the 9 phone recordings in `data/drone/`
   (channel 0) all 9 reach DETECTED (first detection 2.0–9.1 s; 20–82 % of frames), and 0 % of the 237 s ambient
   recording (the rules alone fire there; the gate vetoes it). The thresholds were chosen on dev data that includes
   these phone clips, so treat 9/9 as optimistic until a held-out board session exists;
   `scripts/eval_esc50.py --no-ml`, `python -m interp.final` and `pytest -q` re-measure all of this.
3. **Classifier** (`scripts/train.py`, `classifier.py`). 47×40 log-mel window → 16 k-parameter CNN (24 KB int8,
   0.06 ms/window on a laptop) → p(drone), twice a second. Trained on the 9 phone recordings in `data/drone/`
   (~96 s, each split 70/30 in time), 200 segments of the public DADS drone dataset (`scripts/get_dads.py`) and a
   minority of synthetic combs, against ESC-50 folds 1–4, DADS no-drone clips and `data/non_drone/`. Held-out
   split (ESC-50 fold 5, the last 30 % of each team clip, an unseen DADS shard): int8 accuracy 94.9 % / recall
   88 % / FPR 3.2 % at p > 0.5; `team:drone` recall 84 %, `team:background` 0 % false positives, ESC-50 1.4 % at the
   0.7 gate (airplane, insects, chainsaw, door creaks). **Fusion**: the classifier only gates the onset of a
   detection (p > 0.7 on its latest update or any update in the last ~1 s), together with the DSP gate; it can
   never trigger one alone. With the new rules and both gates (the laptop tools), all 2000 ESC-50 clips: 0 false
   positives; with the old rules it was 0.19 % of frames / 11 clips. What the classifier listens to (the drone's
   fundamental in 200–400 Hz, f0 150–350 Hz only, the range of our synthetic training drones) and why it costs
   recall on noisy and tilted drones: `interp/README.md`. Numbers and procedure: `TRAINING_PLAN.md`.
4. **Bearing and elevation** (`bearing.py`). Four mics: B (nape), L, R in the head plane and T on the crown.
   Rachel's robust solver (`sig_proc_tri_robust.py`), ported to the frame loop in Python and C. For each pair of
   connected mics, the cross-spectrum conj(Xi)·Xj, PHAT-normalised and masked to the bins at k·f0 (the drone's
   harmonics, so other sources are excluded), is evaluated directly at the integer lags within ±(baseline/c)
   (21–28 lags; exact, and far cheaper than an inverse FFT), peak with parabolic interpolation → delay τ; a second
   peak ≥ 0.9× the best inside that range marks the pair aliased (no delay). Each pair also gets a bias-corrected
   coherence from a running average of its cross-spectra over consecutive frames. Reliable pairs (a delay, and
   coherence ≥ 0.35) enter a weighted least-squares fit (weight coherence² × peak, Huber reweighting, residual
   outliers over 3 × 0.5 samples dropped); if a four-mic fit fails her consistency checks, each mic is left out in
   turn and the best passing three are kept. That is the per-frame "best 3 of 4": a dead, weak or noisy mic drops
   out on its own, and `mused` in the T line says which mics were used. Outcomes: four mics → full 3D (`bstat` 3);
   three in a level plane (B/L/R) → azimuth, plus elevation when the "drone above the head" prior
   (`BEARING_UPPER_PRIOR`, +z, through her delay-uncertainty interval) separates the two mirror candidates
   (`bstat` 2); other three-mic sets → resolved by that prior when the drone is clearly above, else ambiguous
   (`bstat` 1: only what both mirror candidates share is reported; with the back mic missing that is the elevation,
   never the azimuth); fewer than three → no bearing (`bstat` 0). Her opt-in "source-facing" prior is not used: it
   gives confident wrong answers for dead or disconnected mics. `conf` is her confidence (clean four-mic frames
   ~0.4–0.7; frames below `BEARING_MIN_CONF` = 0.2 do not move θ). θ comes from an EMA of the 3D unit vector while
   DETECTED (so near-zenith frames, whose azimuth is ill-conditioned, weigh little), held for up to ~1 s of frames
   without an azimuth, then unknown (the motors pulse the ring); elevation is reported per frame when known.
   Overhead = elevation > 60° for ~0.5 s while DETECTED, released after ~0.5 s below 45°.
   Simulated (synthetic drone, plane-wave arrivals, white noise, θ = 60°): elevation 0–85° is recovered to
   0.8–1.2° median per-frame error at 10 dB SNR and 1.3–2.1° at 0 dB (detector bypassed at 0 dB: it does not
   fire); azimuth 0.8–2.1° per frame up to 70° elevation. A drone circling the head and rising to 75° mid-scene over
   the real ambient recording at 3 dB SNR (`scripts/demo_direction.py --bg data/non_drone/ambient_adam.m4a
   --snr 3`) is tracked with 4.3° median azimuth error (almost all EMA lag) and 1.0° median elevation error, and
   is flagged overhead for 89 % of the frames above 60°. The ambient is mixed with an independent offset per mic,
   i.e. spatially uncorrelated, which is easier than a real directional field.
5. **Haptics.** Eight motors 45° apart (F, FR, R, BR, B, BL, L, FL); pan linearly between the two bracketing θ,
   pulse rate from strength. Overhead: every motor weight 1.0 (the firmware decides how to drive that).

Mic geometry (placeholder positions, to be measured on the assembled band), motor angles and every threshold
live in `droneears/config.py`; `scripts/export_c_headers.py` regenerates `firmware/core/dsp_config.h` from it
(including `N_MICS`, `N_PAIRS`, `N_MOTORS`, `A_PINV`) so the C core cannot drift.

## Porting status

| Piece | Python | C++ core | ESP32 |
|---|---|---|---|
| FFT, Hann, power | ✓ | ✓ (radix-2; swap for esp-dsp) | compiles against core |
| Log-mel + window normalisation + int8 quant | ✓ | ✓ | ✓ (feeds TFLM) |
| Baseline, harmonic search, flatness, state machine | ✓ | ✓ bit-matched | ✓ |
| Masked GCC-PHAT, least squares, EMA, haptic pan | ✓ | ✓ bit-matched | ✓ |
| Overhead detection from elevation (top mic T; elev > 60° for 0.5 s, release < 45°), spinning 3-motor ring | ✓ | ✓ bit-matched | ✓ (ring + 3-motor limit in the sketch) |
| Local-flatness tonality test, DSP gate (six rolling features, replaces the classifier veto on the board) | ✓ | ✓ matched (gate to 1e-4) | compiles (`flash.sh --build`); not yet timed on the S3 |
| Classifier inference | TFLite interpreter | — | TFLM (`USE_TFLM 1`, arena 60 KB) |
| PDM capture, four MP34DT01-M mics (B, L, R, T) on one I2S0 port (3 data lines, 6-word DMA frame, shared clock) | — | — | compiles; untested (the 3-mic, 2-line, 4-word version ran: chunk period 21.5 ms, 0 DMA overflows) |
| Eight-motor haptics (ULN2803A, 200 Hz LEDC, duty ≤ 190/255, ≤ 3 on), detect LED, battery sense | — | — | compiles; untested (3 motors verified on the bench) |
| Serial / Wi-Fi protocol: telemetry (`T`, with elev, batt_v, 8 motors), raw 4-ch audio (`R`, 12-byte header), status (`S`), mic levels (`L`, 6 words), motor test (`M`), manual motors (`h`, 8 duties) | — | — | compiles; the 3-mic version of every command was verified over USB (`R`: 288 KB/s, 0 dropped chunks in a 4 s monitor run) |
| IMU (BNO085) head-yaw compensation | — | — | pins only (`USE_IMU 0`) |

Memory and timing on the ESP32-S3, measured on the bench board with the 3-mic firmware (`S` command). With
everything static the first build had 263 KB of globals and no heap left: Wi-Fi's tasks and the telemetry task failed
to start, silently. So the copied-only buffers (int16 sample rings, raw packet pool, TCP carry) live in the module's
8 MB octal PSRAM, the `Spectrum`s (8.2 KB each, `power` is no longer stored) are internal heap, and the rest stays
static (`dsp_core.cpp` 57 KB: FFT tables 20 KB, FFT scratch 16 KB + 16 KB esp-dsp interleave, mel 8.5 KB; `Detector`
17 KB, `BearingEstimator` 16 KB, `FrameEvidence` 4 KB, `ClsWindow` 7.5 KB; PDM chunk, frame/mono/log-mel 12 KB):
176 KB of globals and 39 KB of internal heap free after Wi-Fi came up, plus the 60 KB TFLM arena when `USE_TFLM` is
1. Per frame (budget 21.3 ms): spectra 4.6 ms, log-mel 0.5 ms, detector 3.1 ms, bearing 7.1 ms = 15.3 ms; 0 DMA
overflows. That needed three things on top of the straight port: esp-dsp's assembly FFT (`dsps_fft2r_fc32`; the
radix-2 C version took 3.3 ms each), compiling at -O2 (`flash.sh`), and no `double` in the per-frame path (the S3
emulates it in software: the baseline's per-bin double window sum alone cost 45 ms per frame; it is now an O(N)
compensated-float sliding sum, and log-mel / flatness accumulate in float).

The 4-mic / 8-motor build (177 KB of globals by `flash.sh --build`), measured on the v3 board (2026-10-17, no drone,
so the bearing runs on the wider band mask): 14.1–14.3 ms per frame with four mics connected (spectra 6.1, detector
3.4, robust bearing 3.9–4.1), 10.8 ms with three, 5.6 ms with one, and no DMA overflows. Before the robust port the
bearing alone took 14.3 ms (six 2048-point inverse FFTs) and four mics overran the budget at 24.6 ms with ~36 DMA
overflows a second. `free_heap=` with Wi-Fi up is ~19 KB at boot; drop `DMA_DESC_NUM` to 8 (42.7 ms of DMA ring) if
Wi-Fi runs short. The host harness still matches the Python reference (`pytest -q`, `scripts/compare_core.py`,
including `--active`, `--weak-ch` and `--noisy-ch` cases).
