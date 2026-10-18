## Electronics

The headband has to hear in five directions at once, know exactly where your head is pointing, and
buzz in eight directions. We detail our selected components below:

| | |
|---|---|
| **Microcontroller** | ESP32-S3 (DevKitC-1 N8R8: 8 MB flash, 8 MB PSRAM, native USB, Wi-Fi) |
| **Microphone Input Devices** | 5 × PDM MEMS microphones: front, back, left, right and crown |
| **IMU** | BNO085 9-DoF IMU with on-chip sensor fusion |
| **Haptics Output Device** | 8 × 10 mm vibration motors, 45° apart, driven by a ULN2803A |
| **Power** | 1200 mAh LiPo + PowerBoost 1000C (5.2 V, 1 A), about 2.3 h per charge |
| **Board** | Two-layer carrier PCB, 115 × 66 mm |

### Mic Data Syncing

Direction finding works on tiny differences in when a sound reaches each mic. Across a head, the
biggest possible difference is about half a millisecond, roughly 25 samples at 48 kHz. Any timing
skew between channels eats directly into accuracy, so all five mics run off a single shared
clock from the ESP32.

PDM mics have a select pin that makes them output on either the rising or the falling clock edge,
and go silent the rest of the time. That lets two mics share one data wire, so five mics need just
one clock pin and three data pins (5 / 2 -> 3), and every channel is sampled at exactly the same instant. The
clock is star-routed with its own 33 Ω series resistor per mic, placed at the ESP32 end, to mitigate interference from the long wires.

### IMU 

The BNO085 does sensor fusion on the chip and hands us a ready-made orientation quaternion. Adafruit
warns that its I2C interface doesn't play well with the ESP32-S3, so we run it over UART instead.

The BNO picks its interface from a pin it reads when it comes out of reset. Instead of hard-wiring
that pin, we drive it from a spare GPIO through a 1 kΩ resistor. Firmware can then choose UART, or
fall back to I2C on the same two wires by resetting the chip. 

### Motors

A ULN2803A packs eight low-side Darlington switches into one chip, with flyback diodes built in,
and its inputs take 3.3 V PWM straight from the ESP32. The ULN acts essentially as a "switchboard" for all eight motors. At most two neighbouring motors run at once.
That's enough to place the buzz *between* motors for finer direction cues, and it keeps the power
budget in check. A 470 µF capacitor at the driver soaks up motor start-up current so it doesn't
brown out the microcontroller.

### Power

The PowerBoost charges the LiPo and boosts it to 5.2 V, and a slide switch on its enable pin turns
the whole band on and off. The ESP32 is fed through a diode, so plugging a laptop into it for
development doesn't back-feed the battery rail. While tethered, the laptop powers the ESP32 and the
battery keeps driving the motors. Only the ESP32 branch goes
through the diode, so the motors keep the full 5 V. A divider on the
ESP32's ADC tracks the battery voltage through firmware.


### Schematic

![Schematic](devpost-images/schematic_v3.png)

The whole design fits on a single KiCad sheet, and it reads most easily as five subsystems.

**Microcontroller.** The ESP32-S3 DevKit on the left is the hub that every other subsystem
connects to. It takes its 5 V from the main rail through D2, an SS34 Schottky diode drawn just
above it, and its onboard regulator produces the 3.3 V rail that powers the microphones and the
IMU. Every pin we don't use, including the strapping, USB and PSRAM pins, is marked no-connect so
nothing can be wired to it by mistake.

**Microphones.** The five mics sit in a column on the right. A single clock pin, `PDM_CLK`, feeds a
bank of five 33 Ω resistors (R4–R8, top center), which split it into `PDM_CLK1` through
`PDM_CLK5`, one branch per mic. The front and back mics share the `PDM_DIN0` data line, left and
right share `PDM_DIN1`, and the crown mic has `PDM_DIN2` to itself. Each mic's SEL pin is tied to
either ground or 3.3 V, which decides whether it answers on the rising or the falling clock edge.

**IMU.** The BNO085 sits in the middle. Its SCL and SDA pins carry the UART link to the ESP32,
alongside separate interrupt and reset lines. Its P1 pin, which selects UART or I2C, is driven by
GPIO41 through R9 instead of being hard-wired.

**Haptics.** Eight GPIOs, labelled `M1` to `M8`, drive the inputs of the ULN2803A to the right of
the IMU. Each output pulls the negative lead of one motor to ground, while every motor's positive
lead sits on +5 V; the motors run in a row along the bottom of the sheet. The driver's COM pin is
also tied to +5 V, which enables its built-in flyback diodes, and C1 (470 µF) sits across the +5 V
rail. The detection LED, D1, hangs off GPIO2 through R3.

**Power.** The bottom-left corner holds the power path. The battery plugs into the PowerBoost,
which produces the +5 V rail, and SW1 turns the band off by pulling the PowerBoost's enable pin to
ground. R1 and R2 halve the battery voltage so the ESP32's ADC can read it on `VBAT_SENSE`, and C2
smooths that reading.

### PCB

Everything plugs into a two-layer carrier board: the ESP32 DevKit, IMU, power
board and motor driver, plus a connector for every mic and motor. The board does two important
jobs:

- **It holds the IMU rigidly to the band.** Head tracking only works if the sensor moves exactly
  with the head.
- **It makes every mic cable identical.** The select-pin strapping that gives each mic its channel
  is done on the board, so the five mic cables are plain straight-through cables that can't be mixed
  up.

The motor connectors run along one edge, away from the mic signals, and both USB ports sit at the
board edge.

![PCB, 3D render](devpost-images/pcb_v3_3d_iso.png)

This render shows the two-layer carrier board, 115 × 66 mm, with four M3 mounting holes.
The ESP32 DevKit runs along the right side, and the BNO085 and PowerBoost mount in the outlined
areas in the middle and on the left. The eight JST motor connectors line the top edge, and the
five pin headers for the mics sit along the bottom. The small parts (the resistors, C2 and D2) are
surface-mount, while the ULN2803A and C1 are through-hole.

![PCB, top](devpost-images/pcb_v3_3d_top.png)

From above, you can see how the board is split by noise. The motor connectors and the ULN2803A
take the top half, the mic headers take the bottom, and the IMU sits between them. The clock
resistors R4–R8 are grouped at the top right, right beside the ESP32's clock pin, because series
termination only works at the driving end of the line. The DevKit's USB end stops about a
millimetre short of the board edge, so the cable can still reach the port once the board is in its
enclosure.

![PCB, copper layers](devpost-images/pcb_v3_layout.png)

The copper view shows the routing, with the top layer in red and the bottom in blue. A ground
pour fills both layers. The motor traces stay in the top half of the board and never run down
among the mic headers. The +5 V and ground traces are 0.6 mm wide, compared with 0.25 mm for
signals, because they carry the whole board's current, around 0.7 A at peak with two motors running.

We also routed a **single-sided** version for a desktop CNC in case we couldn't get the shipped one in time.

![Single-sided milled version](devpost-images/pcb_mill_3d_iso.png)

The backup board is laid out so it can be milled on a desktop CNC and soldered by hand. All the
copper is on the bottom layer, with no vias, and every part is through-hole so it can be soldered
from that side. The two connections that couldn't be routed on a single layer are bridged with
wire jumpers, W1 and W2. Traces are 0.4 to 1.2 mm wide and pads are enlarged to 1.9 mm, which
leaves enough copper for a milling bit to cut cleanly. The JST motor connectors are replaced with
plain pin headers, and the ULN2803A sits in a socket.
