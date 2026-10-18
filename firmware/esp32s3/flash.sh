#!/bin/bash
# Build the headband sketch with arduino-cli and flash it over the S3's native USB port.
#
#   firmware/esp32s3/flash.sh                  compile + upload (port auto-detected: first /dev/cu.usbmodem*)
#   firmware/esp32s3/flash.sh --build          compile only
#   firmware/esp32s3/flash.sh --port /dev/cu.usbmodem21101
#
# Needs: brew install arduino-cli; arduino-cli core install esp32:esp32 --additional-urls
# https://espressif.github.io/arduino-esp32/package_esp32_index.json (Arduino-ESP32 3.x). arduino-cli insists that
# the sketch folder is named like the .ino, so this stages a "headband/" folder of symlinks (sketch, pins.h,
# model_data.h, wifi_config.h if present, src/ = the shared DSP core) under $TMPDIR and builds there.
# Board options: native USB CDC on boot (telemetry + raw audio over the USB JTAG/serial port), hardware CDC/JTAG,
# 3 MB app partition (no OTA), octal PSRAM (the bulk buffers live there; the module is an N?R8), 240 MHz.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FQBN="esp32:esp32:esp32s3:CDCOnBoot=cdc,USBMode=hwcdc,PartitionScheme=huge_app,FlashSize=4M,PSRAM=opi,CPUFreq=240,UploadSpeed=921600"
ROOT="${TMPDIR:-/tmp}/headband-arduino"
STAGE="$ROOT/headband"
BUILD="$ROOT/build"

BUILD_ONLY=0; PORT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --build) BUILD_ONLY=1 ;;
        --port) PORT="$2"; shift ;;
        -h|--help) sed -n 2,13p "$0"; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
    shift
done

rm -rf "$STAGE"; mkdir -p "$STAGE/src" "$BUILD"
for f in headband.ino pins.h model_data.h; do ln -s "$HERE/$f" "$STAGE/$f"; done
[ -f "$HERE/wifi_config.h" ] && ln -s "$HERE/wifi_config.h" "$STAGE/wifi_config.h"
for f in dsp_core.cpp dsp_core.h dsp_config.h; do ln -s "$HERE/../core/$f" "$STAGE/src/$f"; done

echo "# compile: $FQBN"
# -O2 instead of the core's -Os: the DSP (six 2048-point FFTs plus the detector per 21.3 ms frame) needs the speed
arduino-cli compile --fqbn "$FQBN" --build-path "$BUILD" --warnings default \
    --build-property compiler.optimization_flags=-O2 "$STAGE"
[ "$BUILD_ONLY" = 1 ] && exit 0

if [ -z "$PORT" ]; then
    PORT="$(ls /dev/cu.usbmodem* 2>/dev/null | head -1 || true)"
    [ -n "$PORT" ] || { echo "no /dev/cu.usbmodem* port; plug the board into its native USB port or pass --port" >&2; exit 1; }
fi
echo "# upload: $PORT"
arduino-cli upload --fqbn "$FQBN" -p "$PORT" --input-dir "$BUILD" "$STAGE"
