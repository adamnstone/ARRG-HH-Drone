"""Audio file loading via ffmpeg (handles wav/m4a/mp3/anything), resampled to the pipeline rate."""
import subprocess, numpy as np
from . import config as C


def load(path, fs=C.FS, mono=True, start=None, duration=None):
    """Decode any audio file to float32 at fs. Returns (n,) if mono else (channels, n)."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(path)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", "-ar", str(fs)]
    if mono:
        cmd += ["-ac", "1"]
    cmd += ["-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32)
    if mono:
        return x
    # probe channel count
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()
    ch = int(p or 1)
    return x.reshape(-1, ch).T


def save_wav(path, y, fs=C.FS):
    import soundfile as sf
    y = np.asarray(y, dtype=np.float32)
    sf.write(str(path), y.T if y.ndim == 2 else y, fs, subtype="PCM_16")


def rms_normalise(x, target=0.05):
    r = np.sqrt(np.mean(x ** 2)) + 1e-9
    return (x * target / r).astype(np.float32)
