---
license: cc-by-4.0
task_categories:
  - audio-classification
language:
  - en
tags:
  - drone
  - uav
  - acoustic-detection
  - audio
  - surveillance
  - domain-shift
pretty_name: ITU ARIS Lab. Acoustic Drone Dataset
size_categories:
  - 1K<n<10K
---

# ITU ARIS Lab. Acoustic Drone Dataset

Two-class acoustic dataset for drone detection, recorded outdoors with multiple
drone models at ranges from a few metres to roughly two kilometres, annotated by
hand from a single continuous 3.12 h session.

| | |
|---|---|
| Classes | `drone` (4 491), `background` (1 000) |
| Segment length | 1.0 s |
| Sampling rate | 22 050 Hz, mono, 16-bit PCM |
| Licence | CC-BY-4.0 |

## Read this before you split the data

**Segments overlap by 50 %.** `drone_0004.wav` shares half a second of audio with
both `drone_0003.wav` and `drone_0005.wav`. A random split therefore evaluates
the model on audio it has already seen, and inflates the result.

`groups.json` maps every segment to the source interval it was cut from — 35
annotated drone intervals plus contiguous background blocks. **Split at group
boundaries.** Two ready-made protocols are in `splits/`.

## Files

```
audio/drone/          4 491 one-second segments
audio/background/     1 000 one-second segments
groups.json           segment -> source interval, with interval times
splits/split_interval_disjoint.json
splits/split_temporal.json
metadata.json
build_acolab_groups.py
```

## Quick start

```python
from huggingface_hub import snapshot_download
import json, soundfile as sf
from pathlib import Path

root = Path(snapshot_download("imm61/itu-aris-lab-acoustic-drone-dataset", repo_type="dataset"))
split = json.loads((root / "splits/split_interval_disjoint.json").read_text())

train = [p for p, s in split["assignment"].items() if s == "train"]
audio, sr = sf.read(root / train[0])
```

## Baselines

Four CNN architectures on 128-band log-Mel spectrograms, mean ± sd over 7 seeds,
using `split_interval_disjoint.json`:

| Training data | Balanced accuracy |
|---|---|
| Public drone datasets only | 61.4 ± 5.8 % |
| Public datasets + this set | **99.2 ± 0.8 %** |

Models trained on the public acoustic drone corpora do not transfer to these
recording conditions. That gap is why the set exists.


## Citation


İ. M. Muhacıroğlu and T. Akgül, "Deep Learning Based Acoustic Drone Detection," 2026 34th Signal Processing and Communications Applications Conference (SIU), İstanbul, Turkiye, 2026, pp. 1-4, 
doi: 10.1109/SIU71813.2026.11636719.


The canonical record is on Zenodo: https://zenodo.org/records/22682339

That link is version 1.0.0. The concept DOI `10.5281/zenodo.22682338` always
resolves to the latest version.

```bibtex
@dataset{muhaciroglu_itu_aris_lab_drone,
  author    = {Muhac{\i}ro{\u{g}}lu, {\.I}hsan Mert and Akg{\"u}l, Tayfun},
  title     = {{ITU ARIS Lab. Acoustic Drone Dataset}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.22682339}
}
```

Contact: İhsan Mert Muhacıroğlu — muhaciroglu19@itu.edu.tr
Electronics and Communication Engineering, Istanbul Technical University
