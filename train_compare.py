"""Compare three drone-detection CNNs on one-second audio windows.

Models:
- standard: larger baseline CNN
- tiny: very small plain CNN
- depthwise_tiny: efficient depthwise-separable CNN for ESP32-S3 testing

Run:
    python train_compare.py

Configuration:
    config.json in the same folder.

Outputs:
    *.pt checkpoints
    *_confusion_matrix.png
    training_loss_curves.png
    model_metric_comparison.png
    roc_curves.png
    comparison.csv / comparison.json
    *_test_predictions.csv
"""
from __future__ import annotations

import copy
import csv
import json
import random
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from hf_public import resolve_public_dataset, public_splits

SR = 16000
SECONDS = 1
N_FFT = 512
HOP = 256
MELS = 32
FMIN = 50
FMAX = 8000
FORMATS = {'.m4a', '.wav', '.flac', '.mp3'}
SEED = 42
ROOT = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_recordings(drone_dir: Path, background_dir: Path):
    records = []
    for folder, label in [(drone_dir, 1), (background_dir, 0)]:
        if not folder.is_dir():
            raise FileNotFoundError(f'Folder not found: {folder}')
        paths = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in FORMATS)
        if not paths:
            raise ValueError(f'No supported audio files in {folder}')
        for p in paths:
            records.append({'path': str(p.resolve()), 'label': label})
    return records


def decode_audio(path: str | Path) -> np.ndarray:
    cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path),
           '-f', 'f32le', '-ac', '1', '-ar', str(SR), 'pipe:1']
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f'FFmpeg failed on {path}: {result.stderr.decode(errors="replace")}')
    samples = np.frombuffer(result.stdout, dtype='<f4').copy()
    if len(samples) == 0 or not np.isfinite(samples).all():
        raise ValueError(f'Empty or invalid audio: {path}')
    return samples


def split_recordings(records, temporal_gap_seconds=5.0):
    """Prefer independent-file splits; fall back to time-block splits when needed.

    All evaluation is WINDOW LEVEL, but original files/sections are used to define
    the train/val/test boundaries so nearby windows do not leak across splits.
    """
    output = {'train': [], 'val': [], 'test': []}
    temporal_fallbacks = []

    for label, name in [(0, 'non-drone'), (1, 'drone')]:
        items = [r for r in records if r['label'] == label]
        if len(items) >= 5:
            trainval, test = train_test_split(items, test_size=0.20, random_state=SEED)
            train, val = train_test_split(trainval, test_size=0.25, random_state=SEED)
            for split_name, subset in [('train', train), ('val', val), ('test', test)]:
                output[split_name].extend(subset)
        else:
            for rec in items:
                length = len(decode_audio(rec['path'])) / SR
                gap = float(temporal_gap_seconds)
                cut1, cut2 = 0.60 * length, 0.80 * length
                pieces = {
                    'train': (0.0, cut1 - gap / 2),
                    'val': (cut1 + gap / 2, cut2 - gap / 2),
                    'test': (cut2 + gap / 2, length),
                }
                if any(end - start < SECONDS * 2 for start, end in pieces.values()):
                    raise ValueError(
                        f'Only {len(items)} {name} recording(s), and {rec["path"]} '
                        f'is too short ({length:.1f}s) for safe time-block splitting. '
                        'Collect more independent recordings in this class.'
                    )
                for split_name, (start, end) in pieces.items():
                    output[split_name].append({
                        **rec,
                        'start_sec': round(start, 6),
                        'end_sec': round(end, 6),
                        'split_method': 'temporal_block',
                    })
                temporal_fallbacks.append(rec['path'])

    for subset in output.values():
        random.Random(SEED).shuffle(subset)
    if temporal_fallbacks:
        print('WARNING: temporal splitting used for classes with fewer than 5 files.')
        print('These test sections are NOT independent recordings; metrics are preliminary.')
        for p in temporal_fallbacks:
            print('  Temporally split:', p)
    return output, temporal_fallbacks


def features(window: np.ndarray) -> np.ndarray:
    stft = librosa.stft(window, n_fft=N_FFT, hop_length=HOP, center=False)
    power = np.abs(stft) ** 2
    mel = librosa.feature.melspectrogram(
        S=power, sr=SR, n_fft=N_FFT, n_mels=MELS,
        fmin=FMIN, fmax=FMAX, htk=False, norm='slaney'
    )
    db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    assert db.shape == (MELS, 61), db.shape
    return db.astype(np.float32)


def make_arrays(records):
    """Turn source recordings/sections into one-second windows."""
    xx, yy, ids, provenance = [], [], [], []
    for rid, rec in enumerate(records):
        audio = decode_audio(rec['path'])
        if 'start_sec' in rec:
            start_sample = round(rec['start_sec'] * SR)
            end_sample = round(rec['end_sec'] * SR)
            audio = audio[start_sample:end_sample]
        for start in range(0, len(audio), SR * SECONDS):
            chunk = audio[start:start + SR * SECONDS]
            if len(chunk) < SR * SECONDS:
                if len(chunk) < SR * SECONDS // 2:
                    continue
                chunk = np.pad(chunk, (0, SR * SECONDS - len(chunk)))
            xx.append(features(chunk))
            yy.append(rec['label'])
            ids.append(rid)
            provenance.append({
                'file': rec['path'],
                'start_sec': round(rec.get('start_sec', 0) + start / SR, 4),
                'end_sec': round(rec.get('start_sec', 0) + (start + len(chunk)) / SR, 4),
            })
    if not xx:
        raise ValueError('No usable windows. Check audio files.')
    return np.stack(xx), np.asarray(yy, dtype=np.int64), np.asarray(ids, dtype=np.int64), provenance


class SpectrogramDataset(Dataset):
    def __init__(self, x, y, ids, provenance, mean, std):
        self.x = torch.from_numpy(((x - mean) / std)[:, None].astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.ids = ids
        self.provenance = provenance

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class StandardCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).flatten()


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 12, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(12, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AvgPool2d((4, 8)), nn.Flatten(), nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).flatten()


class DepthwiseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseTinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(),
            DepthwiseBlock(8, 16, stride=2),
            DepthwiseBlock(16, 24, stride=2),
            DepthwiseBlock(24, 32, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, 1)

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1)).flatten()


def infer(model, ds, device, batch_size):
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in DataLoader(ds, batch_size=batch_size):
            logits = model(xb.to(device))
            out.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.asarray(out)


def metrics(y, score, threshold):
    pred = (score >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    m = dict(
        accuracy=accuracy_score(y, pred),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0),
        specificity=(cm[0, 0] / max(1, cm[0].sum())),
        false_positive_rate=(cm[0, 1] / max(1, cm[0].sum())),
        confusion_matrix=cm.tolist(),
        threshold=float(threshold),
    )
    if len(np.unique(y)) == 2:
        m['roc_auc'] = float(roc_auc_score(y, score))
        m['average_precision'] = float(average_precision_score(y, score))
    else:
        m['roc_auc'] = None
        m['average_precision'] = None
    return m


def choose_threshold(y, probs):
    candidates = np.linspace(0.05, 0.95, 91)
    return float(max(candidates, key=lambda t: (f1_score(y, probs >= t, zero_division=0), -abs(t - 0.5))))


def preprocessing_config():
    return {
        'sample_rate': SR,
        'window_seconds': SECONDS,
        'fft_size': N_FFT,
        'hop_length': HOP,
        'mel_bins': MELS,
        'fmin': FMIN,
        'fmax': FMAX,
        'stft_center': False,
        'mel_htk': False,
        'mel_norm': 'slaney',
        'feature': '10 * log10(max(mel_power, 1e-10))',
        'amplitude_normalization': 'global training mean and std',
    }


def plot_confusion_matrix(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    img = ax.imshow(cm, cmap='Blues')
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha='center', va='center')
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=['Non-drone', 'Drone'], yticklabels=['Non-drone', 'Drone'],
        xlabel='Predicted', ylabel='Actual', title=title,
    )
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_training_curves(histories, out_dir):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, hist in histories.items():
        epochs = np.arange(1, len(hist['train']) + 1)
        ax.plot(epochs, hist['train'], label=f'{name} train')
        ax.plot(epochs, hist['val'], linestyle='--', label=f'{name} val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Binary cross-entropy loss')
    ax.set_title('Training and validation loss')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / 'training_loss_curves.png', dpi=180)
    plt.close(fig)


def plot_model_comparison(outcomes, out_dir):
    models = list(outcomes.keys())
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'false_positive_rate']
    x = np.arange(len(metrics_to_plot))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.5, 5))
    offsets = np.linspace(-width, width, len(models))
    for offset, model in zip(offsets, models):
        values = [outcomes[model][m] for m in metrics_to_plot]
        ax.bar(x + offset, values, width=width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1', 'False+ rate'])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Window-level test metrics by model')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / 'model_metric_comparison.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    params = [outcomes[m]['parameters'] for m in models]
    ax.bar(models, params)
    ax.set_ylabel('Trainable parameters')
    ax.set_title('Model size comparison')
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / 'model_size_comparison.png', dpi=180)
    plt.close(fig)


def plot_roc_curves(roc_data, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    drew_any = False
    for name, data in roc_data.items():
        y_true, probs, auc = data
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, probs)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
        drew_any = True
    ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC curves on test windows')
    ax.grid(alpha=0.25)
    if drew_any:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'roc_curves.png', dpi=180)
    plt.close(fig)


def train_model(name, model, datasets, arrays, device, args, out_dir, mean, std):
    train_ds, val_ds, test_ds = datasets
    y_train = arrays['train'][1]
    class_counts = np.bincount(y_train, minlength=2)
    if np.any(class_counts == 0):
        raise ValueError('Training windows must contain both labels.')
    weights = np.asarray([1.0 / class_counts[y] for y in y_train], dtype=np.float64)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_loss, best_state, stale = float('inf'), None, 0
    history = {'train': [], 'val': []}

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)
        train_loss = total / len(train_ds)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_total += criterion(model(xb), yb).item() * len(yb)
        val_loss = val_total / len(val_ds)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        print(f'{name} epoch {epoch+1:02d}: train={train_loss:.4f}, val={val_loss:.4f}')
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError(f'No checkpoint captured for {name}')
    model.load_state_dict(best_state)

    val_y = arrays['val'][1]
    val_score = infer(model, val_ds, device, args.batch_size)
    threshold = choose_threshold(val_y, val_score)
    test_y = arrays['test'][1]
    test_score = infer(model, test_ds, device, args.batch_size)

    report = metrics(test_y, test_score, threshold)
    report['test_windows'] = int(len(test_y))
    report['parameters'] = sum(p.numel() for p in model.parameters())
    report['fp32_weight_bytes'] = sum(p.numel() * p.element_size() for p in model.parameters())
    report['int8_raw_weight_bytes_estimate'] = report['parameters']

    torch.save({
        'model': model.cpu().state_dict(),
        'architecture': name,
        'mean': mean,
        'std': std,
        'threshold': threshold,
        'preprocessing': preprocessing_config(),
    }, out_dir / f'{name}.pt')
    model.to(device)

    print(f'\n{name} WINDOW-LEVEL TEST metrics:\n{json.dumps(report, indent=2)}')
    print(classification_report(
        test_y, test_score >= threshold,
        labels=[0, 1], target_names=['Non-drone', 'Drone'], zero_division=0,
    ))
    return report, test_y, test_score, model, history


def main():
    config_path = ROOT / 'config.json'
    if not config_path.is_file():
        raise FileNotFoundError(f'Missing configuration file: {config_path}')
    with config_path.open(encoding='utf-8') as fh:
        config = json.load(fh)

    required = ('drone_dir', 'background_dir', 'output_dir', 'epochs', 'patience', 'batch_size', 'learning_rate', 'export_onnx')
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f'Missing config.json settings: {missing}')

    def resolve_path(value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()

    args = SimpleNamespace(
        drone=resolve_path(config['drone_dir']),
        background=resolve_path(config['background_dir']),
        out=resolve_path(config['output_dir']),
        epochs=int(config['epochs']),
        patience=int(config['patience']),
        batch_size=int(config['batch_size']),
        lr=float(config['learning_rate']),
        export_onnx=bool(config['export_onnx']),
        temporal_gap_seconds=float(config.get('temporal_gap_seconds', 5.0)),
    )
    if min(args.epochs, args.patience, args.batch_size) <= 0 or args.lr <= 0:
        raise ValueError('epochs, patience, batch_size and learning_rate must be positive')
    if args.temporal_gap_seconds < 0:
        raise ValueError('temporal_gap_seconds must be nonnegative')
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('FFmpeg is not installed or missing from PATH')

    print(f'Using configuration: {config_path}')
    args.out.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    split, temporal_fallbacks = split_recordings(find_recordings(args.drone, args.background), args.temporal_gap_seconds)
    (args.out / 'split.json').write_text(json.dumps(split, indent=2))
    if temporal_fallbacks:
        (args.out / 'EVALUATION_LIMITATIONS.txt').write_text(
            'TEMPORAL-SPLIT DATA USED. Some validation/test blocks come from recordings also used for training. '
            'Test scores are preliminary and may overestimate generalization. Collect independent recordings and rerun.\n'
        )
    print('Source recording sections:', {k: len(v) for k, v in split.items()}, '| device:', device)

    arrays = {part: make_arrays(records) for part, records in split.items()}
    print('One-second windows:', {k: len(v[1]) for k, v in arrays.items()})
    x_train = arrays['train'][0]
    mean, std = float(x_train.mean()), max(float(x_train.std()), 1e-6)
    (args.out / 'preprocessing.json').write_text(json.dumps({**preprocessing_config(), 'train_mean': mean, 'train_std': std}, indent=2))

    datasets = [SpectrogramDataset(*arrays[k], mean, std) for k in ('train', 'val', 'test')]

    outcomes = {}
    histories = {}
    roc_data = {}
    for name, cls in [('standard', StandardCNN), ('tiny', TinyCNN), ('depthwise_tiny', DepthwiseTinyCNN)]:
        set_seed(SEED)
        report, actual, probs, model, history = train_model(name, cls(), datasets, arrays, device, args, args.out, mean, std)
        outcomes[name] = report
        histories[name] = history
        roc_data[name] = (actual, probs, report.get('roc_auc') or float('nan'))

        with (args.out / f'{name}_test_predictions.csv').open('w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['file', 'start_sec', 'end_sec', 'actual', 'drone_probability', 'predicted'])
            for window, label, prob in zip(arrays['test'][3], actual, probs):
                writer.writerow([window['file'], window['start_sec'], window['end_sec'], int(label), float(prob), int(prob >= report['threshold'])])

        plot_confusion_matrix(np.asarray(report['confusion_matrix']), f'{name} test confusion matrix', args.out / f'{name}_confusion_matrix.png')

        if args.export_onnx:
            try:
                model.cpu().eval()
                torch.onnx.export(
                    model,
                    torch.zeros(1, 1, 32, 61),
                    str(args.out / f'{name}.onnx'),
                    input_names=['spectrogram'],
                    output_names=['logit'],
                    opset_version=13,
                )
                print(f'ONNX exported: {name}')
            except Exception as exc:
                print(f'ONNX export failed for {name}: {exc}')

    (args.out / 'comparison.json').write_text(json.dumps(outcomes, indent=2))
    with (args.out / 'comparison.csv').open('w', newline='') as fh:
        keys = ['model', 'test_windows', 'parameters', 'fp32_weight_bytes', 'int8_raw_weight_bytes_estimate', 'accuracy', 'precision', 'recall', 'f1', 'specificity', 'false_positive_rate', 'roc_auc', 'average_precision', 'threshold']
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for name, result in outcomes.items():
            writer.writerow({'model': name, **{k: result.get(k) for k in keys[1:]}})

    plot_training_curves(histories, args.out)
    plot_model_comparison(outcomes, args.out)
    plot_roc_curves(roc_data, args.out)

    print('\nSummary (TEST, one-second window level):')
    for name, r in outcomes.items():
        print(f"{name:15} accuracy={r['accuracy']:.3f}, recall={r['recall']:.3f}, F1={r['f1']:.3f}, FPR={r['false_positive_rate']:.3f}, parameters={r['parameters']}")
    for name in ('tiny', 'depthwise_tiny'):
        print(f'{name} minus standard F1:', round(outcomes[name]['f1'] - outcomes['standard']['f1'], 4))
    if temporal_fallbacks:
        print('IMPORTANT: Temporal split makes these preliminary, not independent-test scores.')
    print('Outputs:', args.out.resolve())

    # Additional experiment: train all original architectures on OWN + PUBLIC data.
    # IMPORTANT: the original experiment and original results are untouched.
    hf_cfg = config.get('huggingface', {})
    if hf_cfg.get('enabled', False):
        root = resolve_public_dataset(hf_cfg, ROOT)
        public = public_splits(root, hf_cfg.get('max_files_per_class_per_split'))
        combined_out = args.out / 'combined_with_huggingface'
        combined_out.mkdir(parents=True, exist_ok=True)
        public_arrays = {part: make_arrays(public[part]) for part in ('train', 'val', 'test')}
        combined_arrays = {}
        for part in ('train', 'val', 'test'):
            own_x, own_y, own_ids, own_prov = arrays[part]
            hf_x, hf_y, hf_ids, hf_prov = public_arrays[part]
            combined_arrays[part] = (
                np.concatenate([own_x, hf_x]),
                np.concatenate([own_y, hf_y]),
                np.concatenate([own_ids, hf_ids + len(own_ids)]),
                own_prov + hf_prov,
            )
        combined_mean = float(combined_arrays['train'][0].mean())
        combined_std = max(float(combined_arrays['train'][0].std()), 1e-6)
        combined_ds = [SpectrogramDataset(*combined_arrays[k], combined_mean, combined_std) for k in ('train', 'val', 'test')]
        # Evaluate each resulting model separately on both source domains.
        own_test_ds = SpectrogramDataset(*arrays['test'], combined_mean, combined_std)
        public_test_ds = SpectrogramDataset(*public_arrays['test'], combined_mean, combined_std)
        results = {}
        histories2 = {}
        rocs2 = {}
        for name, cls in [('standard', StandardCNN), ('tiny', TinyCNN), ('depthwise_tiny', DepthwiseTinyCNN)]:
            set_seed(SEED)
            report, actual, probs, model, history = train_model(
                name, cls(), combined_ds, combined_arrays, device, args,
                combined_out, combined_mean, combined_std,
            )
            histories2[name] = history
            rocs2[name] = (actual, probs, report.get('roc_auc') or float('nan'))
            source_results = {}
            for source_name, source_arrays, source_ds in [
                ('own_test', arrays['test'], own_test_ds),
                ('huggingface_test', public_arrays['test'], public_test_ds),
            ]:
                ys = source_arrays[1]
                ps = infer(model, source_ds, device, args.batch_size)
                source_report = metrics(ys, ps, report['threshold'])
                source_report['test_windows'] = len(ys)
                source_results[source_name] = source_report
                plot_confusion_matrix(np.asarray(source_report['confusion_matrix']),
                    f'{name}: {source_name} (combined training)',
                    combined_out / f'{name}_{source_name}_confusion_matrix.png')
                with (combined_out / f'{name}_{source_name}_predictions.csv').open('w', newline='') as fh:
                    writer = csv.writer(fh)
                    writer.writerow(['file', 'start_sec', 'end_sec', 'actual', 'drone_probability', 'predicted'])
                    for window, label, prob in zip(source_arrays[3], ys, ps):
                        writer.writerow([window['file'], window['start_sec'], window['end_sec'], int(label), float(prob), int(prob >= report['threshold'])])
            results[name] = {'combined_test': report, **source_results}
            plot_confusion_matrix(np.asarray(report['confusion_matrix']), f'{name}: combined test', combined_out / f'{name}_confusion_matrix.png')
        (combined_out / 'comparison_by_source.json').write_text(json.dumps(results, indent=2))
        with (combined_out / 'comparison_by_source.csv').open('w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['model', 'test_domain', 'test_windows', 'accuracy', 'precision', 'recall', 'f1', 'false_positive_rate', 'roc_auc', 'threshold'])
            for model_name, srcs in results.items():
                for domain, r in srcs.items():
                    writer.writerow([model_name, domain, r['test_windows'], r['accuracy'], r['precision'], r['recall'], r['f1'], r['false_positive_rate'], r['roc_auc'], r['threshold']])
        plot_training_curves(histories2, combined_out)
        plot_model_comparison({name: r['combined_test'] for name, r in results.items()}, combined_out)
        plot_roc_curves(rocs2, combined_out)
        (combined_out / 'preprocessing.json').write_text(json.dumps({**preprocessing_config(), 'train_mean': combined_mean, 'train_std': combined_std}, indent=2))
        (combined_out / 'split.json').write_text(json.dumps({'own': split, 'public': public}, indent=2))
        print('Combined training completed:', combined_out)
        print('CAUTION: public dataset comes from one session; interval-disjoint splits do not equal independent sessions.')



if __name__ == '__main__':
    main()
