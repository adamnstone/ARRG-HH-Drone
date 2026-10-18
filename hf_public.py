"""Download and safely partition the public ITU ARIS drone audio corpus.

Official source: https://huggingface.co/datasets/imm61/itu-aris-lab-acoustic-drone-dataset
Uses publisher-supplied interval-disjoint assignment, not random clip splitting.
"""
import json
from pathlib import Path

HF_REPO = 'imm61/itu-aris-lab-acoustic-drone-dataset'


def resolve_public_dataset(hf_config, project_root):
    if not hf_config.get('enabled', False):
        return None
    raw = Path(hf_config.get('local_dir', 'data/huggingface_itu_aris')).expanduser()
    root = raw if raw.is_absolute() else (project_root / raw).resolve()
    marker = root / 'splits' / 'split_interval_disjoint.json'
    if marker.exists() and (root / 'audio').exists():
        print('Using cached Hugging Face dataset:', root)
        return root
    if not hf_config.get('download_if_missing', True):
        raise FileNotFoundError(f'Public dataset missing at {root}. Set download_if_missing=true or download it first.')
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError('Install huggingface_hub: python -m pip install huggingface_hub') from exc
    print(f'Downloading {HF_REPO} (~244 MB audio) to {root}...')
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO, repo_type='dataset', local_dir=str(root),
        allow_patterns=['audio/**', 'splits/**', 'groups.json', 'metadata.json', 'README.md'],
    )
    if not marker.exists():
        raise FileNotFoundError(f'Official split missing from downloaded dataset: {marker}')
    return root


def public_splits(root, max_files_per_class_per_split=None):
    """Return train/val/test record lists with publisher-approved split membership.

    Optional caps are applied *within* each official split. No examples move
    across splits. Per-class filenames are sorted for reproducibility.
    """
    split_file = root / 'splits' / 'split_interval_disjoint.json'
    raw = json.loads(split_file.read_text(encoding='utf-8'))
    assignment = raw.get('assignment')
    if not isinstance(assignment, dict):
        raise ValueError(f'Expected assignment mapping in {split_file}')
    result = {'train': [], 'val': [], 'test': []}
    missing = []
    for key, split in sorted(assignment.items()):
        split = {'validation': 'val', 'valid': 'val'}.get(split, split)
        if split not in result:
            continue
        path = (root / key).resolve()
        if not path.is_file():
            missing.append(str(path))
            continue
        parts = {p.lower() for p in path.parts}
        if 'drone' in parts:
            label = 1
        elif 'background' in parts:
            label = 0
        else:
            raise ValueError(f'Cannot infer public label from path: {path}')
        result[split].append({'path': str(path), 'label': label, 'source': 'huggingface'})
    if missing:
        raise FileNotFoundError(f'{len(missing)} dataset files listed in official split were not found; e.g. {missing[0]}')
    if max_files_per_class_per_split is not None:
        cap = int(max_files_per_class_per_split)
        if cap <= 0:
            raise ValueError('max_files_per_class_per_split must be positive or null')
        for s in result:
            result[s] = [item for label in (0, 1) for item in [r for r in result[s] if r['label'] == label][:cap]]
    for s, records in result.items():
        labels = {r['label'] for r in records}
        if labels != {0, 1}:
            raise ValueError(f'Official public split {s} is missing a class. Labels present: {labels}')
    print('Official Hugging Face split counts:', {s: {label: sum(r['label'] == label for r in rs) for label in (0, 1)} for s, rs in result.items()})
    return result
