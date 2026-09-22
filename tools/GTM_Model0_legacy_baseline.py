"""Pinned previous-team web baseline: inspect artifacts or explicitly predict on CPU.

Check mode never imports Keras, authenticates, downloads or executes a model.
Predict mode needs a compact prepared scene, >=6 GiB free RAM and strict weights.
"""
from __future__ import annotations
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'research/GTM_Baselines/GTM_Model0_capstone_contract.json'


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def available_ram_gib():
    if os.name == 'nt':
        class Memory(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (k, ctypes.c_ulonglong) for k in ['total', 'available', 'page_total', 'page_available',
                                                'virtual_total', 'virtual_available', 'extended']]
        m = Memory(); m.length = ctypes.sizeof(m)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return m.available / 2**30
    else:
        p = Path('/proc/meminfo')
        if p.exists():
            for line in p.read_text().splitlines():
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) / 1024**2
    return None


def check():
    contract = json.loads(CONTRACT.read_text(encoding='utf-8'))
    frozen = CONTRACT.parent / contract['frozen_module']
    weights = (ROOT / contract['checkpoint_path']).resolve()
    if sha(frozen) != contract['frozen_sha256']:
        raise ValueError('Frozen model source changed')
    if sha(weights) != contract['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from pinned historical artifact')
    return dict(contract=contract, weights=str(weights), source=str(frozen),
                verified_hashes=True, available_ram_gib=available_ram_gib(),
                tensorflow_installed=importlib.util.find_spec('tensorflow') is not None,
                runtime_load_verified=False, inference_performed=False)


def normalize_and_pad(bands, valid):
    import numpy as np
    bands = np.asarray(bands, dtype=np.float32)
    if bands.ndim != 3 or bands.shape[0] != 5 or max(bands.shape[1:]) > 256:
        raise ValueError('Expected five ordered raw Sentinel bands, up to 256x256; no resizing')
    if valid.shape != bands.shape[1:] or not np.isin(valid, [0, 1]).all() or not valid.any():
        raise ValueError('Explicit nonempty Sentinel validity required')
    if not np.isfinite(bands[:, valid.astype(bool)]).all():
        raise ValueError('Valid Sentinel pixels contain nonfinite input')
    tile = np.zeros((256, 256, 5), dtype=np.float32)
    h, w = bands.shape[1:]
    tile[:h, :w] = np.moveaxis(np.where(valid[None], bands, 0), 0, -1)
    scale = float(np.max(tile))
    if scale <= 0:
        raise ValueError('Historical per-tile normalization requires a positive maximum')
    return tile / (scale + 1e-6)


def predict(scene_path, output, metadata_output):
    checked = check()
    ram = checked['available_ram_gib']
    if ram is None or ram < 6:
        raise ValueError(f'CPU transformer load deferred: at least 6 GiB free RAM required; available={ram}')
    if not checked['tensorflow_installed']:
        raise ValueError('TensorFlow is absent from this runtime. Select a compatible CPU runtime before prediction; no automatic install or backend substitution.')
    output = Path(output); metadata_output = Path(metadata_output)
    if output.exists() or metadata_output.exists():
        raise ValueError('Refusing to overwrite prediction or provenance')
    # Set before any Keras/TF import. Never run this baseline on the GPU implicitly.
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'; os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    import numpy as np
    from GTM_Model0_compare_plumes import read_archive, write_json
    scene_path = Path(scene_path).resolve()
    scene = json.loads(scene_path.read_text(encoding='utf-8'))
    if scene['input_bands'] != ['B2', 'B3', 'B4', 'B11', 'B12']:
        raise ValueError('Raw input band contract mismatch')
    data_path = scene_path.parent / scene['input_archive']
    if sha(data_path) != scene['input_archive_sha256']:
        raise ValueError('Prepared input archive changed')
    data = read_archive(data_path); valid = data['valid'].astype(bool)
    if list(valid.shape) != scene['grid']['shape']:
        raise ValueError('Scene grid disagrees with native input')
    tile = normalize_and_pad(data['bands'], data['valid'])
    spec = importlib.util.spec_from_file_location('GTM_capstone_frozen', checked['source'])
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    model = module.create_vit_encoder_decoder()
    model.load_weights(checked['weights'])  # strict: no skip_mismatch, no random fallback
    result = model.predict(tile[None], verbose=0)
    h, w = valid.shape
    probability = np.asarray(result['mask_output'][0, :h, :w, 0], dtype=np.float32)
    regression = np.asarray(result['regression_output'][0, :h, :w, 0], dtype=np.float32)
    if probability.shape != valid.shape or not np.isfinite(probability).all() or not np.isfinite(regression).all():
        raise ValueError('Invalid legacy output; nothing exported')
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError('Legacy mask output is not a probability')
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, probability=probability, valid=valid,
                        regression_normalized=regression, gated_regression=regression * probability)
    write_json(metadata_output, dict(scene_id=scene['id'], source_sha256=scene['source_sha256'],
               grid=scene['grid'], score_kind='plume_probability', model_id=checked['contract']['id'],
               prediction_sha256=sha(output), provenance=dict(checkpoint_sha256=checked['contract']['checkpoint_sha256'],
               source_sha256=checked['contract']['frozen_sha256'], source_commit=checked['contract']['commit'],
               normalization=checked['contract']['normalization'], input_archive_sha256=scene['input_archive_sha256'],
               padding='zero bottom/right to256, crop output back; missing padding never evaluated',
               training_membership='unknown; diagnostic only', backend='tensorflow CPU',
               runtime_load_verified=True)))
    return dict(prediction=str(output), metadata=str(metadata_output))


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('check')
    run = sub.add_parser('predict'); run.add_argument('--scene', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True); run.add_argument('--metadata-output', type=Path, required=True)
    args = p.parse_args()
    try:
        if args.command == 'predict' and args.output.suffix != '.npz':
            raise ValueError('Prediction output must have .npz suffix')
        result = check() if args.command == 'check' else predict(args.scene, args.output, args.metadata_output)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError) as error:
        p.exit(2, str(error) + '\n')


if __name__ == '__main__':
    main()
