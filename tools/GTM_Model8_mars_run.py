"""Run the frozen local baseline and publish its predeclared review checkpoints.

The holdout does not control training or checkpoint selection. Evaluate the
first completed pass and the final validation-selected checkpoint only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STORE = Path('D:/Research/REMOTE-SENSING/ersrr-ordinal-win')
RUN = STORE/'outputs/GTM_Model8_mars_baseline_20261030'
PLAN = ROOT/'tool-diagnostics/mars-baseline-20261030'
LAUNCHER = ROOT/'tools/GTM_research_on_D.py'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def verified_sources():
    paths = [RUN/'acquisition/asset_verification.json', RUN/'acquisition/ch4_extension/asset_verification.json']
    for path in paths:
        receipt = json.loads(path.read_text())
        if receipt['state'] != 'passed' or receipt['failed'] or receipt['verified_objects'] != receipt['objects']:
            raise ValueError(f'Data verification did not pass: {path}')
    return {str(path):sha(path) for path in paths}


def export(checkpoint, stage, completed):
    """Each export has its own process and RNG state, separate from training."""
    checkpoint_identity = sha(checkpoint)
    for role in ('validation', 'holdout'):
        output = RUN/'review'/stage/role
        done = output/'publication.json'
        if done.is_file():
            continue
        if output.exists():
            raise RuntimeError(f'An incomplete review needs inspection before retry: {output}')
        command = [sys.executable, '-u', str(LAUNCHER), 'review', '--role', role,
                   '--checkpoint', str(checkpoint), '--output', str(output), '--publish']
        if completed:
            command.append('--completed')
        with (RUN/f'review-{stage}-{role}.log').open('a', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Review failed; inspect review-{stage}-{role}.log')
    if sha(checkpoint) != checkpoint_identity:
        raise ValueError('Immutable checkpoint changed during export')
    return {'checkpoint':str(checkpoint), 'sha256':checkpoint_identity, 'stage':stage}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', required=True, type=int)
    parser.add_argument('--precision', choices=('fp32','bf16'), required=True)
    parser.add_argument('--num-workers', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=170)
    parser.add_argument('--draws', type=int, default=65536)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--validation-batch-size', type=int, default=8)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    source_receipts = verified_sources()
    state_path = RUN/'controller_state.json'
    if state_path.is_file() and json.loads(state_path.read_text()).get('state') == 'running':
        raise RuntimeError('A controller is already marked running; inspect its PID before resuming')
    settings = vars(args)
    state = {'state':'running', 'controller_pid':os.getpid(), 'settings':settings,
             'data_verification':source_receipts, 'exports':[],
             'holdout_policy':'First completed epoch and final validation-selected checkpoint only; no training choices from holdout results',
             'driver_sha256':sha(Path(__file__))}
    save(state_path, state)
    command = [sys.executable, '-u', str(LAUNCHER), 'baseline', '--mode', 'train',
               '--manifest', str(PLAN/'train_manifest.json')]
    for name, value in settings.items():
        command += ['--'+name.replace('_','-'), str(value)]
    try:
        with (RUN/'training.log').open('a', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state['training_pid'] = process.pid
            save(state_path, state)
            first_done = (RUN/'review/first_pass/holdout/publication.json').is_file()
            while process.poll() is None:
                first = RUN/'fit/epoch_000.pt'
                if first.is_file() and (RUN/'fit/epoch_000.json').is_file() and not first_done:
                    # A viewer failure must not destroy a useful training run.
                    try:
                        state['exports'].append(export(first, 'first_pass', False))
                        first_done = True
                        state.pop('review_error', None)
                    except Exception as error:
                        state['review_error'] = repr(error)
                        first_done = True
                    save(state_path, state)
                time.sleep(10)
            if process.returncode:
                raise RuntimeError(f'Training exited with {process.returncode}; inspect training.log')
        final_checkpoint = RUN/'fit/best_epoch.pt'
        snapshot = RUN/'fit/final_selected_snapshot.pt'
        if snapshot.exists():
            if sha(snapshot) != sha(final_checkpoint):
                raise RuntimeError('Final snapshot identity differs on resume')
        else:
            temporary = snapshot.with_suffix('.tmp')
            shutil.copyfile(final_checkpoint, temporary)
            if sha(temporary) != sha(final_checkpoint):
                raise ValueError('Incomplete final checkpoint copy')
            os.replace(temporary, snapshot)
        state['exports'].append(export(snapshot, 'final', True))
        state['state'] = 'completed'
        save(state_path, state)
    except BaseException as error:
        state.update(state='failed', error=repr(error))
        save(state_path, state)
        raise


if __name__ == '__main__':
    main()
