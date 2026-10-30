"""Render metadata-selected TRAIN/DEV inputs before final evaluation access."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from GTM_Model7_review_repaired import rgb_pair
from GTM_Model7_review_temporal import sha, save


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    cm = manifest.get('native_repeat_factor') == 1
    bins = collections.defaultdict(list)
    for scene in manifest['scenes']:
        if scene['split'] == 'test':
            continue
        bins[(scene['split'], scene.get('state', scene['group_id']), scene['scene_kind'])].append(scene)
    chosen = [sorted(rows, key=lambda s: sha_key(s['id']))[0] for _, rows in sorted(bins.items())]
    args.output.mkdir(parents=True, exist_ok=False)
    font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 15)
    selected = []
    for page in range((len(chosen) + 7) // 8):
        image = Image.new('RGB', (1000, 8 * 226), '#101e25')
        draw = ImageDraw.Draw(image)
        for index, s in enumerate(chosen[page*8:(page+1)*8]):
            file = args.manifest.parent / s['array_file']
            if sha(file) != s['array_sha256']:
                raise ValueError('Data changed')
            with np.load(file, allow_pickle=False) as z:
                d = {k: z[k] for k in z.files}
            current, reference = rgb_pair(d, 1 if cm else 2)
            label = current.copy()
            hit = (d['label_native'] == 1) & d['valid_native']
            label[hit] = (label[hit] * .35 + np.array((46, 224, 219)) * .65).astype('uint8')
            y = index * 226
            title = f"{s['split']} | {s.get('state',s['group_id'])} | {s['scene_kind']} | {s['id']}"
            draw.text((10, y + 2), title, font=font, fill='#e4eef1')
            for column, arr in enumerate((current, reference, label)):
                image.paste(Image.fromarray(arr).resize((192, 192), Image.Resampling.NEAREST), (10 + column * 330, y + 26))
            selected.append(s['id'])
        image.save(args.output / f'inputs_{page:02d}.png')
    save(args.output / 'receipt.json', dict(manifest_sha256=sha(args.manifest), selected=selected,
         columns=['Current real Sentinel', 'Reference real Sentinel', 'Released annotation'],
         rule='First seeded-hash TRAIN/DEV scene per split/state-or-group/class, no test access', test_access=False))
    print(json.dumps({'sampled': len(chosen), 'pages': (len(chosen)+7)//8, 'output': str(args.output)}))


def sha_key(s):
    import hashlib
    return hashlib.sha256(('20261024_input_preview:' + s).encode()).hexdigest()


if __name__ == '__main__':
    main()
