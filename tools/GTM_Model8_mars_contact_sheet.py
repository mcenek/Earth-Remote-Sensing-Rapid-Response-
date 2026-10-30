"""Render all saved review scenes in order, including misses and clear scenes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def render(review: Path, method: str):
    records = json.loads((review/'scene_records.json').read_text())['scenes']
    root = review/method
    destination = review/'contact_sheets'/method
    destination.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 17)
    small = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 15)
    pages = []
    for start in range(0, len(records), 5):
        subset = records[start:start+5]
        page = Image.new('RGB', (960, 65+len(subset)*350), '#101c25')
        pen = ImageDraw.Draw(page)
        pen.text((15, 10), f'{method} | scenes {start+1}-{start+len(subset)} of {len(records)} | fixed cutoff 0.50', font=font, fill='#eff5fa')
        for column, label in enumerate(('Current image', 'Cyan reference + orange prediction', 'Green TP | pink FP | purple misses')):
            pen.text((15+column*315, 40), label, font=small, fill='#a5c6d1')
        for number, record in enumerate(subset):
            row, score = record['source_row'], record['models'][method]
            top = 65+number*350
            kind = 'PLUME' if row['isplume'].lower() == 'true' else 'CLEAR'
            measure = f"F1 {score['f1']:.1%}, precision {score['precision']:.1%}, recall {score['recall']:.1%}" if kind == 'PLUME' else f"{score['fp']} false-positive pixels"
            pen.text((15, top+4), f"{start+number+1}. {kind} | {row['country']} | {row['tile_date'][:10]} | {record['id'][:8]}", font=font, fill='#eff5fa')
            pen.text((15, top+28), measure, font=small, fill='#a5c6d1')
            folder = root/record['id']
            current = Image.open(folder/'satellite.png').convert('RGBA')
            combined = Image.alpha_composite(Image.alpha_composite(current, Image.open(folder/'truth.png').convert('RGBA')), Image.open(folder/'prediction.png').convert('RGBA'))
            errors = Image.alpha_composite(current, Image.open(folder/'errors.png').convert('RGBA'))
            for column, image in enumerate((current, combined, errors)):
                page.paste(image.resize((285, 285), Image.Resampling.NEAREST).convert('RGB'), (15+column*315, top+54))
        path = destination/f'page_{start//5+1:02d}.png'
        page.save(path)
        pages.append(str(path))
    (destination/'index.json').write_text(json.dumps({'scenes':len(records), 'pages':pages}, indent=2)+'\n')
    print(json.dumps({'method':method, 'scenes':len(records), 'pages':pages}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('review', type=Path)
    parser.add_argument('--method', choices=('baseline','released_reference'), default='baseline')
    args = parser.parse_args()
    render(args.review, args.method)
