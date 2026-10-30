"""Train-only, label-blind SWIR anomaly diagnostic; no concentration retrieval."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spectral(current, reference, valid):
    support = valid & np.isfinite(current).all(-1) & np.isfinite(reference).all(-1)
    support &= (current[..., 4:6] > 0).all(-1) & (reference[..., 4:6] > 0).all(-1)
    residuals, coefficients = [], []
    for x in (current, reference):
        b11, b12 = x[..., 4].astype(np.float64), x[..., 5].astype(np.float64)
        c = float(np.dot(b12[support], b11[support]) / np.dot(b12[support], b12[support]))
        fractional = np.zeros(valid.shape, np.float64)
        fractional[support] = c * b12[support] / b11[support] - 1
        residuals.append(fractional)
        coefficients.append(c)
    proxy = residuals[1] - residuals[0]
    center = float(np.median(proxy[support]))
    scale = max(1e-5, float(1.4826 * np.median(np.abs(proxy[support] - center))))
    z = np.zeros(valid.shape, np.float32)
    z[support] = (proxy[support] - center) / scale
    return z, support, dict(coefficients=coefficients, center=center, scale=scale)


def counts(p, y, v):
    p, y = p[v], y[v] == 1
    return dict(tp=int((p & y).sum()), fp=int((p & ~y).sum()),
                fn=int((~p & y).sum()), tn=int((~p & ~y).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.data/'manifest.json'
    manifest_hash = sha(path)
    scenes = [s for s in json.loads(path.read_text())['scenes'] if s['split'] in ('train', 'dev')]
    totals = {split:dict(tp=0,fp=0,fn=0,tn=0) for split in ('train','dev')}
    records, thumbnails = [], []
    for scene in scenes:
        source = (args.data/scene['array_file']).resolve()
        if not source.is_relative_to(args.data.resolve()) or sha(source) != scene['array_sha256']:
            raise ValueError('Unsafe or changed source')
        with np.load(source, allow_pickle=False) as z:
            a = {k:z[k] for k in z.files}
        score, valid, fit = spectral(a['target_image'], a['reference_image'], a['valid'])
        zero, identity_support, _ = spectral(a['target_image'], a['target_image'], a['valid'])
        assert np.all(zero[identity_support] == 0)
        assert np.isfinite(score[valid]).all()
        native = np.repeat(np.repeat(score,2,0),2,1)
        support = np.repeat(np.repeat(valid,2,0),2,1)
        c = counts(native>=3, a['label_native'], support)
        for k in c:
            totals[scene['split']][k] += c[k]
        positive = (a['target']==1)&valid
        record = dict(id=scene['id'], split=scene['split'], label=scene['scene_kind'],
                      **c, **fit, positive_z_median=float(np.median(score[positive])) if positive.any() else None,
                      positive_above3=float(np.mean(score[positive]>=3)) if positive.any() else None,
                      any_activation=bool(c['fp']+c['tp']))
        np.savez_compressed(args.output/(scene['id']+'.npz'), score20=score, valid20=valid)
        records.append(record)
        if scene['scene_kind']=='PLUME' or len([x for x in thumbnails if x[0].startswith('NO_PLUME')])<4:
            image = a['target_image'][...,[2,1,0]]
            lo, hi = np.percentile(image[valid],(2,98),axis=0)
            rgb = (np.clip((image-lo)/np.maximum(hi-lo,1),0,1)*255).astype(np.uint8)
            rgb = np.repeat(np.repeat(rgb,2,0),2,1)
            label = rgb.copy(); label[(a['label_native']==1)&support]=(46,224,219)
            heat = (colormaps['coolwarm']((np.clip(native,-5,5)+5)/10)[...,:3]*255).astype(np.uint8)
            overlay=rgb.copy(); overlay[(native>=3)&support]=(255,151,56)
            for x in (rgb,label,heat,overlay): x[~support]=(58,69,78)
            thumbnails.append((scene['scene_kind']+' | '+scene['id'][5:13]+' | '+scene['split'],[rgb,label,heat,overlay]))
        if sha(source)!=scene['array_sha256']:
            raise ValueError('Diagnostic changed data')
    assert sha(path)==manifest_hash
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',15)
    sheet=Image.new('RGB',(960,len(thumbnails)*245),(13,25,31)); draw=ImageDraw.Draw(sheet)
    for i,(title,images) in enumerate(thumbnails):
        y=i*245; draw.text((8,y+2),title,font=font,fill=(235,242,246))
        for j,(x,label) in enumerate(zip(images,['Current RGB','Reviewed label','SWIR z [-5,+5]','Fixed z>=3'])):
            sheet.paste(Image.fromarray(x),(j*240+8,y+25)); draw.text((j*240+8,y+224),label,font=font,fill=(172,194,204))
    sheet.save(args.output/'training_contact_sheet.png')
    sheet.crop((0,0,960,min(8,len(thumbnails))*245)).save(args.output/'preview.png')
    result=dict(schema='GTM_Model7_spectral_diagnostic_v1',manifest_sha256=manifest_hash,
                source_sha256=sha(__file__), threshold=3, score='SWIR fractional anomaly robust z; not concentration',
                test_rasters_opened=False, checked_scenes=len(records), identity_checks=len(records),
                pooled=totals, scenes=records)
    (args.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='scenes'},indent=2))


if __name__=='__main__': main()
