"""Check exporter IO using two FIT controls and released weights, without publishing."""
import csv
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools'))
import GTM_Model8_mars_review as review


def main():
    store = ROOT
    data = store/'EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MARS-S2L'
    run = store/'outputs/GTM_Model8_mars_baseline_20261030'
    controls = json.loads((run/'fit/preflight.json').read_text())['real_fit_label_controls']
    wanted = {controls['positive_id'], controls['negative_id']}
    with (data/'validated_images_all.csv').open(newline='', encoding='utf-8-sig') as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        rows = {row['id_loc_image']: row for row in reader if row['id_loc_image'] in wanted}
    output = run/'export_probe'
    output.mkdir(exist_ok=True)
    _, read_images, _, Dataset, _, _ = review.training.source_modules()
    with tempfile.TemporaryDirectory(dir=store/'tmp') as temp:
        path = Path(temp)/'fit_controls.csv'
        with path.open('w', newline='', encoding='utf-8') as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows.values())
        frame = read_images(str(path), path_prepend_data=str(data))
    review.torch.set_num_threads(2)
    dataset = Dataset(mode='test', image_dataframe=frame, do_simulation=False,
        multipass=True, wind=True, cloud_mask=True, cat_mbmp=True, norm_wind=True,
        bands_l8=True, cache=False, rotate_data_augmentation=False)
    spec = review.RELEASE_SPECS['mars-s2l']
    checkpoint = ROOT/spec['directory']/'best_epoch'
    assert review.training.sha256(checkpoint) == spec['checkpoint_sha256']
    model = review.load_released_model(checkpoint, review.torch.device('cpu'), 16)
    receipt = {'scope':'FIT controls only; released weights; exporter test, not new baseline results', 'scenes':[]}
    for index, (_, entry) in enumerate(dataset.image_dataframe.iterrows()):
        item, row = dataset[index], rows[entry.id_loc_image]
        with review.rasterio.open(data/row['s2path']) as image, review.rasterio.open(data/row['cloudmaskpath']) as cloud_file:
            raw, cloud = image.read(), cloud_file.read(1)
            affine, crs, resolution = image.transform, image.crs, image.res
        x, target = item['y_context_ls0_0'].numpy(), item['y_target'].numpy()
        rebuilt = review.independent_input(raw, cloud, row)
        error = float(review.np.max(review.np.abs(x-rebuilt)))
        assert review.np.allclose(x, rebuilt, atol=2e-6, rtol=0), error
        valid = review.np.isfinite(raw).all(0) & (cloud == 0) & (raw != 0).all(0)
        with review.torch.no_grad():
            probability = review.torch.sigmoid(model(review.torch.from_numpy(x[None])))[0].numpy()
        folder = output/entry.id_loc_image
        folder.mkdir(exist_ok=True)
        bounds = review.map_files(folder, raw, probability, target, valid, affine, crs)
        grid = json.loads((folder/'grid.json').read_text())
        assert grid['width'] == 200 and grid['height'] == 200 and len(grid['probability']) == 40000
        receipt['scenes'].append(dict(id=entry.id_loc_image, input_max_abs_error=error,
            resolution=list(resolution), bounds=bounds, label_pixels=int(target.sum()),
            marked_pixels=int(((probability >= .5) & valid).sum())))
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
