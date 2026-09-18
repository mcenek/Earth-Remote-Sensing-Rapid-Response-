import io
from pathlib import Path
import sys

import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from acquire_cafo_cog_pilot import read_window, georeference


def test_native_crop_crosses_compression_tiles_without_resampling():
    source = np.arange(64*64, dtype=np.uint16).reshape(64,64)
    stream = io.BytesIO()
    tifffile.imwrite(stream, source, tile=(16,16), compression='deflate')
    stream.seek(0)
    with tifffile.TiffFile(stream) as tif:
        result = read_window(tif.pages[0], stream, 13, 14, 23, 29)
        np.testing.assert_array_equal(result, source[14:43,13:36])
        with pytest.raises(ValueError, match='outside'):
            read_window(tif.pages[0], stream, 63, 0, 2, 2)


def test_explicit_utm_pixel_area_georeferencing():
    stream = io.BytesIO()
    keys = (1,1,0,2,1025,0,1,1,3072,0,1,32615)
    tifffile.imwrite(stream,np.zeros((16,16),dtype='uint16'),tile=(16,16),
                     extratags=[(33550,'d',3,(20.,20.,0.),False),
                                (33922,'d',6,(0.,0.,0.,500000.,4600000.,0.),False),
                                (34735,'H',len(keys),keys,False)])
    stream.seek(0)
    with tifffile.TiffFile(stream) as tif:
        epsg, affine = georeference(tif.pages[0])
        assert epsg==32615
        assert affine*(2,3)==(500040.,4599940.)
