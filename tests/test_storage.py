"""Storage read from the files and the storage-matched re-degradation D(x) (addendum v7)."""
import io

import numpy as np
import pytest
from PIL import Image

from redeg import heat, quantize8, read_store, redegrade
from redeg.storage import jpeg_bytes, jpeg_file_store, jpeg_roundtrip, storage_is_exact, store_label


def test_notebook_self_test_recovers_quality_85_with_420():
    """The assertion of Cell 5.11: a JPEG 85 file written by Pillow reads back as (85, 4:2:0)."""
    assert jpeg_file_store(jpeg_bytes(np.full((32, 32, 3), 120, np.uint8), 85))[:2] == (85, 2)


@pytest.mark.parametrize("q", [30, 75, 85, 95])
@pytest.mark.parametrize("sub", [0, 1, 2])
def test_quality_and_subsampling_are_read_exactly(source_tiles, q, sub):
    assert jpeg_file_store(jpeg_bytes(source_tiles[0], q, sub)) == (q, sub, True)


def test_read_store_modal_pair_and_lossless(source_tiles, tmp_path):
    blobs = [jpeg_bytes(t, 85, 2) for t in source_tiles[:5]] + [jpeg_bytes(source_tiles[5], 60, 0)]
    st = read_store(blobs)
    assert (st["kind"], st["q"], st["sub"], st["n"]) == ("jpeg", 85, 2, 6)
    assert st["share"] == pytest.approx(5 / 6) and st["exact"] == 1.0
    assert store_label(st) == "JPEG q85 4:2:0"
    paths = []
    for i, t in enumerate(source_tiles[:3]):
        p = tmp_path / f"{i}.png"
        Image.fromarray(t).save(p)
        paths.append(str(p))
    assert read_store(paths)["kind"] == "lossless8"


def test_redegrade_forms(source_tiles):
    x = source_tiles[:3]
    f = redegrade(x, 4.0)                                       # floating point: T_s x
    assert f.dtype == np.float64
    assert np.allclose(f, heat(x.astype(np.float64).transpose(0, 3, 1, 2), 4.0).transpose(0, 2, 3, 1))
    lossless = redegrade(x, 4.0, store=dict(kind="lossless8"))
    assert lossless.dtype == np.uint8 and np.array_equal(lossless, quantize8(f))
    st = dict(kind="jpeg", q=85, sub=2)
    jp = redegrade(x, 4.0, store=st)
    assert np.array_equal(jp, jpeg_roundtrip(quantize8(f), 85, 2))


def test_identity_at_zero_for_lossless_targets(source_tiles):
    """Q8(T_0 x) = x for 8-bit x, so the storage-matched source is the clean source (no JPEG step)."""
    assert np.array_equal(redegrade(source_tiles[:4], 0.0, 0.0, dict(kind="lossless8")), source_tiles[:4])
    assert storage_is_exact(0.0, 0.0, dict(kind="lossless8")) and storage_is_exact(3.0, 1.0, None)
    assert not storage_is_exact(0.0, 0.0, dict(kind="jpeg", q=85, sub=2))


def test_noise_is_keyed_and_reproducible(source_tiles):
    a = redegrade(source_tiles[:2], 2.0, 3.0, key=("k",))
    b = redegrade(source_tiles[:2], 2.0, 3.0, key=("k",))
    c = redegrade(source_tiles[:2], 2.0, 3.0, key=("other",))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert abs(np.std(a - redegrade(source_tiles[:2], 2.0)) - 3.0) < 0.05


def test_non_jpeg_bytes_are_not_jpeg(source_tiles):
    buf = io.BytesIO()
    Image.fromarray(source_tiles[0]).save(buf, format="PNG")
    assert jpeg_file_store(buf.getvalue()) is None
