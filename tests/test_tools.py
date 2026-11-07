"""The release tooling: notebook identification and cleaning, and the anonymity scanner."""
import json

import pytest
from check_anonymity import scan_file, scan_text
from prepare_notebooks import HASH_PREFIX, clean, identify_run


def fake_notebook(levels, computed=None):
    """A minimal executed notebook that embeds the given registration hashes and reports the given computed units."""
    cell = dict(cell_type="code", source="\n".join(f"# {HASH_PREFIX[v]}" for v in levels), metadata={"kaggle": {"x": 1}}, outputs=[])
    if computed is not None:
        cell["outputs"] = [dict(output_type="stream", name="stdout",
                                text="computed in this session: " + "; ".join(f"{u} done (1.0 min)" for u in computed))]
    return dict(cells=[cell], metadata={"kaggle": {"dataSources": [{"sourceId": 1}]}, "papermill": {}, "kernelspec": {}},
                nbformat=4, nbformat_minor=5)


ALL = list(HASH_PREFIX)


@pytest.mark.parametrize("levels, computed, expected", [
    (ALL[:1], None, "EXP1_v1.0"),
    (ALL[:2], None, "EXP2_v2"),
    (ALL[:3], None, "EXP3_v3"),
    (ALL[:4], [], "EXP4a_v4_superseded"),
    (ALL[:4], ["B12_cifar10c"], "EXP4b_v4"),
    (ALL[:5], ["B13_cifar100c", "B14_fmow"], "EXP5_v5"),
    (ALL[:6], ["B22_fmow_ts", "B19_kernelgan"], "EXP6_v6"),
    (ALL[:6], ["B18_imagenet"], "EXP7_v6"),
    (ALL[:7], ["B26_ucm_to_resisc45_v7"], "EXP8_v7_first_start"),
    (ALL[:7], ["B24_imagenet_storage", "B25_imagenet_swap"], "EXP8_v7"),
    (ALL, None, "EXP9_v8_interrupted"),
    (ALL, ["B27_imagenet_unseen"], "EXP9_v8_final"),
])
def test_notebooks_are_identified_from_their_content(levels, computed, expected):
    assert identify_run(fake_notebook(levels, computed))[0] == expected


def test_cleaning_removes_kaggle_metadata_and_keeps_outputs():
    nb = clean(fake_notebook(ALL, ["B27_imagenet_unseen"]))
    assert "kaggle" not in nb["metadata"] and "papermill" not in nb["metadata"]
    assert "kaggle" not in nb["cells"][0]["metadata"] and nb["cells"][0]["outputs"]


def test_cleaning_drops_embedded_zip_copies_and_keeps_figures():
    nb = fake_notebook(ALL, ["B27_imagenet_unseen"])
    link = "<a download='outputs.zip' href='data:application/zip;base64," + "UEsDBA" * 20_000 + "'>download</a>"
    nb["cells"][0]["outputs"] += [dict(output_type="display_data", metadata={}, data={"text/html": link, "text/plain": "<HTML>"}),
                                  dict(output_type="display_data", metadata={}, data={"image/png": "iVBORw0KGgo" * 20_000}),
                                  dict(output_type="display_data", metadata={}, data={"text/html": "<b>a small table</b>"})]
    kinds = [sorted(o.get("data", {"stream": 1})) for o in clean(nb)["cells"][0]["outputs"]]
    assert kinds == [["stream"], ["image/png"], ["text/html"]]


def test_scanner_finds_identifying_strings():
    email = "someone" + "@" + "university.edu"
    token = "hf_" + "A1b2C3d4" * 5
    labels = {label for label, _ in scan_text(f"contact {email}; token {token}; /home/" + "alice/data", [])}
    assert {"e-mail address", "access token", "home folder"} <= labels
    assert [hit for _, hit in scan_text("Written by Jane Doe", ["jane doe"])] == ["Jane Doe"]


def test_scanner_ignores_code():
    assert not list(scan_text("E0 = W@np.exp(-s * LAMB); Z = A@B.T; y = x@W.T + b", []))


def test_scanner_reads_notebook_text_and_metadata(tmp_path):
    raw, cleaned = tmp_path / "raw.ipynb", tmp_path / "clean.ipynb"
    raw.write_text(json.dumps(fake_notebook(ALL, ["B27_imagenet_unseen"])))
    cleaned.write_text(json.dumps(clean(fake_notebook(ALL, []))))
    assert any("metadata 'kaggle'" in h for h in scan_file(raw, []))
    assert scan_file(cleaned, []) == []
