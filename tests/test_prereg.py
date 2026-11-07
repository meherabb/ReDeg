"""The registration documents hash to the values printed in the paper and stored by the final run."""
import json
from pathlib import Path

import verify_prereg

ROOT = Path(__file__).resolve().parents[1]


def test_all_registration_hashes_match(capsys):
    assert verify_prereg.main() == 0
    assert "ALL HASHES MATCH" in capsys.readouterr().out


def test_registration_documents_are_complete():
    reg = json.loads((ROOT / "prereg" / "registered_sha256.json").read_text())
    assert list(reg["documents"]) == ["v1.0", "v2", "v3", "v4", "v5", "v6", "v7", "v8"]
    for ver in reg["documents"]:
        doc = json.loads((ROOT / "prereg" / f"{ver}.json").read_text(encoding="utf-8"))
        assert doc["id"].startswith("Re-Degradation Reveals Resolution | CVPR 2027")


def test_b27_record_was_written_before_any_target_image():
    rec = json.loads((ROOT / "prereg" / "b27_registration_record.json").read_text())
    assert rec["before_targets"] is True and rec["created"] == "2026-10-21T00:53:13" and len(rec["models"]) == 8
    assert sum(r["prediction"] == "harm" for r in rec["models"].values()) == 3
