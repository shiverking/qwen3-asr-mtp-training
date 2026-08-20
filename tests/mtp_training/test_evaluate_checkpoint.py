from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mtp_training.evaluate_checkpoint import (
    _filter_dataset_rows,
    _sample_ids_from_report,
)


def test_verifier_sample_ids_filter_dataset_in_report_order(tmp_path):
    report = tmp_path / "verify.json"
    report.write_text(
        json.dumps({"results": [{"id": "b"}, {"id": "a"}]}), encoding="utf-8"
    )
    dataset = SimpleNamespace(rows=[{"id": "a"}, {"id": "b"}, {"id": "c"}])
    ids = _sample_ids_from_report(str(report))
    _filter_dataset_rows(dataset, ids)
    assert [row["id"] for row in dataset.rows] == ["b", "a"]


def test_verifier_sample_ids_reject_missing_id(tmp_path):
    report = tmp_path / "verify.json"
    report.write_text(json.dumps({"results": [{"id": "missing"}]}), encoding="utf-8")
    dataset = SimpleNamespace(rows=[{"id": "a"}])
    with pytest.raises(ValueError, match="missing"):
        _filter_dataset_rows(dataset, _sample_ids_from_report(str(report)))
