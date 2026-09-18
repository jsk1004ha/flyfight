import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
feather = pytest.importorskip("pyarrow.feather")

from flyfight.connectome import Connectome, import_malecns


def write_fixture(root: Path, *, unknown_endpoint: bool = False) -> tuple[Path, Path]:
    annotations = pa.table(
        {
            "bodyId": pa.array([30, 10, 20], type=pa.uint64()),
            "superclass": ["vnc_motor", "ol_sensory", "cb_intrinsic"],
            "type": ["motor", "visual", None],
            "somaSide": ["R", "L", None],
        }
    )
    connections = pa.table(
        {
            "body_pre": pa.array([10, 10, 20], type=pa.uint64()),
            "body_post": pa.array([20, 30, 99 if unknown_endpoint else 30], type=pa.uint64()),
            "weight": pa.array([4, 2, 7], type=pa.int64()),
        }
    )
    annotation_path = root / "annotations.feather"
    weights_path = root / "weights.feather"
    feather.write_feather(annotations, annotation_path)
    feather.write_feather(connections, weights_path, chunksize=2)
    return annotation_path, weights_path


def test_import_preserves_ids_annotations_and_directed_csr(tmp_path):
    annotation_path, weights_path = write_fixture(tmp_path)
    output = tmp_path / "compiled"
    manifest = import_malecns(
        annotation_path, weights_path, output, expected_neurons=None
    )
    graph = Connectome.load(output)

    np.testing.assert_array_equal(graph.body_ids, [10, 20, 30])
    targets, weights = graph.outgoing(0)
    np.testing.assert_array_equal(targets, [1, 2])
    np.testing.assert_array_equal(weights, [4, 2])
    targets, weights = graph.outgoing(1)
    np.testing.assert_array_equal(targets, [2])
    np.testing.assert_array_equal(weights, [7])
    assert graph.neuron_count == 3
    assert graph.edge_count == 3
    assert graph.index_of(20) == 1
    with pytest.raises(KeyError):
        graph.index_of(99)
    assert manifest["graph"]["synaptic_contact_count"] == 13
    assert manifest["coordinates"]["included"] is False
    assert manifest["coordinates"]["synthetic_coordinates"] is False
    assert manifest["activity"]["included"] is False

    sorted_annotations = feather.read_table(output / "annotations.feather")
    assert sorted_annotations["bodyId"].to_pylist() == [10, 20, 30]
    assert sorted_annotations["type"].to_pylist() == ["visual", None, "motor"]
    stored = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert stored["dataset"]["sources"]["annotations"]["sha256"]
    assert stored["dataset"]["license"] == "CC-BY-4.0"


def test_import_rejects_wrong_release_count(tmp_path):
    annotation_path, weights_path = write_fixture(tmp_path)
    with pytest.raises(ValueError, match="Expected 166,700"):
        import_malecns(annotation_path, weights_path, tmp_path / "compiled")


def test_import_drops_graph_fragment_without_neuron_annotation(tmp_path):
    annotation_path, weights_path = write_fixture(tmp_path, unknown_endpoint=True)
    output = tmp_path / "compiled"
    manifest = import_malecns(
        annotation_path, weights_path, output, expected_neurons=None
    )
    graph = Connectome.load(output)
    np.testing.assert_array_equal(graph.body_ids, [10, 20, 30])
    assert manifest["graph"]["annotated_neuron_count"] == 3
    assert manifest["graph"]["unannotated_neuron_count"] == 0
    assert graph.edge_count == 2


def test_cli_imports_local_files_without_downloading(tmp_path):
    annotation_path, weights_path = write_fixture(tmp_path)
    output = tmp_path / "cli-output"
    completed = subprocess.run(
        [
            sys.executable,
            "import_mcns.py",
            str(tmp_path),
            str(output),
            "--annotations",
            annotation_path.name,
            "--weights",
            weights_path.name,
            "--allow-nonstandard-count",
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert '"neuron_count": 3' in completed.stdout
    assert Connectome.load(output).edge_count == 3
