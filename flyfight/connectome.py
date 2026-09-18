"""MaleCNS connectome import and memory-mapped runtime access.

The importer preserves official body IDs and directed synapse counts.  It does
not invent neuron coordinates, dynamics, signs, or activity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterator
import uuid

import numpy as np


FORMAT_NAME = "flyfight-malecns-csr"
FORMAT_VERSION = 1
DATASET_NAME = "MaleCNS v1.0"
EXPECTED_NEURONS = 166_700
PUBLISHED_NEURONS = 166_691
DOWNLOAD_PAGE = "https://male-cns.janelia.org/download/"
OFFICIAL_BUCKET = "gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/"
OFFICIAL_HTTPS_BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
ANNOTATIONS_FILENAME = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
WEIGHTS_FILENAME = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"


def _arrow_modules():
    try:
        import pyarrow as pa
        import pyarrow.feather as feather
        import pyarrow.ipc as ipc
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "MaleCNS import requires pyarrow (install with: python -m pip install pyarrow)"
        ) from exc
    return pa, feather, ipc


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_batches(path: Path) -> Iterator[object]:
    pa, _, ipc = _arrow_modules()
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        for batch_index in range(reader.num_record_batches):
            yield reader.get_batch(batch_index)


def _required_column(schema, candidates: tuple[str, ...], description: str) -> str:
    for name in candidates:
        if schema.get_field_index(name) >= 0:
            return name
    raise ValueError(f"Missing {description} column; expected one of {candidates!r}")


def _numpy_column(batch, name: str, dtype: np.dtype) -> np.ndarray:
    column = batch.column(batch.schema.get_field_index(name))
    if column.null_count:
        raise ValueError(f"Column {name!r} contains null values")
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=dtype)


def _map_body_ids(body_ids: np.ndarray, values: np.ndarray, label: str) -> np.ndarray:
    positions = np.searchsorted(body_ids, values)
    valid = positions < body_ids.size
    valid[valid] &= body_ids[positions[valid]] == values[valid]
    if not np.all(valid):
        sample = values[~valid][:5].tolist()
        raise ValueError(f"{label} contains body IDs absent from annotations: {sample!r}")
    return positions


@dataclass(frozen=True)
class Connectome:
    """Memory-mapped MaleCNS graph with rows as presynaptic neurons."""

    root: Path
    body_ids: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    manifest: dict

    @classmethod
    def load(cls, root: str | Path, *, mmap_mode: str | None = "r") -> "Connectome":
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT_NAME or manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported FlyFight connectome format")
        graph = manifest["graph"]
        result = cls(
            root=root,
            body_ids=np.load(root / graph["body_ids_file"], mmap_mode=mmap_mode),
            indptr=np.load(root / graph["indptr_file"], mmap_mode=mmap_mode),
            indices=np.load(root / graph["indices_file"], mmap_mode=mmap_mode),
            weights=np.load(root / graph["weights_file"], mmap_mode=mmap_mode),
            manifest=manifest,
        )
        result.validate()
        return result

    @property
    def neuron_count(self) -> int:
        return int(self.body_ids.size)

    @property
    def edge_count(self) -> int:
        return int(self.weights.size)

    def validate(self) -> None:
        if self.body_ids.ndim != 1 or np.any(self.body_ids[1:] <= self.body_ids[:-1]):
            raise ValueError("body_ids must be unique and strictly increasing")
        if self.indptr.shape != (self.neuron_count + 1,):
            raise ValueError("indptr length does not match neuron count")
        if self.indptr[0] != 0 or self.indptr[-1] != self.edge_count:
            raise ValueError("indptr bounds do not match edge count")
        if np.any(self.indptr[1:] < self.indptr[:-1]):
            raise ValueError("indptr must be nondecreasing")
        if self.indices.shape != self.weights.shape:
            raise ValueError("indices and weights lengths differ")
        if self.indices.size and int(self.indices.max()) >= self.neuron_count:
            raise ValueError("graph contains an out-of-range postsynaptic index")
        if np.any(self.weights <= 0):
            raise ValueError("synapse weights must be positive")

    def outgoing(self, neuron_index: int) -> tuple[np.ndarray, np.ndarray]:
        start, stop = int(self.indptr[neuron_index]), int(self.indptr[neuron_index + 1])
        return self.indices[start:stop], self.weights[start:stop]

    def index_of(self, body_id: int) -> int:
        """Resolve an official body ID without allocating a 166k-entry dict."""

        position = int(np.searchsorted(self.body_ids, np.uint64(body_id)))
        if position >= self.neuron_count or int(self.body_ids[position]) != body_id:
            raise KeyError(body_id)
        return position


def import_malecns(
    annotations_path: str | Path,
    weights_path: str | Path,
    output_dir: str | Path,
    *,
    expected_neurons: int | None = EXPECTED_NEURONS,
    overwrite: bool = False,
) -> dict:
    """Compile official MaleCNS Feather tables into a memory-mappable CSR directory.

    ``expected_neurons=None`` is the explicit exploratory mode for fixtures or a
    future release.  Production imports should retain the v1.0 count check.
    """

    pa, feather, _ = _arrow_modules()
    annotations_path = Path(annotations_path).resolve()
    weights_path = Path(weights_path).resolve()
    output_dir = Path(output_dir).resolve()
    for path in (annotations_path, weights_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_dir}")

    first_batch = next(_record_batches(weights_path), None)
    if first_batch is None:
        raise ValueError("Connection Feather file contains no record batches")
    pre_column = _required_column(first_batch.schema, ("body_pre", "pre"), "presynaptic body ID")
    post_column = _required_column(first_batch.schema, ("body_post", "post"), "postsynaptic body ID")
    weight_column = _required_column(first_batch.schema, ("weight", "syn_count"), "connection weight")

    annotations = feather.read_table(annotations_path, memory_map=True)
    body_column = _required_column(annotations.schema, ("bodyId", "body"), "annotation body ID")
    superclass_column = _required_column(annotations.schema, ("superclass",), "annotation superclass")
    annotation_ids_all = np.asarray(
        annotations[body_column].to_numpy(zero_copy_only=False), dtype=np.uint64
    )
    if np.unique(annotation_ids_all).size != annotation_ids_all.size:
        raise ValueError("Annotation body IDs must be unique")
    superclass = annotations[superclass_column].to_pylist()
    neuron_mask = np.fromiter(
        (value is not None and str(value).strip() != "" for value in superclass),
        dtype=bool,
        count=len(superclass),
    )
    selected_rows = np.flatnonzero(neuron_mask)
    selected_ids = annotation_ids_all[neuron_mask]
    selected_order = np.argsort(selected_ids, kind="stable")
    body_ids = selected_ids[selected_order]
    annotations = annotations.take(pa.array(selected_rows[selected_order]))
    neuron_count = int(body_ids.size)
    if expected_neurons is not None and neuron_count != expected_neurons:
        raise ValueError(
            f"Expected {expected_neurons:,} annotated MaleCNS v1.0 neurons, found {neuron_count:,}; "
            "pass expected_neurons=None only for an intentional non-production import"
        )

    # The full flat graph also contains unproofread fragments. Keep only edges
    # whose endpoints are both members of the released annotated-neuron census.
    edge_count = 0
    contact_count = 0
    for batch in _record_batches(weights_path):
        pre = _numpy_column(batch, pre_column, np.uint64)
        post = _numpy_column(batch, post_column, np.uint64)
        weight = _numpy_column(batch, weight_column, np.int64)
        if np.any(weight <= 0):
            raise ValueError("Connection weights must be positive integers")
        pre_pos = np.searchsorted(body_ids, pre)
        post_pos = np.searchsorted(body_ids, post)
        valid = (pre_pos < neuron_count) & (post_pos < neuron_count)
        valid[valid] &= body_ids[pre_pos[valid]] == pre[valid]
        valid[valid] &= body_ids[post_pos[valid]] == post[valid]
        edge_count += int(valid.sum())
        contact_count += int(weight[valid].sum(dtype=np.int64))

    degrees = np.zeros(neuron_count, dtype=np.uint64)
    for batch in _record_batches(weights_path):
        pre = _numpy_column(batch, pre_column, np.uint64)
        post = _numpy_column(batch, post_column, np.uint64)
        pre_pos = np.searchsorted(body_ids, pre)
        post_pos = np.searchsorted(body_ids, post)
        valid = (pre_pos < neuron_count) & (post_pos < neuron_count)
        valid[valid] &= body_ids[pre_pos[valid]] == pre[valid]
        valid[valid] &= body_ids[post_pos[valid]] == post[valid]
        pre_index = pre_pos[valid]
        degrees += np.bincount(pre_index, minlength=neuron_count).astype(np.uint64)

    build_dir = output_dir.with_name(f".{output_dir.name}.build-{uuid.uuid4().hex}")
    build_dir.mkdir(parents=True)
    try:
        np.save(build_dir / "body_ids.npy", body_ids, allow_pickle=False)
        indptr = np.empty(neuron_count + 1, dtype=np.uint64)
        indptr[0] = 0
        np.cumsum(degrees, out=indptr[1:])
        np.save(build_dir / "indptr.npy", indptr, allow_pickle=False)
        indices = np.lib.format.open_memmap(
            build_dir / "indices.npy", mode="w+", dtype=np.uint32, shape=(edge_count,)
        )
        edge_weights = np.lib.format.open_memmap(
            build_dir / "weights.npy", mode="w+", dtype=np.uint32, shape=(edge_count,)
        )
        cursor = indptr[:-1].copy()
        for batch in _record_batches(weights_path):
            pre = _numpy_column(batch, pre_column, np.uint64)
            post = _numpy_column(batch, post_column, np.uint64)
            weight = _numpy_column(batch, weight_column, np.uint64)
            if weight.size and int(weight.max()) > np.iinfo(np.uint32).max:
                raise ValueError("Connection weight exceeds uint32 storage")
            pre_pos = np.searchsorted(body_ids, pre)
            post_pos = np.searchsorted(body_ids, post)
            valid = (pre_pos < neuron_count) & (post_pos < neuron_count)
            valid[valid] &= body_ids[pre_pos[valid]] == pre[valid]
            valid[valid] &= body_ids[post_pos[valid]] == post[valid]
            pre_index = pre_pos[valid]
            post_index = post_pos[valid]
            weight = weight[valid]
            if pre_index.size == 0:
                continue
            # Group a batch by source, then write every edge with vectorized
            # NumPy assignments.  ``cursor`` carries row offsets across batches.
            order = np.argsort(pre_index, kind="stable")
            grouped_pre = pre_index[order]
            group_start = np.empty(grouped_pre.size, dtype=bool)
            group_start[0] = True
            group_start[1:] = grouped_pre[1:] != grouped_pre[:-1]
            starts = np.flatnonzero(group_start)
            local_offset = np.arange(grouped_pre.size, dtype=np.uint64)
            lengths = np.diff(np.r_[starts, np.int64(grouped_pre.size)])
            local_offset -= np.repeat(starts.astype(np.uint64), lengths)
            destination = cursor[grouped_pre] + local_offset
            indices[destination] = post_index[order]
            edge_weights[destination] = weight[order]
            cursor += np.bincount(pre_index, minlength=neuron_count).astype(np.uint64)
        indices.flush()
        edge_weights.flush()
        del indices, edge_weights

        feather.write_feather(annotations, build_dir / "annotations.feather", compression="zstd")
        sources = {
            "annotations": {
                "path": str(annotations_path),
                "official_url": OFFICIAL_HTTPS_BASE + ANNOTATIONS_FILENAME,
                "bytes": annotations_path.stat().st_size,
                "sha256": sha256_file(annotations_path),
            },
            "connections": {
                "path": str(weights_path),
                "official_url": OFFICIAL_HTTPS_BASE + WEIGHTS_FILENAME,
                "bytes": weights_path.stat().st_size,
                "sha256": sha256_file(weights_path),
            },
        }
        manifest = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "name": DATASET_NAME,
                "expected_neurons": expected_neurons,
                "published_neuron_count": PUBLISHED_NEURONS,
                "release_annotation_neuron_count": neuron_count,
                "count_note": (
                    "The v1.0 annotation file contains 166,700 non-empty superclass rows, "
                    "nine more than the paper/blog census of 166,691. This import preserves "
                    "the release table instead of silently dropping unidentified rows."
                ),
                "download_page": DOWNLOAD_PAGE,
                "official_bucket": OFFICIAL_BUCKET,
                "license": "CC-BY-4.0",
                "attribution": "MaleCNS collaboration: HHMI Janelia FlyEM, Google Research, and collaborators",
                "sources": sources,
            },
            "graph": {
                "neuron_count": neuron_count,
                "annotated_neuron_count": neuron_count,
                "unannotated_neuron_count": 0,
                "edge_count": edge_count,
                "synaptic_contact_count": contact_count,
                "orientation": "CSR rows are presynaptic; indices are postsynaptic",
                "body_ids_file": "body_ids.npy",
                "indptr_file": "indptr.npy",
                "indices_file": "indices.npy",
                "weights_file": "weights.npy",
                "annotations_file": "annotations.feather",
                "annotations_note": "Contains all source rows with a non-empty superclass, sorted by body ID.",
                "weights_semantics": "official unsigned synapse counts; neurotransmitter signs are not inferred",
            },
            "coordinates": {
                "included": False,
                "reason": (
                    "The annotation and connectome-weight tables do not provide complete neuron geometry. "
                    "Use official MaleCNS v1.0 skeletons or soma data and an explicit coordinate-space importer."
                ),
                "synthetic_coordinates": False,
            },
            "activity": {
                "included": False,
                "reason": "The connectome release supplies anatomy and connectivity, not measured live activity.",
            },
        }
        (build_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        Connectome.load(build_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        build_dir.replace(output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise
