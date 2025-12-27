"""
episode_49.hdf5 구조 요약
- action: dataset, shape (500, 14), dtype float32
- observations: group
  - images: group
    - wrist1: dataset, shape (500, 480, 640, 3), dtype uint8
    - wrist2: dataset, shape (500, 480, 640, 3), dtype uint8
  - qpos: dataset, shape (500, 16), dtype float32 (앞에 14개가 joint_positions, 뒤에 2개가 gripper_position)
  - qvel: dataset, shape (500, 16), dtype float32 (앞에 14개가 joint_velocities, 뒤에 2개가 gripper_velocity)

우리 데이터는 총 7자유도 로봇팔 2대라 총 16자유도이다
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import h5py
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def collect_structure(h5_obj):
    """Return metadata for group or dataset."""
    info = {
        "type": "group" if isinstance(h5_obj, h5py.Group) else "dataset",
        "attrs": {
            key: _maybe_to_native(value)
            for key, value in h5_obj.attrs.items()
        },
    }
    if isinstance(h5_obj, h5py.Dataset):
        info["shape"] = h5_obj.shape
        info["dtype"] = str(h5_obj.dtype)
    return info


def _maybe_to_native(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return value


def describe_hdf5(path: Path):
    structure = {}

    with h5py.File(path, "r") as h5_file:
        def visitor(name, obj):
            structure[name or "/"] = collect_structure(obj)

        h5_file.visititems(visitor)

    return structure


def analyze_pkl_root(
    root: Path,
    samples_per_run: int = 1,
    max_runs: int | None = None,
) -> dict[str, Any]:
    root = root.expanduser()
    run_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if max_runs is not None:
        run_dirs = run_dirs[:max_runs]

    per_run_frames: dict[str, int] = {}
    key_shapes: dict[str, set[tuple | None]] = defaultdict(set)
    key_dtypes: dict[str, set[str]] = defaultdict(set)
    key_counts: Counter[str] = Counter()
    key_examples: dict[str, dict[str, Any]] = {}

    total_frames = 0
    sampled_files = 0

    for run_dir in run_dirs:
        pkl_files = sorted(run_dir.glob("*.pkl"))
        frame_count = len(pkl_files)
        per_run_frames[run_dir.name] = frame_count
        total_frames += frame_count

        for pkl_path in pkl_files[:samples_per_run]:
            data = _load_pickle(pkl_path)
            sampled_files += 1
            for key, value in data.items():
                shape = _shape_of(value)
                dtype = _dtype_of(value)
                key_shapes[key].add(shape)
                key_dtypes[key].add(dtype)
                key_counts[key] += 1
                if key not in key_examples:
                    key_examples[key] = {
                        "run": run_dir.name,
                        "file": pkl_path.name,
                        "shape": shape,
                        "dtype": dtype,
                    }

    summary = {
        "root": str(root),
        "run_dirs_scanned": len(run_dirs),
        "samples_per_run": samples_per_run,
        "sampled_files": sampled_files,
        "total_frames": total_frames,
        "per_run_frame_counts": per_run_frames,
        "keys": {},
    }

    for key in sorted(key_shapes.keys()):
        summary["keys"][key] = {
            "shapes": sorted(_stringify_shape(s) for s in key_shapes[key]),
            "dtypes": sorted(key_dtypes[key]),
            "seen_in_samples": key_counts[key],
            "example": key_examples.get(key),
        }

    return summary


def _shape_of(value: Any):
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def _dtype_of(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        return str(dtype)
    if isinstance(value, (bytes, bytearray)):
        return "bytes"
    return type(value).__name__


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {path}, got {type(data)}")
    return data


def _stringify_shape(shape: tuple | None) -> str:
    if shape is None:
        return "scalar/unknown"
    return "x".join(str(dim) for dim in shape)


TARGET_IMAGE_SIZE = (480, 640)  # (height, width)
GRIPPER_INDICES = (7, 15)  # zero-based indices in the original 16-dof vectors


def convert_episode_to_hdf5(
    run_dir: Path,
    output_path: Path,
    *,
    qpos_dims: int = 16,
    qvel_dims: int = 16,
    action_dims: int = 16,
    compression: Optional[str] = "gzip",
    compression_opts: Optional[int] = 4,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser()
    output_path = output_path.expanduser()
    pkl_files = sorted(run_dir.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No PKL files found in {run_dir}")

    first = _load_pickle(pkl_files[0])
    num_frames = len(pkl_files)

    resized_shape = (*TARGET_IMAGE_SIZE, first["wrist1_rgb"].shape[2])

    wrist1 = np.empty((num_frames, *resized_shape), dtype=first["wrist1_rgb"].dtype)
    wrist2 = np.empty_like(wrist1)
    qpos = np.empty((num_frames, qpos_dims), dtype=np.float32)
    qvel = np.empty((num_frames, qvel_dims), dtype=np.float32)
    actions = np.empty((num_frames, action_dims), dtype=np.float32)
    timestamps = np.empty(num_frames, dtype=np.float64)

    for idx, pkl_path in enumerate(pkl_files):
        data = first if idx == 0 else _load_pickle(pkl_path)
        wrist1[idx] = _resize_rgb(data["wrist1_rgb"])
        wrist2[idx] = _resize_rgb(data["wrist2_rgb"])
        qpos[idx] = _reorder_and_slice(
            data["joint_positions"], qpos_dims, GRIPPER_INDICES
        )
        qvel[idx] = _reorder_and_slice(
            data["joint_velocities"], qvel_dims, GRIPPER_INDICES
        )
        actions[idx] = _reorder_and_slice(
            data["control"], action_dims, GRIPPER_INDICES
        )
        timestamps[idx] = _filename_timestamp(pkl_path.name)

    compression_kwargs = _compression_kwargs(compression, compression_opts)

    with h5py.File(output_path, "w") as h5f:
        h5f.create_dataset("action", data=actions, **compression_kwargs)

        obs_grp = h5f.create_group("observations")
        img_grp = obs_grp.create_group("images")
        img_grp.create_dataset("wrist1", data=wrist1, **compression_kwargs)
        img_grp.create_dataset("wrist2", data=wrist2, **compression_kwargs)

        obs_grp.create_dataset("qpos", data=qpos, **compression_kwargs)
        obs_grp.create_dataset("qvel", data=qvel, **compression_kwargs)
        h5f.create_dataset("timestamps", data=timestamps, **compression_kwargs)

    return {
        "frames": num_frames,
        "run_dir": str(run_dir),
        "output": str(output_path),
        "wrist_shape": resized_shape,
    }


def convert_all_runs(
    root_dir: Path,
    output_dir: Path,
    *,
    qpos_dims: int = 16,
    qvel_dims: int = 16,
    action_dims: int = 16,
    compression: Optional[str] = "gzip",
    compression_opts: Optional[int] = 4,
) -> dict[str, Any]:
    root_dir = root_dir.expanduser()
    output_act_dir = (output_dir.expanduser() / "act").resolve()
    output_act_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    summaries = []
    for idx, run_dir in enumerate(run_dirs):
        target_path = output_act_dir / f"episode_{idx}.hdf5"
        summary = convert_episode_to_hdf5(
            run_dir=run_dir,
            output_path=target_path,
            qpos_dims=qpos_dims,
            qvel_dims=qvel_dims,
            action_dims=action_dims,
            compression=compression,
            compression_opts=compression_opts,
        )
        summaries.append(summary)

    return {
        "run_root": str(root_dir),
        "output_dir": str(output_act_dir),
        "converted_runs": len(summaries),
        "runs": summaries,
    }


def _slice_vector(value: Any, dims: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.shape[0] < dims:
        raise ValueError(
            f"Vector length {arr.shape[0]} is shorter than required dims {dims}"
        )
    return arr[:dims].astype(np.float32, copy=False)


def _reorder_and_slice(
    value: Any,
    dims: int,
    gripper_indices: Iterable[int],
) -> np.ndarray:
    arr = np.asarray(value).reshape(-1)
    if arr.shape[0] < dims:
        raise ValueError(
            f"Vector length {arr.shape[0]} is shorter than required dims {dims}"
        )
    sliced = arr[:dims]
    idx = sorted(set(int(i) for i in gripper_indices))
    gripper_vals = sliced[idx]
    mask = np.ones(sliced.shape[0], dtype=bool)
    mask[idx] = False
    reordered = np.concatenate([sliced[mask], gripper_vals])
    return reordered.astype(np.float32, copy=False)


def _resize_rgb(array: np.ndarray) -> np.ndarray:
    target_h, target_w = TARGET_IMAGE_SIZE
    if array.shape[0] == target_h and array.shape[1] == target_w:
        return array
    if cv2 is not None:
        resized = cv2.resize(array, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return resized.astype(array.dtype, copy=False)
    from PIL import Image

    image = Image.fromarray(array)
    resized = image.resize((target_w, target_h), Image.BILINEAR)
    return np.asarray(resized, dtype=array.dtype)


def _compression_kwargs(
    compression: Optional[str],
    compression_opts: Optional[int],
) -> dict[str, Any]:
    if compression is None or compression.lower() == "none":
        return {}
    kwargs: dict[str, Any] = {"compression": compression}
    if compression_opts is not None:
        kwargs["compression_opts"] = compression_opts
    return kwargs


def _filename_timestamp(filename: str) -> float:
    stem = filename[:-4] if filename.endswith(".pkl") else filename
    try:
        return datetime.fromisoformat(stem).timestamp()
    except ValueError:
        return 0.0


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect HDF5 hierarchy, analyze PKL datasets, "
            "or convert a single episode directory into HDF5."
        )
    )
    parser.add_argument(
        "hdf5_path",
        nargs="?",
        default="/home/demo-panda/workspace/gello_software/episode_0.hdf5",
        help="Target HDF5 file to inspect.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level for printing.",
    )
    parser.add_argument(
        "--pkl-root",
        type=str,
        help="Analyze PKL dataset under this root directory.",
    )
    parser.add_argument(
        "--samples-per-run",
        type=int,
        default=1,
        help="How many PKL files per run directory to sample for structure.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        help="Limit number of run directories to scan (useful for testing).",
    )
    parser.add_argument(
        "--convert-run",
        type=str,
        help="Path to an episode directory containing PKL files.",
    )
    parser.add_argument(
        "--output-hdf5",
        type=str,
        help="Destination HDF5 path (required with --convert-run).",
    )
    parser.add_argument(
        "--qpos-dims",
        type=int,
        default=16,
        help="How many joint position entries to keep in qpos.",
    )
    parser.add_argument(
        "--qvel-dims",
        type=int,
        default=16,
        help="How many joint velocity entries to keep in qvel.",
    )
    parser.add_argument(
        "--action-dims",
        type=int,
        default=16,
        help="How many control entries to keep in action.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="gzip",
        help="Compression algorithm for datasets (use 'none' to disable).",
    )
    parser.add_argument(
        "--compression-opts",
        type=int,
        default=4,
        help="Compression level/options (ignored if compression is 'none').",
    )
    parser.add_argument(
        "--convert-all",
        action="store_true",
        help="Convert every episode directory inside --dataset-root.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/home/demo-panda/workspace/gello_software/etri_switchoff/gello",
        help="Directory containing episode subdirectories (used by --convert-all).",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="/home/demo-panda/workspace/gello_software/etri_switchoff",
        help="Base directory where ACT outputs will be stored (used by --convert-all).",
    )
    args = parser.parse_args()

    if args.convert_all:
        summary = convert_all_runs(
            root_dir=Path(args.dataset_root),
            output_dir=Path(args.output_base),
            qpos_dims=args.qpos_dims,
            qvel_dims=args.qvel_dims,
            action_dims=args.action_dims,
            compression=args.compression,
            compression_opts=args.compression_opts,
        )
        print(json.dumps(summary, indent=args.indent, ensure_ascii=False))
        return

    if args.convert_run:
        if not args.output_hdf5:
            raise SystemExit("--output-hdf5 is required when using --convert-run")
        summary = convert_episode_to_hdf5(
            Path(args.convert_run),
            Path(args.output_hdf5),
            qpos_dims=args.qpos_dims,
            qvel_dims=args.qvel_dims,
            action_dims=args.action_dims,
            compression=args.compression,
            compression_opts=args.compression_opts,
        )
        print(json.dumps(summary, indent=args.indent, ensure_ascii=False))
        return

    if args.pkl_root:
        summary = analyze_pkl_root(
            Path(args.pkl_root),
            samples_per_run=max(1, args.samples_per_run),
            max_runs=args.max_runs,
        )
        print(json.dumps(summary, indent=args.indent, ensure_ascii=False))
    else:
        structure = describe_hdf5(Path(args.hdf5_path))
        print(json.dumps(structure, indent=args.indent, ensure_ascii=False))


if __name__ == "__main__":
    main()