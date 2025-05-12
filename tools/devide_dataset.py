#!/usr/bin/env python3
"""
Script to reproducibly subsample a Dex-Net NPZ dataset, preserving original per-prefix NPZ bundles.
Integrates bundles to count total samples, subsamples at given fraction, then rebundles
into same bundle_size, and regenerates train/val splits and metadata using subsample positions.
Usage:
    python subsample_and_bundle.py
"""
import os
import re
import shutil
import json
import numpy as np
from math import ceil


def copy_base_structure(src, dst):
    """
    Copy directory structure from src to dst, excluding top-level 'tensors' and 'splits'.
    """
    def ignore_root(dirpath, names):
        if os.path.abspath(dirpath) == os.path.abspath(src):
            return ['tensors', 'splits']
        return []

    shutil.copytree(src, dst, ignore=ignore_root)
    # Recreate empty tensors/ and splits/image_wise
    os.makedirs(os.path.join(dst, 'tensors'), exist_ok=True)
    os.makedirs(os.path.join(dst, 'splits', 'image_wise'), exist_ok=True)


def gather_prefix_bundles(tensors_src):
    """
    Scan tensors_src for files '<prefix>_NNNNN.npz'.
    Return dict: prefix -> list of (bundle_idx, filepath), sorted by bundle_idx.
    """
    pat = re.compile(r'^(?P<prefix>.+)_(?P<idx>\d{5})\.npz$')
    prefix2files = {}
    for fn in os.listdir(tensors_src):
        m = pat.match(fn)
        if not m:
            continue
        prefix = m.group('prefix')
        idx = int(m.group('idx'))  # original bundle index
        prefix2files.setdefault(prefix, []).append((idx, os.path.join(tensors_src, fn)))
    for prefix in prefix2files:
        prefix2files[prefix].sort(key=lambda x: x[0])
    return prefix2files


def integrate_all(prefix2files):
    """
    For each prefix, load each bundle NPZ and concatenate along sample axis (axis=0).
    Print full shape per prefix, e.g., (1000, 32, 32, 1).
    Return dict: prefix -> integrated array, and int N_total.
    """
    integrated = {}
    N_total = None
    for prefix, bundles in prefix2files.items():
        parts = []
        for idx, path in bundles:
            with np.load(path) as data:
                keys = data.files
                if len(keys) != 1:
                    raise ValueError(f"Expected 1 array in {path}, found {len(keys)}")
                arr = data[keys[0]]
            parts.append(arr)
        full = np.concatenate(parts, axis=0)
        integrated[prefix] = full
        #print(f"Prefix '{prefix}': integrated shape = {full.shape}")
        if N_total is None:
            N_total = full.shape[0]
        elif N_total != full.shape[0]:
            raise ValueError(f"Inconsistent sample counts: {N_total} vs {full.shape[0]} for prefix {prefix}")
    return integrated, N_total


def subsample_indices(N_total, fraction, seed):
    """
    Randomly choose a subset of sample indices without replacement, then sort.
    Print n_sub and N_total. Returns sorted subsample positions.
    """
    rng = np.random.RandomState(seed)
    n_sub = int(N_total * fraction)
    print(f"Sampling {n_sub} out of {N_total} total samples (fraction={fraction})")
    sub_idxs = rng.choice(N_total, size=n_sub, replace=False)
    return np.sort(sub_idxs)


def bundle_subsamples(integrated, sub_idxs, dst_dir, bundle_size):
    """
    For each prefix, extract subsampled entries along sample axis and rebundle
    into files '<prefix>_NNNNN.npz' of up to bundle_size samples.
    """
    os.makedirs(dst_dir, exist_ok=True)
    n_sub = len(sub_idxs)
    n_bundles = ceil(n_sub / bundle_size)
    for prefix, full in integrated.items():
        sel = full[sub_idxs]  # shape (n_sub, ...)
        for b in range(n_bundles):
            start = b * bundle_size
            end = min(start + bundle_size, n_sub)
            arr_bundle = sel[start:end]
            out_name = f"{prefix}_{b:05d}.npz"
            out_path = os.path.join(dst_dir, out_name)
            np.savez_compressed(out_path, arr_bundle)
        print(f"Wrote {n_bundles} bundles for prefix '{prefix}' to {dst_dir}")


def create_splits(n_sub, train_fraction, seed, split_dir):
    """
    Split subsamples (positions 0..n_sub-1) into train/val using given fraction.
    Write train/val index files relative to subsample positions, and metadata.json.
    """
    rng = np.random.RandomState(seed)
    positions = np.arange(n_sub)
    rng.shuffle(positions)

    n_train = int(n_sub * train_fraction)
    train_pos = np.sort(positions[:n_train]).astype(np.int32)
    val_pos   = np.sort(positions[n_train:]).astype(np.int32)

    os.makedirs(split_dir, exist_ok=True)
    np.savez_compressed(os.path.join(split_dir, 'train_indices.npz'), train_pos)
    np.savez_compressed(os.path.join(split_dir, 'val_indices.npz'),   val_pos)

    meta = {
        'num_train': int(len(train_pos)),
        'num_val':   int(len(val_pos)),
        'total':     int(n_sub)
    }
    meta_org = {
    "field_name": "index",
    "train_pct": 0.8
    }
    with open(os.path.join(split_dir, 'metadata_subsample.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(split_dir, 'metadata.json'), 'w') as f:
        json.dump(meta_org, f, indent=2)
    print(f"Wrote splits and metadata to {split_dir}")


def main(fr):
    # ─── User Configuration ───
    base_dirs       = ["/home/deepstation/Downloads/scale_test/dexnet_150", "/home/deepstation/Downloads/scale_test/egad_150", "/home/deepstation/Downloads/scale_test/gfdb_006_150_v2", "/home/deepstation/Downloads/scale_test/gfdb_030_150_v2", "/home/deepstation/Downloads/scale_test/gfdb_070_150_v2", "/home/deepstation/Downloads/scale_test/gfdb_006_030_070_150", "/home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/primitive"]
    fraction       = fr      # Subsample rate
    train_fraction = 0.8      # Train split ratio
    seeds           = [123]      # RNG seed for reproducibility
    seeds           = [1,2,3,4,5]      # RNG seed for reproducibility
    bundle_size    = 1000     # Samples per NPZ bundle
    count_only = False
    # ──────────────────────────

    for base_dir in base_dirs:
        for seed in seeds:
            out_dir        = f"{base_dir}_{int(fraction*100):03}_seed{seed}"
            if not count_only:
                if os.path.exists(out_dir):
                    raise RuntimeError(f"Output directory {out_dir} already exists!")

            # 1) Copy base structure excluding old tensors/ and splits/
            if not count_only:
                copy_base_structure(base_dir, out_dir)

            # 2) Gather all prefix bundles
            tensors_src  = os.path.join(base_dir, 'tensors')
            prefix2files = gather_prefix_bundles(tensors_src)

            # 3) Integrate bundles, report shapes
            integrated, N_total = integrate_all(prefix2files)

            # 4) Subsample positions
            sub_idxs = subsample_indices(N_total, fraction, seed)

            # 5) Create new bundles from subsamples
            if not count_only:
                tensors_dst = os.path.join(out_dir, 'tensors')
                bundle_subsamples(integrated, sub_idxs, tensors_dst, bundle_size)

            # 6) Generate splits relative to subsample
            if not count_only:
                split_dir = os.path.join(out_dir, 'splits', 'image_wise')
                create_splits(len(sub_idxs), train_fraction, seed, split_dir)

            print("Subsampling and rebundling complete.")

if __name__ == '__main__':
    for i in range(9):
        fr = 0.1 * (i+1)
        main(fr)
