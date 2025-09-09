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


def ffstratified_subsample_indices(integrated, label_key, overall_fraction, pos_ratio, seed):
    """
    Positive/Negative をラベル配列（0 or 1）から取り出し、
    全体の overall_fraction に応じた数 n_sub を計算、
    そのうち pos_ratio 分だけ Positive、残りを Negative からサンプリング。
    """
    labels = integrated[label_key].ravel()
    N = labels.shape[0]
    n_sub = int(N * overall_fraction)
    n_pos = int(n_sub * pos_ratio)
    n_neg = n_sub - n_pos

    threshhold = 0.002

    pos_idxs = np.where(labels >= threshhold)[0]
    neg_idxs = np.where(labels < threshhold)[0]
    if n_pos > len(pos_idxs) or n_neg > len(neg_idxs):
        raise ValueError(f"要求されたサンプル数がクラス数を超えています: pos {n_pos}/{len(pos_idxs)}, neg {n_neg}/{len(neg_idxs)}")

    rng = np.random.RandomState(seed)
    sel_pos = rng.choice(pos_idxs, size=n_pos, replace=False)
    sel_neg = rng.choice(neg_idxs, size=n_neg, replace=False)
    subs = np.sort(np.concatenate([sel_pos, sel_neg]))
    print(f"Sampling total={n_sub} (pos={n_pos}, neg={n_neg}) out of {N}")
    return subs

def stratified_subsample_indices(
    integrated, label_key,
    overall_fraction, pos_ratio, seed,
    oversample_minority=False
):
    """
    Positive/Negative のインデックスを層別抽出します。
    - overall_fraction: 全体サンプル中の取得割合
    - pos_ratio: そのうちPositiveに割り当てる比率
    - oversample_minority: Trueなら足りないクラスは複製（replace=True）、
                           Falseなら可能な限り抽出して残りはもう一方から補填
    """
    labels = integrated[label_key].ravel()
    N = labels.shape[0]
    threshhold = 0.002

    pos_idxs = np.where(labels >= threshhold)[0]
    neg_idxs = np.where(labels < threshhold)[0]
    print(f"positive:{len(pos_idxs)}")
    print(f"negative:{len(neg_idxs)}")

    if len(pos_idxs) >= len(neg_idxs):
        n_sub = int(len(neg_idxs) * overall_fraction)
    else:
        n_sub = int(len(pos_idxs) * overall_fraction)
    
    #n_sub = int(N * overall_fraction)
    n_pos = int(n_sub * pos_ratio)
    n_neg = n_sub - n_pos

    rng = np.random.RandomState(seed)

    # Positive 抽出
    if len(pos_idxs) >= n_pos:
        sel_pos = rng.choice(pos_idxs, size=n_pos, replace=False)
    else:
        if oversample_minority:
            # 足りない分は複製して補う
            sel_pos = rng.choice(pos_idxs, size=n_pos, replace=True)
            print(f"Warning: Positive が不足({len(pos_idxs)}<{n_pos})。オーバーサンプリングします。")
        else:
            # 可能な限り／残りを Negative から補填
            sel_pos = pos_idxs.copy()
            n_neg += (n_pos - len(pos_idxs))
            n_pos = len(pos_idxs)
            print(f"Warning: Positive が不足({len(pos_idxs)}<{n_pos})。残り{n_pos - len(pos_idxs)}をNegativeから補填します。")

    # Negative 抽出
    if len(neg_idxs) >= n_neg:
        sel_neg = rng.choice(neg_idxs, size=n_neg, replace=False)
    else:
        if oversample_minority:
            sel_neg = rng.choice(neg_idxs, size=n_neg, replace=True)
            print(f"Warning: Negative が不足({len(neg_idxs)}<{n_neg})。オーバーサンプリングします。")
        else:
            sel_neg = neg_idxs.copy()
            # もしまだ総数が不足すればPositiveからも補填
            extra = n_neg - len(neg_idxs)
            if extra > 0:
                add_from_pos = rng.choice(pos_idxs, size=extra, replace=False)
                sel_neg = np.concatenate([sel_neg, add_from_pos])
                print(f"Warning: Negative が不足。Positiveから{extra}サンプルを追加補填します。")

    subs = np.sort(np.concatenate([sel_pos, sel_neg]))
    print(f"Sampling total={len(subs)} (pos={n_pos}, neg={n_neg}) out of {N}")
    return subs


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
    #base_dirs       = ["/home/deepstation/Downloads/scale_test/dexnet_150", "/home/deepstation/Downloads/scale_test/egad_150", "/home/deepstation/Downloads/scale_test/gfdb_006_150_v2", "/home/deepstation/Downloads/scale_test/gfdb_030_150_v2", "/home/deepstation/Downloads/scale_test/gfdb_070_150_v2", "/home/deepstation/Downloads/scale_test/gfdb_006_030_070_150", "/home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/primitive"]
    base_dirs       = ["/home/deepstation/Downloads/scale_test/dexnet_150"]
    fraction       = fr      # Subsample rate
    train_fraction = 0.8      # Train split ratio
    seeds           = [123]      # RNG seed for reproducibility
    #seeds           = [1,2,3,4,5]      # RNG seed for reproducibility
    bundle_size    = 1000     # Samples per NPZ bundle
    count_only = False
    label_key        = 'robust_ferrari_canny'  # NPZ内でラベルを持つprefix名
    pos_ratio        = 0.1
    # ──────────────────────────

    for base_dir in base_dirs:
        for seed in seeds:
            out_dir        = f"{base_dir}_{int(fraction*100):03}_seed{seed}_pos{int(pos_ratio*100):03}"
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
            print(integrated['robust_ferrari_canny'])
            # 4) Subsample positions
            sub_idxs = stratified_subsample_indices(
                integrated,
                label_key,       # 例: 'labels'
                fr,
                pos_ratio,
                seed
            )

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
    # for i in range(9):
    #     fr = 0.1 * (i+1)
    #     main(fr)
    main(1.0)
