#!/usr/bin/env python3
"""
Fixed version – ensures **total sample count** matches train / val splits
========================================================
主な修正点
------------
1. **総サンプル数(`n_samples`)の定義を 1 prefix の行数で一貫させる**  
   `merge_subsamples` で prefix ごとに加算していたため倍計上されていた。
2. **各 prefix の行数が同一か assert**  
   不一致の場合は即エラー。
3. **`integrate_and_subsample`** 内で prefix ごとの行数整合性もチェック。

使い方・目的は元スクリプトと同じです。
"""
import os, re, shutil, json, argparse, numpy as np
from math import ceil

# ---------- Utility ----------

def copy_base_structure(src, dst):
    def ignore_root(dirpath, names):
        if os.path.abspath(dirpath) == os.path.abspath(src):
            return ['tensors','splits','dataset_generation.json',
                    'object_category_map.json','pose_category_map.json']
        return []
    shutil.copytree(src, dst, ignore=ignore_root)
    os.makedirs(os.path.join(dst,'tensors'), exist_ok=True)
    os.makedirs(os.path.join(dst,'splits','image_wise'), exist_ok=True)


def load_category_maps(bases):
    obj_keys, pose_keys = set(), set()
    for d in bases:
        for fn in ['object_category_map.json','pose_category_map.json']:
            p = os.path.join(d, fn)
            if os.path.isfile(p):
                m=json.load(open(p))
                (obj_keys if 'object' in fn else pose_keys).update(m.keys())
    obj_map={k:i for i,k in enumerate(sorted(obj_keys))}
    pose_map={k:i for i,k in enumerate(sorted(pose_keys))}
    return obj_map, pose_map


def merge_category_maps(bases, out):
    obj_map, pose_map = load_category_maps(bases)
    json.dump(obj_map,  open(os.path.join(out,'object_category_map.json'),'w'), indent=2)
    json.dump(pose_map, open(os.path.join(out,'pose_category_map.json'),'w'),  indent=2)
    return obj_map, pose_map


def update_generation_json(src_base, out):
    src=os.path.join(src_base,'dataset_generation.json')
    if not os.path.isfile(src):
        return
    gen=json.load(open(src))
    gen['object_category_map']='object_category_map.json'
    gen['pose_category_map']='pose_category_map.json'
    json.dump(gen, open(os.path.join(out,'dataset_generation.json'),'w'), indent=2)

# ---------- Data helpers ----------

def gather_prefix_bundles(tdir):
    pat=re.compile(r'^(?P<prefix>.+)_(?P<idx>\d{5})\.npz$')
    d={}
    for fn in os.listdir(tdir):
        m=pat.match(fn)
        if not m: continue
        p=m.group('prefix'); idx=int(m.group('idx'))
        d.setdefault(p,[]).append((idx, os.path.join(tdir,fn)))
    for p in d:
        d[p].sort(key=lambda x:x[0])
    return d


def integrate_and_subsample(bundles, frac, seed, obj_map, pose_map):
    """Return (subsamples_dict, n_sub)"""
    integrated={}
    n_total=None
    # integrate
    for pre, files in bundles.items():
        fparts={}
        for _,path in files:
            with np.load(path) as data:
                for k in data.files:
                    fparts.setdefault(k,[]).append(data[k])
        integrated[pre]={k:np.concatenate(v,0) for k,v in fparts.items()}
        rows=next(iter(integrated[pre].values())).shape[0]
        if n_total is None:
            n_total=rows
        else:
            assert rows==n_total, f"Prefix '{pre}' rows {rows} != {n_total}"
    # subsample idx
    rng=np.random.RandomState(seed)
    n_sub=int(n_total*frac)
    idxs=np.sort(rng.choice(n_total, n_sub, replace=False))
    # extract & remap
    subs={}
    for pre,fd in integrated.items():
        sub={}
        for k,arr in fd.items():
            sel=arr[idxs]
            if k=='obj_labels': sel=np.vectorize(lambda x:obj_map[str(x)])(sel)
            if k=='pose_labels':sel=np.vectorize(lambda x:pose_map[str(x)])(sel)
            sub[k]=sel
        subs[pre]=sub
    return subs, n_sub


def merge_subsamples(sub_lists):
    merged={}
    for subs,_ in sub_lists:
        for pre,fd in subs.items():
            for k,a in fd.items():
                merged.setdefault(pre,{}).setdefault(k,[]).append(a)
    # concat & row consistency
    n_samples=None
    for pre,fd in merged.items():
        for k,parts in fd.items():
            fd[k]=np.concatenate(parts,0)
        rows=next(iter(fd.values())).shape[0]
        if n_samples is None:
            n_samples=rows
        else:
            assert rows==n_samples, f"Row mismatch after merge in prefix '{pre}'"
    return merged, n_samples

# ---------- Output ----------

def bundle_to_npz(merged, out_dir, bsize):
    os.makedirs(out_dir, exist_ok=True)
    for pre,fd in merged.items():
        rows=next(iter(fd.values())).shape[0]
        n_b=ceil(rows/bsize)
        for b in range(n_b):
            s,e=b*bsize, min((b+1)*bsize, rows)
            out={k:v[s:e] for k,v in fd.items()}
            np.savez_compressed(os.path.join(out_dir,f"{pre}_{b:05d}.npz"), **out)


def create_splits(n, train_frac, seed, split_dir):
    rng=np.random.RandomState(seed)
    idx=np.arange(n); rng.shuffle(idx)
    n_train=int(n*train_frac)
    train=np.sort(idx[:n_train]); val=np.sort(idx[n_train:])
    os.makedirs(split_dir, exist_ok=True)
    np.savez_compressed(os.path.join(split_dir,'train_indices.npz'), train)
    np.savez_compressed(os.path.join(split_dir,'val_indices.npz'), val)
    json.dump({'num_train':train.size,'num_val':val.size,'total':n},
              open(os.path.join(split_dir,'metadata_subsample.json'),'w'), indent=2)
    json.dump({'field_name':'index','train_pct':train_frac},
              open(os.path.join(split_dir,'metadata.json'),'w'), indent=2)

# ---------- Main ----------

def main(args):
    if len(args.base_dirs)!=len(args.fractions):
        raise ValueError("--fractions length must match --base-dirs length")
    copy_base_structure(args.base_dirs[0], args.output_dir)
    obj_map, pose_map = merge_category_maps(args.base_dirs, args.output_dir)
    update_generation_json(args.base_dirs[0], args.output_dir)

    subs_list=[]
    for base,frac in zip(args.base_dirs,args.fractions):
        bundles=gather_prefix_bundles(os.path.join(base,'tensors'))
        subs,n=integrate_and_subsample(bundles, frac, args.seed, obj_map, pose_map)
        subs_list.append((subs,n))
    merged, n_samples = merge_subsamples(subs_list)
    print(f"[INFO] merged dataset rows = {n_samples}")
    bundle_to_npz(merged, os.path.join(args.output_dir,'tensors'), args.bundle_size)
    create_splits(n_samples, args.train_fraction, args.seed,
                  os.path.join(args.output_dir,'splits','image_wise'))
    print("Merge complete – splits and bundles written.")

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--base-dirs', nargs='+', required=True)
    ap.add_argument('--fractions', nargs='+', type=float, required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--train-fraction', type=float, default=0.8)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--bundle-size', type=int, default=1000)
    main(ap.parse_args())
