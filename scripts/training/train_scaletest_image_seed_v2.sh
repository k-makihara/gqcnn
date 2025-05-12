#!/bin/bash

base_dir="/home/deepstation/Downloads/scale_test"
config="cfg/train_dex-net_2.0.yaml"

# 手法名 (表示用) → ディレクトリ＆ファイル名プレフィックス のマッピング
declare -A methods=(
  ["Dexnet-150"]="dexnet_150"
  ["EGAD-150"]="egad_150"
  ["GFDB-006-150"]="gfdb_006_150_v2"
  ["GFDB-030-150"]="gfdb_030_150_v2"
  ["GFDB-070-150"]="gfdb_070_150_v2"
  ["GFDB-006-030-070-150"]="gfdb_006_030_070_150"
  ["Primitive-150"]="primitive"
)

# 010,020,...,090 の 9 種類
percentages=(010 020 030 040 050 060 070 080 090)

for seed in {1..5}; do
  for label in "${!methods[@]}"; do
    prefix=${methods[$label]}

    for p in "${percentages[@]}"; do
      # "010"→10 のように先頭 0 を外す
      num=$((10#$p))

      path="$base_dir/${prefix}_${p}_seed${seed}"
      name="${label}-${num}-seed${seed}"

      # 実際に動かすときは↓をアンコメント
      python tools/train.py "$path" \
          --config_filename "$config" \
          --name "$name"

      #echo "path: $path"
      #echo "config: $config"
      #echo "name: $name"
    done
  done
done
