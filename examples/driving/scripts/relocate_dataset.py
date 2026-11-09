#!/usr/bin/env python3
"""Rewrite the absolute paths stored in the driving split CSVs.

sequence_data_module.py reads image / state paths directly from
<data_src>/{train,val,test}_split.csv. Those CSVs contain absolute paths from the
machine on which they were generated. After moving or downloading the dataset,
run this once to point them at the new location:

    python scripts/relocate_dataset.py --data-src /path/to/data

A backup of each CSV is written as <name>.csv.bak (only if no backup exists).
"""
import argparse
import os
import shutil

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-src", required=True, help="dataset root containing images/, state/ and *_split.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    data_src = os.path.abspath(args.data_src)

    for split in ("train", "val", "test"):
        path = os.path.join(data_src, f"{split}_split.csv")
        df = pd.read_csv(path, index_col=0, dtype=str, keep_default_na=False)
        first = df["image"].iloc[0]
        marker = os.sep + "images" + os.sep
        if marker not in first:
            raise ValueError(f"cannot find '{marker}' in {first!r}")
        old_root = first.split(marker)[0]
        changed = 0
        for col in ("image", "control", "obstacle"):
            mask = df[col].astype(str).str.startswith(old_root)
            df.loc[mask, col] = df.loc[mask, col].astype(str).str.replace(old_root, data_src, n=1, regex=False)
            changed += int(mask.sum())
        print(f"{split}: {old_root} -> {data_src} ({changed} cells)")
        missing = [p for p in df["image"].iloc[[0, len(df) // 2, -1]] if not os.path.exists(p)]
        if missing:
            print(f"  WARNING: sample paths do not exist after rewrite: {missing}")
        if not args.dry_run:
            bak = path + ".bak"
            if not os.path.exists(bak):
                shutil.copy2(path, bak)
            df.to_csv(path)


if __name__ == "__main__":
    main()
