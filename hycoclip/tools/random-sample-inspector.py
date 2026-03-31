#!/usr/bin/env python3
"""
Random sample inspector for Grounded WebDataset TAR shards.

This script picks a random TAR file from the given directory, randomly selects
up to 20 samples from it, and for each sample:
- Logs all details (caption text, parent texts) to a log file
- Saves the caption image and all parent box images to a subfolder

Usage example:
  python random-sample-inspector.py --tar-dir /path/to/tars --out-dir ./output
"""
from __future__ import annotations

import argparse
import random
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List
import sys

from PIL import Image


def get_random_tar(tar_dir: Path) -> Path:
    tar_paths = list(tar_dir.glob('*.tar'))
    if not tar_paths:
        raise ValueError(f"No .tar files found in {tar_dir}")
    return random.choice(tar_paths)


def get_all_sample_keys(tar_path: Path) -> List[str]:
    samples = set()
    try:
        with tarfile.open(tar_path, 'r') as tf:
            for member in tf.getmembers():
                if member.isreg():
                    name = Path(member.name).name
                    if '.' not in name:
                        continue
                    key, _ = name.rsplit('.', 1)
                    samples.add(key)
    except Exception as e:
        print(f"Failed to read {tar_path}: {e}")
        sys.exit(1)
    return list(samples)


def read_text_from_tar(tf: tarfile.TarFile, member: tarfile.TarInfo) -> str:
    f = tf.extractfile(member)
    if f is None:
        return ""
    return f.read().decode('utf-8', errors='replace').strip()


def read_image_from_tar(tf: tarfile.TarFile, member: tarfile.TarInfo) -> Image.Image | None:
    f = tf.extractfile(member)
    if f is None:
        return None
    try:
        return Image.open(BytesIO(f.read())).convert('RGB')
    except Exception:
        return None

def process_sample(tar_path: Path, key: str, out_dir: Path, log_file: Path):
    sample_dir = out_dir / key
    sample_dir.mkdir(parents=True, exist_ok=True)

    shortkey = key.split('.')[0]

    # Load hierarchy info from json_pairwise
    import json
    json_dir = Path('/leonardo/home/userexternal/bjuhasz0/fast/barnabas/project_hierarchies/HyCoCLIP/datasets/HyperHi5/v1/json_pairwise')
    hierarchy_info = {}
    json_path = json_dir / f"{tar_path.stem}.json"
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as jf:
                hierarchy_info = json.load(jf)
        except Exception:
            print("hierarchy info set to empty!")
            hierarchy_info = {}
    else:
        print(f"No json path under {json_path}!")

    with tarfile.open(tar_path, 'r') as tf:
        members = {Path(m.name).name: m for m in tf.getmembers() if m.isreg()}

        # Read caption text
        caption_txt = None
        for name, m in members.items():
            if name == f"{shortkey}.txt" or (name.endswith('.txt') and 'child' in name and name.startswith(shortkey)):
                try:
                    caption_txt = read_text_from_tar(tf, m)
                except Exception:
                    caption_txt = None
                break

        # Read caption image
        caption_img = None
        for name, m in members.items():
            if name in [f"{key}.jpg", f"{key}.png", f"{key}.jpeg"] or (name.endswith(('.jpg', '.png', '.jpeg')) and 'caption' in name and name.startswith(key)):
                caption_img = read_image_from_tar(tf, m)
                if caption_img:
                    caption_path = sample_dir / f"caption_{name}.jpg"
                    caption_img.save(caption_path)
                break

        # Collect parent images and texts
        parents = {}  # idx -> (img, txt, hierarchy)
        for name, m in members.items():
            if not name.startswith(f"{shortkey}."):
                continue
            short = name[len(f"{shortkey}."):]
            print(f"Processing member: {short} whose short key is: {shortkey}")
            if short.startswith('parent'): #and short.endswith(('.jpg', '.png', '.jpeg')):
                print(f"Found parent member: {short}")
                idx = short[len('parent'):].rsplit('.', 1)[0]
                img = read_image_from_tar(tf, m)
                # Try to read corresponding txt
                txt_name = f"{shortkey}.parent{idx}.txt"
                txt = None
                if txt_name in members:
                    txt = read_text_from_tar(tf, members[txt_name])
                # Try to get hierarchy from json_pairwise
                hierarchy = None
                if hierarchy_info:
                    # parent_key = f"parent{idx}"
                    # print(f"Found hierarchy info for {shortkey} and parent{idx} in json!")
                    if shortkey in hierarchy_info and f"parent{idx}" in hierarchy_info[shortkey]:
                        hdata = hierarchy_info[shortkey][f"parent{idx}"]
                        print(f"hierarchy info found for parent{idx} in json!")
                        # Prefer 'original' then 'proposed'
                        if 'original' in hdata and 'hierarchy' in hdata['original']:
                            hierarchy = hdata['original']['hierarchy']
                        elif 'proposed' in hdata and 'hierarchy' in hdata['proposed']:
                            hierarchy = hdata['proposed']['hierarchy']
                    else:
                        print(f"No hierarchy info for parent{idx} in json")
                else:
                    print("No hierarchy info available at all...!")
                parents[idx] = (img, txt, hierarchy)
                if img:
                    img_path = sample_dir / f"parent{idx}.jpg"
                    img.save(img_path)

        # Log details in hierarchical indentation
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"caption: {key}\n")
            f.write(f"  Text: {caption_txt or 'N/A'}\n")
            for idx in sorted(parents.keys(), key=int):
                img, txt, hierarchy = parents[idx]
                f.write(f"  Parent {idx}: {txt or 'N/A'}\n")
                if hierarchy:
                    f.write(f"    Hierarchy:\n")
                    for h in hierarchy:
                        f.write(f"      - {h}\n")
            f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tar-dir', type=Path,
                        default=Path('/leonardo/home/userexternal/bjuhasz0/fast/barnabas/project_hierarchies/HyCoCLIP/datasets/train/GRIT/tar_fitted'),
                        help='Directory containing .tar shards')
    parser.add_argument('--out-dir', type=Path, default=Path('./random_sample_output'),
                        help='Output directory for images and log')
    parser.add_argument('--num-samples', type=int, default=20,
                        help='Number of random samples to select (default: 20)')
    parser.add_argument('--log-file', type=Path, default=None,
                        help='Path to log file (default: out_dir/sample_details.log)')

    args = parser.parse_args()

    if not args.tar_dir.exists():
        print('Tar directory not found:', args.tar_dir)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.log_file is None:
        args.log_file = args.out_dir / 'sample_details.log'

    print('Picking random tar file...')
    tar_path = get_random_tar(args.tar_dir)
    print(f'Selected: {tar_path}')

    print('Collecting sample keys...')
    all_keys = get_all_sample_keys(tar_path)
    print(f'Found {len(all_keys)} samples')

    if not all_keys:
        print('No samples found in the tar file')
        sys.exit(1)

    num_to_select = min(args.num_samples, len(all_keys))
    selected_keys = random.sample(all_keys, num_to_select)
    print(f'Selected {num_to_select} random samples')

    # Clear log file
    with open(args.log_file, 'w', encoding='utf-8') as f:
        f.write(f"Random samples from {tar_path}\n\n")

    for key in selected_keys:
        print(f'Processing sample: {key}')
        process_sample(tar_path, key, args.out_dir, args.log_file)

    print(f'Completed. Images saved to {args.out_dir}, details logged to {args.log_file}')


if __name__ == '__main__':
    main()