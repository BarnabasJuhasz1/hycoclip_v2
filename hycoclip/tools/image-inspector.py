#!/usr/bin/env python3
"""
Simple image inspector for Grounded WebDataset TAR shards.

Usage examples:
  python image-inspector.py --query "dog" --field text
  python image-inspector.py --query "window" --field box_text --substring

This script scans TAR files under the given folder and finds samples where
the `child.txt` (text) or any `parentNNN.txt` (box text) matches the query.
For each match it shows the child image and the matched parent box images.
Interactive controls: [n]ext, [p]rev, [s]ave, [q]uit.
"""
from __future__ import annotations

import argparse
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List
import sys

from PIL import Image


def scan_tar_for_samples(tar_path: Path) -> Dict[str, Dict[str, tarfile.TarInfo]]:
    samples = {}
    try:
        with tarfile.open(tar_path, 'r') as tf:
            for member in tf.getmembers():
                if member.isreg():
                    name = Path(member.name).name
                    if '.' not in name:
                        continue
                    key, ext = name.rsplit('.', 1)
                    samples.setdefault(key, {})[ext] = member
    except Exception as e:
        print(f"Failed to read {tar_path}: {e}")
    return samples


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


def collect_matches(tar_dir: Path, query: str, field: str, substring: bool = False):
    tar_paths = sorted(tar_dir.glob('*.tar'))
    matches = []  # list of tuples (tar_path, key, matched_parent_indices)
    for tar_path in tar_paths:
        with tarfile.open(tar_path, 'r') as tf:
            samples = {}
            for member in tf.getmembers():
                if not member.isreg():
                    continue
                name = Path(member.name).name
                if '.' not in name:
                    continue
                key, ext = name.rsplit('.', 1)
                samples.setdefault(key, {})[ext] = member

            for key, files in samples.items():
                # read child text
                child_text = None
                if 'txt' in files:
                    child_text = read_text_from_tar(tf, files['txt'])

                # gather parent texts
                parent_texts = []
                parent_indices = []
                for ext_name, member in files.items():
                    # expect parentNNN.txt and parentNNN.jpg naming
                    if ext_name.startswith('parent') and ext_name.endswith('.txt'):
                        pass

                # Alternative: iterate members keys to find parentNNN.txt
                for nm, member in files.items():
                    if nm.startswith('parent') and nm.endswith('.txt'):
                        idx = nm[len('parent'): -len('.txt')]
                        txt = read_text_from_tar(tf, member)
                        parent_texts.append((idx, txt))

                if field == 'text':
                    if child_text is None:
                        continue
                    match = (query in child_text) if substring else (query == child_text)
                    if match:
                        # no parent index selection here; show all parents
                        matches.append((tar_path, key, None))
                else:  # box_text
                    matched_indices = []
                    for idx, txt in parent_texts:
                        if substring:
                            if query in txt:
                                matched_indices.append(idx)
                        else:
                            if query == txt:
                                matched_indices.append(idx)
                    if matched_indices:
                        matches.append((tar_path, key, matched_indices))
    return matches


def show_sample(tar_path: Path, key: str, matched_parents, out_dir: Path):
    with tarfile.open(tar_path, 'r') as tf:
        # read child image and text
        members = {Path(m.name).name: m for m in tf.getmembers() if m.isreg()}
        # find the exact child image member: key + .jpg OR child.jpg pattern
        child_img_member = None
        for name, m in members.items():
            if name == f"{key}.jpg" or name == f"{key}.png" or name == f"{key}.jpeg":
                child_img_member = m
                break
        # fallback to child.jpg/child.png naming inside a grouping
        if child_img_member is None:
            for suffix in ('child.jpg', 'child.png', 'child.jpeg'):
                nm = f"{key}.{suffix.split('.')[-1]}"
                if nm in members:
                    child_img_member = members[nm]
                    break

        child_txt = None
        for name, m in members.items():
            if name == f"{key}.txt" or name.endswith('child.txt') and name.startswith(key):
                try:
                    child_txt = read_text_from_tar(tf, m)
                except Exception:
                    child_txt = None
                break

        child_img = None
        if child_img_member:
            child_img = read_image_from_tar(tf, child_img_member)

        # collect parent images and texts
        parent_imgs = []  # list of tuples (index, img, text)
        # search members for parentNNN.jpg and parentNNN.txt for this key
        prefix = f"{key}."  # members are often stored as <key>.parent000.jpg etc
        # better approach: find members whose name starts with key + '.' and then parent
        for name, m in members.items():
            if not name.startswith(f"{key}."):
                continue
            short = name[len(f"{key}."):]
            if short.startswith('parent') and short.endswith('.jpg'):
                idx = short[len('parent'): -len('.jpg')]
                img = read_image_from_tar(tf, m)
                # try read corresponding txt
                txt_member_name = f"{key}.parent{idx}.txt"
                txt = None
                if txt_member_name in members:
                    txt = read_text_from_tar(tf, members[txt_member_name])
                parent_imgs.append((idx, img, txt))

        # display
        print(f"Sample: {key} in {tar_path}")
        if child_txt:
            print(' Child text:', child_txt)
        if child_img:
            try:
                child_img.show()
            except Exception:
                pass
            # also save
            out_dir.mkdir(parents=True, exist_ok=True)
            child_path = out_dir / f"{key}_child.jpg"
            child_img.save(child_path)
            print(' Saved child image to', child_path)

        # show matched parents (if matched_parents is None, show all)
        for idx, img, txt in parent_imgs:
            show_this = False
            if matched_parents is None:
                show_this = True
            else:
                if idx in matched_parents:
                    show_this = True
            if not show_this:
                continue
            print(f" Parent {idx} text:", txt)
            if img:
                try:
                    img.show()
                except Exception:
                    pass
                ppath = out_dir / f"{key}_parent{idx}.jpg"
                img.save(ppath)
                print(' Saved parent image to', ppath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tar-dir', type=Path,
                        default=Path('/leonardo/home/userexternal/bjuhasz0/fast/barnabas/project_hierarchies/HyCoCLIP/datasets/train/GRIT/tar_fitted'),
                        help='Directory containing .tar shards')
    parser.add_argument('--query', required=True, help='Text to search for')
    parser.add_argument('--field', choices=['text', 'box_text'], default='text', help='Search in child text or parent box text')
    parser.add_argument('--substring', action='store_true', help='Match substring instead of exact')
    parser.add_argument('--out-dir', type=Path, default=Path('./image_inspector_output'), help='Where to save extracted images')

    args = parser.parse_args()

    if not args.tar_dir.exists():
        print('tar directory not found:', args.tar_dir)
        sys.exit(1)

    print('Scanning shards (this may take a while)...')
    matches = collect_matches(args.tar_dir, args.query, args.field, args.substring)
    print(f'Found {len(matches)} matching samples')
    if not matches:
        return

    idx = 0
    while 0 <= idx < len(matches):
        tar_path, key, matched_parents = matches[idx]
        show_sample(tar_path, key, matched_parents, args.out_dir)
        cmd = input(f'[{idx+1}/{len(matches)}] Enter command ([n]ext,[p]rev,[q]uit): ').strip().lower()
        if cmd in ('n', '', 'next'):
            idx += 1
        elif cmd in ('p', 'prev'):
            idx = max(0, idx-1)
        elif cmd in ('q', 'quit'):
            break
        else:
            print('Unknown command')


if __name__ == '__main__':
    main()
