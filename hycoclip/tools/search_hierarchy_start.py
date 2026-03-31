#!/usr/bin/env python3
"""
Script to search for hierarchies starting with a given text in JSON files.

Usage:
  python search_hierarchy_start.py --json-dir /path/to/json_pairwise --query "A nurse"
"""
import argparse
import json
from pathlib import Path


def search_hierarchies(json_dir: Path, query: str):
    matches = []
    for json_file in sorted(json_dir.glob('*.json')):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        for key, parents in data.items():
            for parent_key, parent_data in parents.items():
                for version in ('original', 'proposed'):
                    if version in parent_data and 'hierarchy' in parent_data[version]:
                        hierarchy = parent_data[version]['hierarchy']
                        if hierarchy and isinstance(hierarchy, list) and hierarchy[0].strip().lower() == query.strip().lower():
                            matches.append((json_file, key, parent_key, version, hierarchy))
    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-dir', type=Path, required=True, help='Directory containing JSON files')
    parser.add_argument('--query', type=str, required=True, help='Text to match at the start of hierarchy')
    args = parser.parse_args()

    results = search_hierarchies(args.json_dir, args.query)
    if not results:
        print('No matches found.')
    else:
        print(f'Found {len(results)} matches:')
        for json_file, key, parent_key, version, hierarchy in results:
            print(f'- File: {json_file.name}, Key: {key}, Parent: {parent_key}, Version: {version}')
            print(f'  Hierarchy: {hierarchy}')


if __name__ == '__main__':
    main()
