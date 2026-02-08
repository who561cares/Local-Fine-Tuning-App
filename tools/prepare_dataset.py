#!/usr/bin/env python3
"""
Simple dataset preparation helper.

Supports csv, json, jsonl for tabular/text and copies/validates image folders.
Produces a small, predictable layout under the output directory:
  out/images/  (jpg/png copied)
  out/labels.csv (for image tasks)
  out/text.csv (for text tasks)

This is a helper for the prototype — not a production pipeline.
"""
import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import pandas as pd


def ensure_out(dirpath: Path):
    dirpath.mkdir(parents=True, exist_ok=True)


def process_tabular(input_path: Path, out_path: Path):
    ext = input_path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(input_path)
    elif ext == ".json":
        df = pd.read_json(input_path)
    elif ext == ".jsonl":
        df = pd.read_json(input_path, lines=True)
    else:
        raise ValueError("Unsupported tabular extension: %s" % ext)

    out_file = out_path / "text.csv"
    df.to_csv(out_file, index=False)
    print("Wrote:", out_file)


def process_images(input_path: Path, out_path: Path):
    images_out = out_path / "images"
    ensure_out(images_out)
    labels = []
    if input_path.is_dir():
        # expect structure input_path/<class>/*.jpg
        for class_dir in sorted(input_path.iterdir()):
            if not class_dir.is_dir():
                continue
            for img in class_dir.glob("*.*"):
                if img.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                    continue
                dest = images_out / img.name
                shutil.copy2(img, dest)
                labels.append((dest.name, class_dir.name))
    else:
        raise ValueError("For images, provide a folder with class subfolders")

    labels_file = out_path / "labels.csv"
    with open(labels_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(labels)

    print("Copied images ->", images_out)
    print("Wrote labels ->", labels_file)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--type", choices=["tabular", "images"], required=True)
    args = p.parse_args()

    input_path = Path(args.input)
    out_path = Path(args.out)
    ensure_out(out_path)

    if args.type == "tabular":
        process_tabular(input_path, out_path)
    else:
        process_images(input_path, out_path)


if __name__ == "__main__":
    main()
