"""Rebuild the exact train/validation/test split from Microsoft's original zip file.

Download kagglecatsanddogs_5340.zip (about 790 MB) from
https://www.microsoft.com/en-us/download/details.aspx?id=54765, then run from the project folder:
    python -m src.prepare_data path/to/kagglecatsanddogs_5340.zip

splits/split_manifest.csv lists all 25,000 images in the zip. Each one has the split it
belongs to, or the reason it was left out (259 images: files that aren't really JPEGs,
unreadable files and duplicates). Following the list gives everyone the same split, so
results can be compared fairly.
"""
import argparse
import collections
import csv
import zipfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Split the cats and dogs dataset.")
    parser.add_argument("zip", type=Path, help="kagglecatsanddogs_5340.zip from Microsoft")
    parser.add_argument("--out", type=Path, default=Path("data/catsanddogs"))
    parser.add_argument("--manifest", type=Path, default=Path("splits/split_manifest.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))

    kept = [r for r in rows if r["split"] != "excluded"]
    counts = collections.Counter()
    with zipfile.ZipFile(args.zip) as z:
        missing = [r["zip_path"] for r in kept if r["zip_path"] not in z.NameToInfo]
        if missing:
            raise SystemExit(f"{len(missing)} images from the manifest aren't in the zip, "
                             f"for example {missing[0]}. Is this the right file?")
        for r in kept:
            dest = args.out / r["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(r["zip_path"]))
            counts[(r["split"], r["class"])] += 1

    for (split, cls), n in sorted(counts.items()):
        print(f"{split:<10} {cls}: {n:,}")
    print(f"Kept {len(kept):,} images in {args.out}, left out {len(rows) - len(kept)}")


if __name__ == "__main__":
    main()
