from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(root: Path):
    images = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        images.extend(root.rglob(pattern))
    return sorted(set(images))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check exact duplicate images across TRAIN and TEST")
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    root = Path(args.data_dir)
    train = collect(root / "TRAIN")
    test = collect(root / "TEST")

    train_hashes = defaultdict(list)
    test_hashes = defaultdict(list)
    for path in train:
        train_hashes[md5(path)].append(path)
    for path in test:
        test_hashes[md5(path)].append(path)

    overlap = sorted(set(train_hashes) & set(test_hashes))
    internal_train = {h: p for h, p in train_hashes.items() if len(p) > 1}
    internal_test = {h: p for h, p in test_hashes.items() if len(p) > 1}

    print(f"TRAIN images: {len(train)}")
    print(f"TEST images: {len(test)}")
    print(f"Exact duplicate groups within TRAIN: {len(internal_train)}")
    print(f"Exact duplicate groups within TEST: {len(internal_test)}")
    print(f"Exact hashes shared between TRAIN and TEST: {len(overlap)}")

    if overlap:
        print("\nExamples of TRAIN/TEST overlap:")
        for h in overlap[:20]:
            print(f"Hash: {h}")
            print("  TRAIN:")
            for p in train_hashes[h]:
                print(f"    {p}")
            print("  TEST:")
            for p in test_hashes[h]:
                print(f"    {p}")


if __name__ == "__main__":
    main()
