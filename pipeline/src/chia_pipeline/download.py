"""Download and extract the raw CHIA corpus (Kury et al. 2020) from figshare."""

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

FIGSHARE_URL = "https://ndownloader.figshare.com/files/21728853"  # chia_without_scope.zip

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PIPELINE_ROOT.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def download_and_extract(raw_dir: Path = DEFAULT_RAW_DIR, force: bool = False) -> Path:
    extract_dir = raw_dir / "chia_without_scope"
    if extract_dir.exists() and not force:
        print(f"Already present: {extract_dir}")
        return extract_dir

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "chia_without_scope.zip"
    print(f"Downloading {FIGSHARE_URL} ...")
    urlretrieve(FIGSHARE_URL, zip_path)

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    print(f"Extracted to {extract_dir}")
    return extract_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_and_extract(args.raw_dir, args.force)


if __name__ == "__main__":
    main()
