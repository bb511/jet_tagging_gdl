"""Download the top-tagging reference data set and convert it for one constituent count.

The datamodule prepares a missing cache itself, but every process of a launch would then
convert the record at once: run this once per host or shared volume beforehand.

python scripts/data/prepare_top_tagging.py --nconstituents 100 [--cache-dir <dir>]
"""

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.top_tagging_datamodule import TopTaggingDataModule, read_meta

# The datamodule's default, cache_dir: ${paths.base_data_dir}/top_tagging.
DEFAULT_CACHE = REPO_ROOT / "data" / "top_tagging"


def main() -> None:
    """Prepare the cache and print where it is and what it holds."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nconstituents", type=int, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"the parent of raw/ and nconst_<N>/ (default {DEFAULT_CACHE})",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    datamodule = TopTaggingDataModule(
        cache_dir=str(args.cache_dir), nconstituents=args.nconstituents
    )
    datamodule.prepare_data()
    print(datamodule.cache_folder)
    print(read_meta(datamodule.cache_folder)["splits"])


if __name__ == "__main__":
    main()
