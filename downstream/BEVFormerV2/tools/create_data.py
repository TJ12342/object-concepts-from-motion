import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data_converter.nuscenes_converter import create_nuscenes_infos


def main():
    parser = argparse.ArgumentParser(
        description='Create temporal nuScenes annotations for BEVFormer V2.')
    parser.add_argument('--root-path', default='data/nuscenes')
    parser.add_argument('--out-dir', default='data/nuscenes')
    parser.add_argument('--canbus', default='data')
    parser.add_argument('--version', choices=['v1.0', 'v1.0-mini'], default='v1.0')
    parser.add_argument('--extra-tag', default='nuscenes')
    parser.add_argument('--max-sweeps', type=int, default=10)
    args = parser.parse_args()

    versions = (['v1.0-trainval', 'v1.0-test']
                if args.version == 'v1.0' else ['v1.0-mini'])
    for version in versions:
        create_nuscenes_infos(
            root_path=args.root_path,
            out_path=args.out_dir,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=version,
            max_sweeps=args.max_sweeps,
        )


if __name__ == '__main__':
    main()
