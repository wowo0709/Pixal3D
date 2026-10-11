"""Native training with explicit GPU assignment and stage-local runtime hooks."""
import argparse
from pathlib import Path
import runpy
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

if __name__ == '__main__':
    if not {'-h', '--help'}.intersection(sys.argv[1:]):
        from experiments.multiview_finetune.paths import assigned_gpu_count
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--num_gpus', type=int)
        launch, _ = parser.parse_known_args()
        count = assigned_gpu_count(launch.num_gpus)
        if launch.num_gpus is None:
            sys.argv.extend(['--num_gpus', str(count)])
        sys.argv.append('--multiview_runtime')
    sys.argv[0] = str(REPO / 'train.py')
    runpy.run_path(str(REPO / 'train.py'), run_name='__main__')
