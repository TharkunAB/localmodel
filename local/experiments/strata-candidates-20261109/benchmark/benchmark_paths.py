"""Keep candidate harness code separate from measured results."""
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[2]
HERE = ROOT / 'results/strata-candidates-20261109/harness'
