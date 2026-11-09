"""Select separate evidence directories for the two official releases."""
import os
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[2]
BASE = ROOT / 'results/strata-0213-20261109'
RELEASE = os.environ.get('STRATA_BENCH_RELEASE', '')
VERSIONS = {'v0.2.12': '0.1.58', 'v0.2.13': '0.1.59'}
if RELEASE not in VERSIONS:
    raise RuntimeError('Set STRATA_BENCH_RELEASE to v0.2.12 or v0.2.13.')
HERE = BASE / RELEASE
