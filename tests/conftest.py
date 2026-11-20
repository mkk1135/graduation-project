"""The scripts are run from the project folder (decide.py opens results/results.db relative to it), and so are the tests."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
