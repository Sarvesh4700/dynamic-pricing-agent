"""Makes 'src' and 'tests' importable when pytest is invoked from anywhere."""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
