import os

os.environ.setdefault("MPLBACKEND", "Agg")
# Force Agg before any matplotlib import during tests.
import matplotlib
matplotlib.use("Agg", force=True)
