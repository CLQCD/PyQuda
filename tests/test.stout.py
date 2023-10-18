import os
import sys

import cupy as cp

test_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(test_dir, ".."))
from pyquda import core, mpi
from pyquda.utils import io

os.environ["QUDA_RESOURCE_PATH"] = ".cache"
mpi.init()

latt_size = [4, 4, 4, 8]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt

gauge = io.readQIOGauge(os.path.join(test_dir, "weak_field.lime"))


core.stout_smear(latt_size, gauge, 1, 0.241)
# gauge.setAntiPeroidicT()  # for fermion smearing

gauge_chroma = io.readQIOGauge("stout.lime")
print(cp.linalg.norm(gauge.data - gauge_chroma.data))
