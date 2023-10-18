import os
import sys

import cupy as cp

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))
from pyquda import core, mpi
from pyquda.utils import io

os.environ["QUDA_RESOURCE_PATH"] = ".cache"
mpi.init()

latt_size = [4, 4, 4, 8]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt

gauge = io.readQIOGauge(os.path.join(test_dir, "weak_field.lime"))


core.hyp_smear(latt_size, gauge, 1, 0.75, 0.6, 0.3)
# gauge.setAntiPeroidicT()  # for fermion smearing

gauge_chroma = io.readQIOGauge("hyp.lime")
print(cp.linalg.norm(gauge.data - gauge_chroma.data))
