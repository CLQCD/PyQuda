import os
import sys
import numpy as np
import torch

test_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(test_dir, ".."))
from pyquda import core, mpi, field
from pyquda.field import Nc, Ns
from pyquda.utils import io

field.CUDA_BACKEND = "torch"

os.environ["QUDA_RESOURCE_PATH"] = ".cache"
mpi.init()

latt_size = [4, 4, 4, 8]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt

xi_0, nu = 2.464, 0.95
kappa = 0.115
coeff = 1.17
coeff_r, coeff_t = 0.91, 1.07

mass = 1 / (2 * kappa) - 4

dslash = core.getDslash(latt_size, mass, 1e-9, 1000, xi_0, nu, coeff_t, coeff_r, multigrid=False)
gauge = io.readQIOGauge(os.path.join(test_dir, "weak_field.lime"))


dslash.loadGauge(gauge)

propagator = core.invert(dslash, "point", [0, 0, 0, 0])

dslash.destroy()

propagator_chroma = (
    torch.from_numpy(np.fromfile("pt_prop_1", ">c16", offset=8).astype("<c16")).to("cuda").reshape(Vol, Ns, Ns, Nc, Nc)
)
print(
    torch.linalg.norm(propagator.data.reshape(Vol, Ns, Ns, Nc, Nc) - propagator_chroma.transpose(1, 2).transpose(3, 4))
)
