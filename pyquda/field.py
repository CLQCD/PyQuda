from typing import List, Literal

import numpy

from .pointer import ndarrayPointer, Pointer, Pointers


class LatticeInfo:
    Ns: int = 4
    Nc: int = 3
    Nd: int = 4

    def __init__(
        self,
        latt_size: List[int],
        t_boundary: Literal[1, -1] = 1,
        anisotropy: float = 1.0,
    ) -> None:
        from . import getMPIComm, getMPISize, getMPIRank, getGridSize, getGridCoord

        if getMPIComm() is None:
            raise RuntimeError("pyquda.init() must be called before contructing LatticeInfo")

        self.mpi_size = getMPISize()
        self.mpi_rank = getMPIRank()
        self.grid_size = getGridSize()
        self.grid_coord = getGridCoord()

        Gx, Gy, Gz, Gt = self.grid_size
        gx, gy, gz, gt = self.grid_coord
        Lx, Ly, Lz, Lt = latt_size

        assert (
            Lx % (2 * Gx) == 0 and Ly % (2 * Gy) == 0 and Lz % (2 * Gz) == 0 and Lt % Gt == 0
        ), "Necessary for consistant even-odd preconditioning"
        self.Gx, self.Gy, self.Gz, self.Gt = Gx, Gy, Gz, Gt
        self.gx, self.gy, self.gz, self.gt = gx, gy, gz, gt
        self.global_size = [Lx, Ly, Lz, Lt]
        self.global_volume = Lx * Ly * Lz * Lt
        Lx, Ly, Lz, Lt = Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt
        self.Lx, self.Ly, self.Lz, self.Lt = Lx, Ly, Lz, Lt
        self.size = [Lx, Ly, Lz, Lt]
        self.volume = Lx * Ly * Lz * Lt
        self.size_cb2 = [Lx // 2, Ly, Lz, Lt]
        self.volume_cb2 = Lx * Ly * Lz * Lt // 2
        self.ga_pad = Lx * Ly * Lz * Lt // min(Lx, Ly, Lz, Lt) // 2

        self.t_boundary = t_boundary
        self.anisotropy = anisotropy


Ns, Nc, Nd = LatticeInfo.Ns, LatticeInfo.Nc, LatticeInfo.Nd


def lexico(data: numpy.ndarray, axes: List[int], dtype=None):
    shape = data.shape
    Np, Lt, Lz, Ly, Lx = [shape[axis] for axis in axes]
    assert Np == 2, "There must be 2 parities."
    Lx *= 2
    Npre = int(numpy.prod(shape[: axes[0]]))
    Nsuf = int(numpy.prod(shape[axes[-1] + 1 :]))
    dtype = data.dtype if dtype is None else dtype
    data_cb2 = data.reshape(Npre, 2, Lt, Lz, Ly, Lx // 2, Nsuf)
    data_lexico = numpy.zeros((Npre, Lt, Lz, Ly, Lx, Nsuf), dtype)
    for t in range(Lt):
        for z in range(Lz):
            for y in range(Ly):
                eo = (t + z + y) % 2
                if eo == 0:
                    data_lexico[:, t, z, y, 0::2] = data_cb2[:, 0, t, z, y, :]
                    data_lexico[:, t, z, y, 1::2] = data_cb2[:, 1, t, z, y, :]
                else:
                    data_lexico[:, t, z, y, 1::2] = data_cb2[:, 0, t, z, y, :]
                    data_lexico[:, t, z, y, 0::2] = data_cb2[:, 1, t, z, y, :]
    return data_lexico.reshape(*shape[: axes[0]], Lt, Lz, Ly, Lx, *shape[axes[-1] + 1 :])


def cb2(data: numpy.ndarray, axes: List[int], dtype=None):
    shape = data.shape
    Lt, Lz, Ly, Lx = [shape[axis] for axis in axes]
    Npre = int(numpy.prod(shape[: axes[0]]))
    Nsuf = int(numpy.prod(shape[axes[-1] + 1 :]))
    dtype = data.dtype if dtype is None else dtype
    data_lexico = data.reshape(Npre, Lt, Lz, Ly, Lx, Nsuf)
    data_cb2 = numpy.zeros((Npre, 2, Lt, Lz, Ly, Lx // 2, Nsuf), dtype)
    for t in range(Lt):
        for z in range(Lz):
            for y in range(Ly):
                eo = (t + z + y) % 2
                if eo == 0:
                    data_cb2[:, 0, t, z, y, :] = data_lexico[:, t, z, y, 0::2]
                    data_cb2[:, 1, t, z, y, :] = data_lexico[:, t, z, y, 1::2]
                else:
                    data_cb2[:, 0, t, z, y, :] = data_lexico[:, t, z, y, 1::2]
                    data_cb2[:, 1, t, z, y, :] = data_lexico[:, t, z, y, 0::2]
    return data_cb2.reshape(*shape[: axes[0]], 2, Lt, Lz, Ly, Lx // 2, *shape[axes[-1] + 1 :])


def newLatticeFieldData(latt_info: LatticeInfo, dtype: str):
    from . import getCUDABackend

    backend = getCUDABackend()
    Lx, Ly, Lz, Lt = latt_info.size
    if backend == "numpy":
        if dtype == "Gauge":
            ret = numpy.zeros((Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc), "<c16")
            ret[:] = numpy.identity(Nc)
            return ret
        elif dtype == "Colorvector":
            return numpy.zeros((2, Lt, Lz, Ly, Lx // 2, Nc), "<c16")
        elif dtype == "Fermion":
            return numpy.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Nc), "<c16")
        elif dtype == "Propagator":
            return numpy.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc, Nc), "<c16")
        elif dtype == "StaggeredFermion":
            return numpy.zeros((2, Lt, Lz, Ly, Lx // 2, Nc), "<c16")
        elif dtype == "StaggeredPropagator":
            return numpy.zeros((2, Lt, Lz, Ly, Lx // 2, Nc, Nc), "<c16")
        elif dtype == "Clover":
            return numpy.zeros((2, Lt, Lz, Ly, Lx // 2, 2, ((Ns // 2) * Nc) ** 2), "<f8")
        else:
            raise ValueError(f"Unsupported lattice field type {dtype}")
    elif backend == "cupy":
        import cupy

        if dtype == "Gauge":
            ret = cupy.zeros((Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc), "<c16")
            ret[:] = cupy.identity(Nc)
            return ret
        elif dtype == "Colorvector":
            return cupy.zeros((2, Lt, Lz, Ly, Lx // 2, Nc), "<c16")
        elif dtype == "Fermion":
            return cupy.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Nc), "<c16")
        elif dtype == "Propagator":
            return cupy.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc, Nc), "<c16")
        elif dtype == "StaggeredFermion":
            return cupy.zeros((2, Lt, Lz, Ly, Lx // 2, Nc), "<c16")
        elif dtype == "StaggeredPropagator":
            return cupy.zeros((2, Lt, Lz, Ly, Lx // 2, Nc, Nc), "<c16")
        elif dtype == "Clover":
            return cupy.zeros((2, Lt, Lz, Ly, Lx // 2, 2, ((Ns // 2) * Nc) ** 2), "<f8")
        else:
            raise ValueError(f"Unsupported lattice field type {dtype}")
    elif backend == "torch":
        import torch

        if dtype == "Gauge":
            ret = torch.zeros((Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc), dtype=torch.complex128)
            ret[:] = torch.eye(Nc)
            return ret
        elif dtype == "Colorvector":
            return torch.zeros((2, Lt, Lz, Ly, Lx // 2, Nc), dtype=torch.complex128)
        elif dtype == "Fermion":
            return torch.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Nc), dtype=torch.complex128)
        elif dtype == "Propagator":
            return torch.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc, Nc), dtype=torch.complex128)
        elif dtype == "StaggeredFermion":
            return torch.zeros((2, Lt, Lz, Ly, Lx // 2, Nc), dtype=torch.complex128)
        elif dtype == "StaggeredPropagator":
            return torch.zeros((2, Lt, Lz, Ly, Lx // 2, Nc, Nc), dtype=torch.complex128)
        elif dtype == "Clover":
            return torch.zeros((2, Lt, Lz, Ly, Lx // 2, 2, ((Ns // 2) * Nc) ** 2), dtype=torch.float64)
        else:
            raise ValueError(f"Unsupported lattice field type {dtype}")
    else:
        raise ValueError(f"Unsupported CUDA backend {backend}")


class LatticeField:
    def __init__(self, latt_info: LatticeInfo) -> None:
        self.latt_info = latt_info
        self.data = None

    def setData(self, data):
        from . import getCUDABackend

        backend = getCUDABackend()
        if isinstance(data, numpy.ndarray) or backend == "numpy":
            self.data = numpy.ascontiguousarray(data)
        elif backend == "cupy":
            import cupy

            self.data = cupy.ascontiguousarray(data)
        elif backend == "torch":
            self.data = data.contiguous()
        else:
            raise ValueError(f"Unsupported CUDA backend {backend}")

    def backup(self):
        from . import getCUDABackend

        backend = getCUDABackend()
        if isinstance(self.data, numpy.ndarray) or backend == "numpy":
            return self.data.copy()
        elif backend == "cupy":
            return self.data.copy()
        elif backend == "torch":
            return self.data.clone()
        else:
            raise ValueError(f"Unsupported CUDA backend {backend}")

    def toDevice(self):
        from . import getCUDABackend

        backend = getCUDABackend()
        if backend == "numpy":
            pass
        elif backend == "cupy":
            import cupy

            self.data = cupy.asarray(self.data)
        elif backend == "torch":
            import torch

            self.data = torch.as_tensor(self.data)
        else:
            raise ValueError(f"Unsupported CUDA backend {backend}")

    def toHost(self):
        from . import getCUDABackend

        backend = getCUDABackend()
        if isinstance(self.data, numpy.ndarray) or backend == "numpy":
            pass
        elif backend == "cupy":
            self.data = self.data.get()
        elif backend == "torch":
            self.data = self.data.cpu().numpy()
        else:
            raise ValueError(f"Unsupported CUDA backend {backend}")

    def getHost(self):
        from . import getCUDABackend

        backend = getCUDABackend()
        if isinstance(self.data, numpy.ndarray) or backend == "numpy":
            return self.data.copy()
        elif backend == "cupy":
            return self.data.get()
        elif backend == "torch":
            return self.data.cpu().numpy()
        else:
            raise ValueError(f"Unsupported CUDA backend {backend}")


class LatticeGauge(LatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        Lx, Ly, Lz, Lt = latt_info.size
        if value is None:
            self.setData(newLatticeFieldData(latt_info, "Gauge"))
        else:
            self.setData(value.reshape(Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc))
        self.pure_gauge = None

    def copy(self):
        return LatticeGauge(self.latt_info, self.backup())

    def setAntiPeroidicT(self):
        if self.latt_info.gt == self.latt_info.Gt - 1:
            self.data[Nd - 1, :, self.latt_info.Lt - 1] *= -1

    def setAnisotropy(self, anisotropy: float):
        self.data[: Nd - 1] /= anisotropy

    @property
    def data_ptr(self) -> Pointer:
        return ndarrayPointer(self.data.reshape(-1), True)

    @property
    def data_ptrs(self) -> Pointers:
        return ndarrayPointer(self.data.reshape(4, -1), True)

    def lexico(self):
        return lexico(self.getHost(), [1, 2, 3, 4, 5])

    def ensurePureGauge(self):
        if self.pure_gauge is None:
            from .dirac.pure_gauge import PureGauge

            self.pure_gauge = PureGauge(self.latt_info)

    def staggeredPhase(self):
        self.ensurePureGauge()
        self.pure_gauge.staggeredPhase(self)

    def projectSU3(self, tol: float):
        self.ensurePureGauge()
        self.pure_gauge.projectSU3(self, tol)

    def smearAPE(self, n_steps: int, alpha: float, dir_ignore: int):
        self.ensurePureGauge()
        self.pure_gauge.loadGauge(self)
        self.pure_gauge.smearAPE(n_steps, alpha, dir_ignore)
        self.pure_gauge.saveSmearedGauge(self)

    def smearSTOUT(self, n_steps: int, rho: float, dir_ignore: int):
        self.ensurePureGauge()
        self.pure_gauge.loadGauge(self)
        self.pure_gauge.smearSTOUT(n_steps, rho, dir_ignore)
        self.pure_gauge.saveSmearedGauge(self)

    def smearHYP(self, n_steps: int, alpha1: float, alpha2: float, alpha3: float, dir_ignore: int):
        self.ensurePureGauge()
        self.pure_gauge.loadGauge(self)
        self.pure_gauge.smearHYP(n_steps, alpha1, alpha2, alpha3, dir_ignore)
        self.pure_gauge.saveSmearedGauge(self)

    def plaquette(self):
        self.ensurePureGauge()
        self.pure_gauge.loadGauge(self)
        return self.pure_gauge.plaquette()

    def polyakovLoop(self):
        self.ensurePureGauge()
        self.pure_gauge.loadGauge(self)
        return self.pure_gauge.polyakovLoop()

    def energy(self):
        self.ensurePureGauge()
        self.pure_gauge.loadGauge(self)
        return self.pure_gauge.energy()

    def qcharge(self):
        self.ensurePureGauge()
        self.pure_gauge.loadGauge(self)
        return self.pure_gauge.qcharge()

    def gauss(self, seed: int, sigma: float):
        """
        Generate Gaussian distributed fields and store in the
        resident gauge field.  We create a Gaussian-distributed su(n)
        field and exponentiate it, e.g., U = exp(sigma * H), where H is
        the distributed su(n) field and sigma is the width of the
        distribution (sigma = 0 results in a free field, and sigma = 1 has
        maximum disorder).

        seed: int
            The seed used for the RNG
        sigma: float
            Width of Gaussian distrubution
        """
        self.ensurePureGauge()
        self.pure_gauge.loadGauge(self)
        self.pure_gauge.gauss(seed, sigma)
        self.pure_gauge.saveGauge(self)

    def fixingOVR(
        self,
        gauge_dir: Literal[3, 4],
        Nsteps: int,
        verbose_interval: int,
        relax_boost: float,
        tolerance: float,
        reunit_interval: int,
        stopWtheta: int,
    ):
        """
        Gauge fixing with overrelaxation with support for single and multi GPU.

        Parameters
        ----------
        gauge_dir: {3, 4}
            3 for Coulomb gauge fixing, 4 for Landau gauge fixing
        Nsteps: int
            maximum number of steps to perform gauge fixing
        verbose_interval: int
            print gauge fixing info when iteration count is a multiple of this
        relax_boost: float
            gauge fixing parameter of the overrelaxation method, most common value is 1.5 or 1.7.
        tolerance: float
            torelance value to stop the method, if this value is zero then the method stops when
            iteration reachs the maximum number of steps defined by Nsteps
        reunit_interval: int
            reunitarize gauge field when iteration count is a multiple of this
        stopWtheta: int
            0 for MILC criterion and 1 to use the theta value
        """
        self.ensurePureGauge()
        self.pure_gauge.fixingOVR(
            self, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta
        )

    def fixingFFT(
        self,
        gauge_dir: Literal[3, 4],
        Nsteps: int,
        verbose_interval: int,
        alpha: float,
        autotune: int,
        tolerance: float,
        stopWtheta: int,
    ):
        """
        Gauge fixing with Steepest descent method with FFTs with support for single GPU only.

        Parameters
        ----------
        gauge_dir: {3, 4}
            3 for Coulomb gauge fixing, 4 for Landau gauge fixing
        Nsteps: int
            maximum number of steps to perform gauge fixing
        verbose_interval: int
            print gauge fixing info when iteration count is a multiple of this
        alpha: float
            gauge fixing parameter of the method, most common value is 0.08
        autotune: int
            1 to autotune the method, i.e., if the Fg inverts its tendency we decrease the alpha value
        tolerance: float
            torelance value to stop the method, if this value is zero then the method stops when
            iteration reachs the maximum number of steps defined by Nsteps
        stopWtheta: int
            0 for MILC criterion and 1 to use the theta value
        """
        self.ensurePureGauge()
        self.pure_gauge.fixingFFT(self, gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta)


class LatticeClover(LatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        Lx, Ly, Lz, Lt = latt_info.size
        if value is None:
            self.setData(newLatticeFieldData(latt_info, "Clover"))
        else:
            self.setData(value.reshape(2, Lt, Lz, Ly, Lx // 2, 2, ((Ns // 2) * Nc) ** 2))

    @property
    def data_ptr(self) -> Pointer:
        return ndarrayPointer(self.data.reshape(-1), True)


class LatticeFermion(LatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        Lx, Ly, Lz, Lt = latt_info.size
        if value is None:
            self.setData(newLatticeFieldData(latt_info, "Fermion"))
        else:
            self.setData(value.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Nc))

    @property
    def even(self):
        return self.data[0]

    @even.setter
    def even(self, value):
        self.data[0] = value

    @property
    def odd(self):
        return self.data[1]

    @odd.setter
    def odd(self, value):
        self.data[1] = value

    @property
    def data_ptr(self) -> Pointer:
        return ndarrayPointer(self.data.reshape(-1), True)

    @property
    def even_ptr(self) -> Pointer:
        return ndarrayPointer(self.data.reshape(2, -1)[0], True)

    @property
    def odd_ptr(self) -> Pointer:
        return ndarrayPointer(self.data.reshape(2, -1)[1], True)

    def lexico(self):
        return lexico(self.getHost(), [0, 1, 2, 3, 4])


class LatticePropagator(LatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        Lx, Ly, Lz, Lt = latt_info.size
        if value is None:
            self.setData(newLatticeFieldData(latt_info, "Propagator"))
        else:
            self.setData(value.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc, Nc))

    def lexico(self):
        return lexico(self.getHost(), [0, 1, 2, 3, 4])

    def transpose(self):
        return self.data.transpose(0, 1, 2, 3, 4, 6, 5, 8, 7).copy()


class LatticeStaggeredFermion(LatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        Lx, Ly, Lz, Lt = latt_info.size
        if value is None:
            self.setData(newLatticeFieldData(latt_info, "StaggeredFermion"))
        else:
            self.setData(value.reshape(2, Lt, Lz, Ly, Lx // 2, Nc))

    @property
    def even(self):
        return self.data[0]

    @even.setter
    def even(self, value):
        self.data[0] = value

    @property
    def odd(self):
        return self.data[1]

    @odd.setter
    def odd(self, value):
        self.data[1] = value

    @property
    def data_ptr(self) -> Pointer:
        return ndarrayPointer(self.data.reshape(-1), True)

    @property
    def even_ptr(self) -> Pointer:
        return ndarrayPointer(self.data.reshape(2, -1)[0], True)

    @property
    def odd_ptr(self) -> Pointer:
        return ndarrayPointer(self.data.reshape(2, -1)[1], True)

    def lexico(self):
        return lexico(self.getHost(), [0, 1, 2, 3, 4])


class LatticeStaggeredPropagator(LatticeField):
    def __init__(self, latt_info: LatticeInfo, value=None) -> None:
        super().__init__(latt_info)
        Lx, Ly, Lz, Lt = latt_info.size
        if value is None:
            self.setData(newLatticeFieldData(latt_info, "StaggeredPropagator"))
        else:
            self.setData(value.reshape(2, Lt, Lz, Ly, Lx // 2, Nc, Nc))

    def lexico(self):
        return lexico(self.getHost(), [0, 1, 2, 3, 4])

    def transpose(self):
        return self.data.transpose(0, 1, 2, 3, 4, 6, 5).copy()
