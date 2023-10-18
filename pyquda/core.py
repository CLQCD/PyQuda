from typing import List, Union
import warnings
from math import sqrt

from . import pyquda as quda
from . import enum_quda
from .field import (
    LatticeGauge,
    LatticeFermion,
    LatticePropagator,
    LatticeStaggeredFermion,
    LatticeStaggeredPropagator,
    Nc,
    Nd,
    Ns,
    lexico,
    cb2,
)
from .dslash.abstract import Dslash
from .utils.source import source


def hyp_smear(latt_size: List[int], gauge: LatticeGauge, nstep: int, alpha1: float, alpha2: float, alpha3: float):
    smear_param = quda.QudaGaugeSmearParam()
    smear_param.n_steps = nstep
    smear_param.alpha1 = alpha1
    smear_param.alpha2 = alpha2
    smear_param.alpha3 = alpha3
    smear_param.meas_interval = nstep + 1
    smear_param.smear_type = enum_quda.QudaGaugeSmearType.QUDA_GAUGE_SMEAR_HYP
    obs_param = quda.QudaGaugeObservableParam()
    obs_param.compute_qcharge = enum_quda.QudaBoolean.QUDA_BOOLEAN_TRUE
    dslash = getDslash(latt_size, 0, 0, 0, anti_periodic_t=False)
    dslash.gauge_param.reconstruct = enum_quda.QudaReconstructType.QUDA_RECONSTRUCT_NO
    dslash.loadGauge(gauge)
    quda.performGaugeSmearQuda(smear_param, obs_param)
    dslash.gauge_param.type = enum_quda.QudaLinkType.QUDA_SMEARED_LINKS
    quda.saveGaugeQuda(gauge.data_ptrs, dslash.gauge_param)


def smear(latt_size: List[int], gauge: LatticeGauge, nstep: int, rho: float):
    """Preserve for compability."""
    warnings.warn("Deprecated. Pleause `stout_smear()` instead.", DeprecationWarning)
    stout_smear(latt_size, gauge, nstep, rho)


def stout_smear(latt_size: List[int], gauge: LatticeGauge, nstep: int, rho: float):
    smear_param = quda.QudaGaugeSmearParam()
    smear_param.n_steps = nstep
    smear_param.rho = rho
    smear_param.meas_interval = nstep + 1
    smear_param.smear_type = enum_quda.QudaGaugeSmearType.QUDA_GAUGE_SMEAR_STOUT
    obs_param = quda.QudaGaugeObservableParam()
    obs_param.compute_qcharge = enum_quda.QudaBoolean.QUDA_BOOLEAN_TRUE
    dslash = getDslash(latt_size, 0, 0, 0, anti_periodic_t=False)
    dslash.gauge_param.reconstruct = enum_quda.QudaReconstructType.QUDA_RECONSTRUCT_NO
    dslash.loadGauge(gauge)
    quda.performGaugeSmearQuda(smear_param, obs_param)
    dslash.gauge_param.type = enum_quda.QudaLinkType.QUDA_SMEARED_LINKS
    quda.saveGaugeQuda(gauge.data_ptrs, dslash.gauge_param)


def smear4(latt_size: List[int], gauge: LatticeGauge, nstep: int, rho: float):
    """Preserve for compability."""
    warnings.warn("Deprecated. Pleause `stout_smear_4d()` instead.", DeprecationWarning)
    stout_smear_4d(latt_size, gauge, nstep, rho)


def stout_smear_4d(latt_size: List[int], gauge: LatticeGauge, nstep: int, rho: float):
    smear_param = quda.QudaGaugeSmearParam()
    smear_param.n_steps = nstep
    smear_param.rho = rho
    smear_param.epsilon = 1.0
    smear_param.meas_interval = nstep + 1
    smear_param.smear_type = enum_quda.QudaGaugeSmearType.QUDA_GAUGE_SMEAR_OVRIMP_STOUT
    obs_param = quda.QudaGaugeObservableParam()
    obs_param.compute_qcharge = enum_quda.QudaBoolean.QUDA_BOOLEAN_TRUE
    dslash = getDslash(latt_size, 0, 0, 0, anti_periodic_t=False)
    dslash.gauge_param.reconstruct = enum_quda.QudaReconstructType.QUDA_RECONSTRUCT_NO
    dslash.loadGauge(gauge)
    quda.performGaugeSmearQuda(smear_param, obs_param)
    dslash.gauge_param.type = enum_quda.QudaLinkType.QUDA_SMEARED_LINKS
    quda.saveGaugeQuda(gauge.data_ptrs, dslash.gauge_param)


def invert(
    dslash: Dslash,
    source_type: str,
    t_srce: Union[int, List[int]],
    source_phase=None,
    rho: float = 0.0,
    nsteps: int = 1,
):
    latt_size = dslash.gauge_param.X
    xi = dslash.gauge_param.anisotropy
    Lx, Ly, Lz, Lt = latt_size
    Vol = Lx * Ly * Lz * Lt

    prop = LatticePropagator(latt_size)
    data = prop.data.reshape(Vol, Ns, Ns, Nc, Nc)
    for spin in range(Ns):
        for color in range(Nc):
            b = source(latt_size, source_type, t_srce, spin, color, source_phase, rho, nsteps, xi)
            x = dslash.invert(b)
            data[:, :, spin, :, color] = x.data.reshape(Vol, Ns, Nc)

    return prop


def invertStaggered(
    dslash: Dslash,
    source_type: str,
    t_srce: Union[int, List[int]],
    source_phase=None,
    rho: float = 0.0,
    nsteps: int = 1,
):
    latt_size = dslash.gauge_param.X
    xi = dslash.gauge_param.anisotropy
    Lx, Ly, Lz, Lt = latt_size
    Vol = Lx * Ly * Lz * Lt

    prop = LatticeStaggeredPropagator(latt_size)
    data = prop.data.reshape(Vol, Nc, Nc)
    for color in range(Nc):
        b = source(latt_size, source_type, t_srce, None, color, source_phase, rho, nsteps, xi)
        x = dslash.invert(b)
        data[:, :, color] = x.data.reshape(Vol, Nc)

    return prop


def invert12(b12: LatticePropagator, dslash: Dslash):
    latt_size = b12.latt_size
    Lx, Ly, Lz, Lt = latt_size
    Vol = Lx * Ly * Lz * Lt

    x12 = LatticePropagator(latt_size)
    for spin in range(Ns):
        for color in range(Nc):
            b = LatticeFermion(latt_size)
            data = b.data.reshape(Vol, Ns, Nc)
            data[:] = b12.data.reshape(Vol, Ns, Ns, Nc, Nc)[:, :, spin, :, color]
            x = dslash.invert(b)
            data = x12.data.reshape(Vol, Ns, Ns, Nc, Nc)
            data[:, :, spin, :, color] = x.data.reshape(Vol, Ns, Nc)

    return x12


def getDslash(
    latt_size: List[int],
    mass: float,
    tol: float,
    maxiter: int,
    xi_0: float = 1.0,
    nu: float = 1.0,
    clover_coeff_t: float = 0.0,
    clover_coeff_r: float = 1.0,
    anti_periodic_t: bool = True,
    multigrid: List[List[int]] = None,
):
    xi = xi_0 / nu
    kappa = 1 / (2 * (mass + 1 + (Nd - 1) / xi))
    if xi != 1.0:
        clover_coeff = xi_0 * clover_coeff_t**2 / clover_coeff_r
        clover_xi = sqrt(xi_0 * clover_coeff_t / clover_coeff_r)
    else:
        clover_coeff = clover_coeff_t
        clover_xi = 1.0
    if anti_periodic_t:
        t_boundary = -1
    else:
        t_boundary = 1
    if not multigrid:
        geo_block_size = None
    else:
        if not isinstance(multigrid, list):
            geo_block_size = [[2, 2, 2, 2], [4, 4, 4, 4]]
        else:
            geo_block_size = multigrid

    if clover_coeff != 0.0:
        from .dslash import clover_wilson

        return clover_wilson.CloverWilson(
            latt_size, mass, kappa, tol, maxiter, xi, clover_coeff, clover_xi, t_boundary, geo_block_size
        )
    else:
        from .dslash import wilson

        return wilson.Wilson(latt_size, mass, kappa, tol, maxiter, xi, t_boundary, geo_block_size)


def getStaggeredDslash(
    latt_size: List[int],
    mass: float,
    tol: float,
    maxiter: int,
    tadpole_coeff: float = 1.0,
    naik_epsilon: float = 0.0,
    anti_periodic_t: bool = True,
):
    kappa = 1 / (2 * (mass + Nd))
    if anti_periodic_t:
        t_boundary = -1
    else:
        t_boundary = 1

    from .dslash import hisq

    return hisq.HISQ(latt_size, mass, kappa, tol, maxiter, tadpole_coeff, naik_epsilon, t_boundary, None)
