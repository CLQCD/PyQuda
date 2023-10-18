import os
import sys
import io
from contextlib import contextmanager
from tempfile import TemporaryFile

import ctypes
import cython

from libc.stdio cimport stdout

cimport quda
from pyquda.pointer cimport Pointer, Pointers, Pointerss

libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')

@contextmanager
def redirect_stdout(stream):
    stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to_fd):
        """Redirect stdout to the given file descriptor."""
        libc.fflush(c_stdout)
        sys.stdout.close()
        os.dup2(to_fd, stdout_fd)
        sys.stdout = io.TextIOWrapper(os.fdopen(stdout_fd, 'wb'))

    saved_stdout_fd = os.dup(stdout_fd)
    try:
        tfile = TemporaryFile(mode='w+b')
        _redirect_stdout(tfile.fileno())
        yield
        _redirect_stdout(saved_stdout_fd)
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stdout_fd)

cdef class QudaGaugeParam:
    cdef quda.QudaGaugeParam param

    def __init__(self):
        self.param = quda.newQudaGaugeParam()

    def __repr__(self):
        buf = io.BytesIO()
        with redirect_stdout(buf):
            quda.printQudaGaugeParam(&self.param)
        ret = buf.getvalue().decode("utf-8")
        return ret

    cdef from_ptr(self, quda.QudaGaugeParam *ptr):
        self.param = cython.operator.dereference(ptr)

    @property
    def struct_size(self):
        return self.param.struct_size

    @struct_size.setter
    def struct_size(self, value):
        self.param.struct_size = value

    @property
    def location(self):
        return self.param.location

    @location.setter
    def location(self, value):
        self.param.location = value

    @property
    def X(self):
        return self.param.X

    @X.setter
    def X(self, value):
        self.param.X = value

    @property
    def anisotropy(self):
        return self.param.anisotropy

    @anisotropy.setter
    def anisotropy(self, value):
        self.param.anisotropy = value

    @property
    def tadpole_coeff(self):
        return self.param.tadpole_coeff

    @tadpole_coeff.setter
    def tadpole_coeff(self, value):
        self.param.tadpole_coeff = value

    @property
    def scale(self):
        return self.param.scale

    @scale.setter
    def scale(self, value):
        self.param.scale = value

    @property
    def type(self):
        return self.param.type

    @type.setter
    def type(self, value):
        self.param.type = value

    @property
    def gauge_order(self):
        return self.param.gauge_order

    @gauge_order.setter
    def gauge_order(self, value):
        self.param.gauge_order = value

    @property
    def t_boundary(self):
        return self.param.t_boundary

    @t_boundary.setter
    def t_boundary(self, value):
        self.param.t_boundary = value

    @property
    def cpu_prec(self):
        return self.param.cpu_prec

    @cpu_prec.setter
    def cpu_prec(self, value):
        self.param.cpu_prec = value

    @property
    def cuda_prec(self):
        return self.param.cuda_prec

    @cuda_prec.setter
    def cuda_prec(self, value):
        self.param.cuda_prec = value

    @property
    def reconstruct(self):
        return self.param.reconstruct

    @reconstruct.setter
    def reconstruct(self, value):
        self.param.reconstruct = value

    @property
    def cuda_prec_sloppy(self):
        return self.param.cuda_prec_sloppy

    @cuda_prec_sloppy.setter
    def cuda_prec_sloppy(self, value):
        self.param.cuda_prec_sloppy = value

    @property
    def reconstruct_sloppy(self):
        return self.param.reconstruct_sloppy

    @reconstruct_sloppy.setter
    def reconstruct_sloppy(self, value):
        self.param.reconstruct_sloppy = value

    @property
    def cuda_prec_refinement_sloppy(self):
        return self.param.cuda_prec_refinement_sloppy

    @cuda_prec_refinement_sloppy.setter
    def cuda_prec_refinement_sloppy(self, value):
        self.param.cuda_prec_refinement_sloppy = value

    @property
    def reconstruct_refinement_sloppy(self):
        return self.param.reconstruct_refinement_sloppy

    @reconstruct_refinement_sloppy.setter
    def reconstruct_refinement_sloppy(self, value):
        self.param.reconstruct_refinement_sloppy = value

    @property
    def cuda_prec_precondition(self):
        return self.param.cuda_prec_precondition

    @cuda_prec_precondition.setter
    def cuda_prec_precondition(self, value):
        self.param.cuda_prec_precondition = value

    @property
    def reconstruct_precondition(self):
        return self.param.reconstruct_precondition

    @reconstruct_precondition.setter
    def reconstruct_precondition(self, value):
        self.param.reconstruct_precondition = value

    @property
    def cuda_prec_eigensolver(self):
        return self.param.cuda_prec_eigensolver

    @cuda_prec_eigensolver.setter
    def cuda_prec_eigensolver(self, value):
        self.param.cuda_prec_eigensolver = value

    @property
    def reconstruct_eigensolver(self):
        return self.param.reconstruct_eigensolver

    @reconstruct_eigensolver.setter
    def reconstruct_eigensolver(self, value):
        self.param.reconstruct_eigensolver = value

    @property
    def gauge_fix(self):
        return self.param.gauge_fix

    @gauge_fix.setter
    def gauge_fix(self, value):
        self.param.gauge_fix = value

    @property
    def ga_pad(self):
        return self.param.ga_pad

    @ga_pad.setter
    def ga_pad(self, value):
        self.param.ga_pad = value

    @property
    def site_ga_pad(self):
        return self.param.site_ga_pad

    @site_ga_pad.setter
    def site_ga_pad(self, value):
        self.param.site_ga_pad = value

    @property
    def staple_pad(self):
        return self.param.staple_pad

    @staple_pad.setter
    def staple_pad(self, value):
        self.param.staple_pad = value

    @property
    def llfat_ga_pad(self):
        return self.param.llfat_ga_pad

    @llfat_ga_pad.setter
    def llfat_ga_pad(self, value):
        self.param.llfat_ga_pad = value

    @property
    def mom_ga_pad(self):
        return self.param.mom_ga_pad

    @mom_ga_pad.setter
    def mom_ga_pad(self, value):
        self.param.mom_ga_pad = value

    @property
    def staggered_phase_type(self):
        return self.param.staggered_phase_type

    @staggered_phase_type.setter
    def staggered_phase_type(self, value):
        self.param.staggered_phase_type = value

    @property
    def staggered_phase_applied(self):
        return self.param.staggered_phase_applied

    @staggered_phase_applied.setter
    def staggered_phase_applied(self, value):
        self.param.staggered_phase_applied = value

    @property
    def i_mu(self):
        return self.param.i_mu

    @i_mu.setter
    def i_mu(self, value):
        self.param.i_mu = value

    @property
    def overlap(self):
        return self.param.overlap

    @overlap.setter
    def overlap(self, value):
        self.param.overlap = value

    @property
    def overwrite_gauge(self):
        return self.param.overwrite_gauge

    @overwrite_gauge.setter
    def overwrite_gauge(self, value):
        self.param.overwrite_gauge = value

    @property
    def overwrite_mom(self):
        return self.param.overwrite_mom

    @overwrite_mom.setter
    def overwrite_mom(self, value):
        self.param.overwrite_mom = value

    @property
    def use_resident_gauge(self):
        return self.param.use_resident_gauge

    @use_resident_gauge.setter
    def use_resident_gauge(self, value):
        self.param.use_resident_gauge = value

    @property
    def use_resident_mom(self):
        return self.param.use_resident_mom

    @use_resident_mom.setter
    def use_resident_mom(self, value):
        self.param.use_resident_mom = value

    @property
    def make_resident_gauge(self):
        return self.param.make_resident_gauge

    @make_resident_gauge.setter
    def make_resident_gauge(self, value):
        self.param.make_resident_gauge = value

    @property
    def make_resident_mom(self):
        return self.param.make_resident_mom

    @make_resident_mom.setter
    def make_resident_mom(self, value):
        self.param.make_resident_mom = value

    @property
    def return_result_gauge(self):
        return self.param.return_result_gauge

    @return_result_gauge.setter
    def return_result_gauge(self, value):
        self.param.return_result_gauge = value

    @property
    def return_result_mom(self):
        return self.param.return_result_mom

    @return_result_mom.setter
    def return_result_mom(self, value):
        self.param.return_result_mom = value

    @property
    def gauge_offset(self):
        return self.param.gauge_offset

    @gauge_offset.setter
    def gauge_offset(self, value):
        self.param.gauge_offset = value

    @property
    def mom_offset(self):
        return self.param.mom_offset

    @mom_offset.setter
    def mom_offset(self, value):
        self.param.mom_offset = value

    @property
    def site_size(self):
        return self.param.site_size

    @site_size.setter
    def site_size(self, value):
        self.param.site_size = value

cdef class QudaInvertParam:
    cdef quda.QudaInvertParam param

    def __init__(self):
        self.param = quda.newQudaInvertParam()

    def __repr__(self):
        buf = io.BytesIO()
        with redirect_stdout(buf):
            quda.printQudaInvertParam(&self.param)
        ret = buf.getvalue().decode("utf-8")
        return ret

    cdef from_ptr(self, quda.QudaInvertParam *ptr):
        self.param = cython.operator.dereference(ptr)

    @property
    def struct_size(self):
        return self.param.struct_size

    @struct_size.setter
    def struct_size(self, value):
        self.param.struct_size = value

    @property
    def input_location(self):
        return self.param.input_location

    @input_location.setter
    def input_location(self, value):
        self.param.input_location = value

    @property
    def output_location(self):
        return self.param.output_location

    @output_location.setter
    def output_location(self, value):
        self.param.output_location = value

    @property
    def dslash_type(self):
        return self.param.dslash_type

    @dslash_type.setter
    def dslash_type(self, value):
        self.param.dslash_type = value

    @property
    def inv_type(self):
        return self.param.inv_type

    @inv_type.setter
    def inv_type(self, value):
        self.param.inv_type = value

    @property
    def mass(self):
        return self.param.mass

    @mass.setter
    def mass(self, value):
        self.param.mass = value

    @property
    def kappa(self):
        return self.param.kappa

    @kappa.setter
    def kappa(self, value):
        self.param.kappa = value

    @property
    def m5(self):
        return self.param.m5

    @m5.setter
    def m5(self, value):
        self.param.m5 = value

    @property
    def Ls(self):
        return self.param.Ls

    @Ls.setter
    def Ls(self, value):
        self.param.Ls = value

    @property
    def b_5(self):
        return self.param.b_5

    @b_5.setter
    def b_5(self, value):
        self.param.b_5 = value

    @property
    def c_5(self):
        return self.param.c_5

    @c_5.setter
    def c_5(self, value):
        self.param.c_5 = value

    @property
    def eofa_shift(self):
        return self.param.eofa_shift

    @eofa_shift.setter
    def eofa_shift(self, value):
        self.param.eofa_shift = value

    @property
    def eofa_pm(self):
        return self.param.eofa_pm

    @eofa_pm.setter
    def eofa_pm(self, value):
        self.param.eofa_pm = value

    @property
    def mq1(self):
        return self.param.mq1

    @mq1.setter
    def mq1(self, value):
        self.param.mq1 = value

    @property
    def mq2(self):
        return self.param.mq2

    @mq2.setter
    def mq2(self, value):
        self.param.mq2 = value

    @property
    def mq3(self):
        return self.param.mq3

    @mq3.setter
    def mq3(self, value):
        self.param.mq3 = value

    @property
    def tm_rho(self):
        return self.param.tm_rho

    @tm_rho.setter
    def tm_rho(self, value):
        self.param.tm_rho = value

    @property
    def mu(self):
        return self.param.mu

    @mu.setter
    def mu(self, value):
        self.param.mu = value

    @property
    def epsilon(self):
        return self.param.epsilon

    @epsilon.setter
    def epsilon(self, value):
        self.param.epsilon = value

    @property
    def twist_flavor(self):
        return self.param.twist_flavor

    @twist_flavor.setter
    def twist_flavor(self, value):
        self.param.twist_flavor = value

    @property
    def laplace3D(self):
        return self.param.laplace3D

    @laplace3D.setter
    def laplace3D(self, value):
        self.param.laplace3D = value

    @property
    def tol(self):
        return self.param.tol

    @tol.setter
    def tol(self, value):
        self.param.tol = value

    @property
    def tol_restart(self):
        return self.param.tol_restart

    @tol_restart.setter
    def tol_restart(self, value):
        self.param.tol_restart = value

    @property
    def tol_hq(self):
        return self.param.tol_hq

    @tol_hq.setter
    def tol_hq(self, value):
        self.param.tol_hq = value

    @property
    def compute_true_res(self):
        return self.param.compute_true_res

    @compute_true_res.setter
    def compute_true_res(self, value):
        self.param.compute_true_res = value

    @property
    def true_res(self):
        return self.param.true_res

    @true_res.setter
    def true_res(self, value):
        self.param.true_res = value

    @property
    def true_res_hq(self):
        return self.param.true_res_hq

    @true_res_hq.setter
    def true_res_hq(self, value):
        self.param.true_res_hq = value

    @property
    def maxiter(self):
        return self.param.maxiter

    @maxiter.setter
    def maxiter(self, value):
        self.param.maxiter = value

    @property
    def reliable_delta(self):
        return self.param.reliable_delta

    @reliable_delta.setter
    def reliable_delta(self, value):
        self.param.reliable_delta = value

    @property
    def reliable_delta_refinement(self):
        return self.param.reliable_delta_refinement

    @reliable_delta_refinement.setter
    def reliable_delta_refinement(self, value):
        self.param.reliable_delta_refinement = value

    @property
    def use_alternative_reliable(self):
        return self.param.use_alternative_reliable

    @use_alternative_reliable.setter
    def use_alternative_reliable(self, value):
        self.param.use_alternative_reliable = value

    @property
    def use_sloppy_partial_accumulator(self):
        return self.param.use_sloppy_partial_accumulator

    @use_sloppy_partial_accumulator.setter
    def use_sloppy_partial_accumulator(self, value):
        self.param.use_sloppy_partial_accumulator = value

    @property
    def solution_accumulator_pipeline(self):
        return self.param.solution_accumulator_pipeline

    @solution_accumulator_pipeline.setter
    def solution_accumulator_pipeline(self, value):
        self.param.solution_accumulator_pipeline = value

    @property
    def max_res_increase(self):
        return self.param.max_res_increase

    @max_res_increase.setter
    def max_res_increase(self, value):
        self.param.max_res_increase = value

    @property
    def max_res_increase_total(self):
        return self.param.max_res_increase_total

    @max_res_increase_total.setter
    def max_res_increase_total(self, value):
        self.param.max_res_increase_total = value

    @property
    def max_hq_res_increase(self):
        return self.param.max_hq_res_increase

    @max_hq_res_increase.setter
    def max_hq_res_increase(self, value):
        self.param.max_hq_res_increase = value

    @property
    def max_hq_res_restart_total(self):
        return self.param.max_hq_res_restart_total

    @max_hq_res_restart_total.setter
    def max_hq_res_restart_total(self, value):
        self.param.max_hq_res_restart_total = value

    @property
    def heavy_quark_check(self):
        return self.param.heavy_quark_check

    @heavy_quark_check.setter
    def heavy_quark_check(self, value):
        self.param.heavy_quark_check = value

    @property
    def pipeline(self):
        return self.param.pipeline

    @pipeline.setter
    def pipeline(self, value):
        self.param.pipeline = value

    @property
    def num_offset(self):
        return self.param.num_offset

    @num_offset.setter
    def num_offset(self, value):
        self.param.num_offset = value

    @property
    def num_src(self):
        return self.param.num_src

    @num_src.setter
    def num_src(self, value):
        self.param.num_src = value

    @property
    def num_src_per_sub_partition(self):
        return self.param.num_src_per_sub_partition

    @num_src_per_sub_partition.setter
    def num_src_per_sub_partition(self, value):
        self.param.num_src_per_sub_partition = value

    @property
    def split_grid(self):
        return self.param.split_grid

    @split_grid.setter
    def split_grid(self, value):
        self.param.split_grid = value

    @property
    def overlap(self):
        return self.param.overlap

    @overlap.setter
    def overlap(self, value):
        self.param.overlap = value

    @property
    def offset(self):
        return self.param.offset

    @offset.setter
    def offset(self, value):
        self.param.offset = value

    @property
    def tol_offset(self):
        return self.param.tol_offset

    @tol_offset.setter
    def tol_offset(self, value):
        self.param.tol_offset = value

    @property
    def tol_hq_offset(self):
        return self.param.tol_hq_offset

    @tol_hq_offset.setter
    def tol_hq_offset(self, value):
        self.param.tol_hq_offset = value

    @property
    def true_res_offset(self):
        return self.param.true_res_offset

    @true_res_offset.setter
    def true_res_offset(self, value):
        self.param.true_res_offset = value

    @property
    def iter_res_offset(self):
        return self.param.iter_res_offset

    @iter_res_offset.setter
    def iter_res_offset(self, value):
        self.param.iter_res_offset = value

    @property
    def true_res_hq_offset(self):
        return self.param.true_res_hq_offset

    @true_res_hq_offset.setter
    def true_res_hq_offset(self, value):
        self.param.true_res_hq_offset = value

    @property
    def residue(self):
        return self.param.residue

    @residue.setter
    def residue(self, value):
        self.param.residue = value

    @property
    def compute_action(self):
        return self.param.compute_action

    @compute_action.setter
    def compute_action(self, value):
        self.param.compute_action = value

    @property
    def action(self):
        return self.param.action

    @action.setter
    def action(self, value):
        self.param.action = value

    @property
    def solution_type(self):
        return self.param.solution_type

    @solution_type.setter
    def solution_type(self, value):
        self.param.solution_type = value

    @property
    def solve_type(self):
        return self.param.solve_type

    @solve_type.setter
    def solve_type(self, value):
        self.param.solve_type = value

    @property
    def matpc_type(self):
        return self.param.matpc_type

    @matpc_type.setter
    def matpc_type(self, value):
        self.param.matpc_type = value

    @property
    def dagger(self):
        return self.param.dagger

    @dagger.setter
    def dagger(self, value):
        self.param.dagger = value

    @property
    def mass_normalization(self):
        return self.param.mass_normalization

    @mass_normalization.setter
    def mass_normalization(self, value):
        self.param.mass_normalization = value

    @property
    def solver_normalization(self):
        return self.param.solver_normalization

    @solver_normalization.setter
    def solver_normalization(self, value):
        self.param.solver_normalization = value

    @property
    def preserve_source(self):
        return self.param.preserve_source

    @preserve_source.setter
    def preserve_source(self, value):
        self.param.preserve_source = value

    @property
    def cpu_prec(self):
        return self.param.cpu_prec

    @cpu_prec.setter
    def cpu_prec(self, value):
        self.param.cpu_prec = value

    @property
    def cuda_prec(self):
        return self.param.cuda_prec

    @cuda_prec.setter
    def cuda_prec(self, value):
        self.param.cuda_prec = value

    @property
    def cuda_prec_sloppy(self):
        return self.param.cuda_prec_sloppy

    @cuda_prec_sloppy.setter
    def cuda_prec_sloppy(self, value):
        self.param.cuda_prec_sloppy = value

    @property
    def cuda_prec_refinement_sloppy(self):
        return self.param.cuda_prec_refinement_sloppy

    @cuda_prec_refinement_sloppy.setter
    def cuda_prec_refinement_sloppy(self, value):
        self.param.cuda_prec_refinement_sloppy = value

    @property
    def cuda_prec_precondition(self):
        return self.param.cuda_prec_precondition

    @cuda_prec_precondition.setter
    def cuda_prec_precondition(self, value):
        self.param.cuda_prec_precondition = value

    @property
    def cuda_prec_eigensolver(self):
        return self.param.cuda_prec_eigensolver

    @cuda_prec_eigensolver.setter
    def cuda_prec_eigensolver(self, value):
        self.param.cuda_prec_eigensolver = value

    @property
    def dirac_order(self):
        return self.param.dirac_order

    @dirac_order.setter
    def dirac_order(self, value):
        self.param.dirac_order = value

    @property
    def gamma_basis(self):
        return self.param.gamma_basis

    @gamma_basis.setter
    def gamma_basis(self, value):
        self.param.gamma_basis = value

    @property
    def clover_location(self):
        return self.param.clover_location

    @clover_location.setter
    def clover_location(self, value):
        self.param.clover_location = value

    @property
    def clover_cpu_prec(self):
        return self.param.clover_cpu_prec

    @clover_cpu_prec.setter
    def clover_cpu_prec(self, value):
        self.param.clover_cpu_prec = value

    @property
    def clover_cuda_prec(self):
        return self.param.clover_cuda_prec

    @clover_cuda_prec.setter
    def clover_cuda_prec(self, value):
        self.param.clover_cuda_prec = value

    @property
    def clover_cuda_prec_sloppy(self):
        return self.param.clover_cuda_prec_sloppy

    @clover_cuda_prec_sloppy.setter
    def clover_cuda_prec_sloppy(self, value):
        self.param.clover_cuda_prec_sloppy = value

    @property
    def clover_cuda_prec_refinement_sloppy(self):
        return self.param.clover_cuda_prec_refinement_sloppy

    @clover_cuda_prec_refinement_sloppy.setter
    def clover_cuda_prec_refinement_sloppy(self, value):
        self.param.clover_cuda_prec_refinement_sloppy = value

    @property
    def clover_cuda_prec_precondition(self):
        return self.param.clover_cuda_prec_precondition

    @clover_cuda_prec_precondition.setter
    def clover_cuda_prec_precondition(self, value):
        self.param.clover_cuda_prec_precondition = value

    @property
    def clover_cuda_prec_eigensolver(self):
        return self.param.clover_cuda_prec_eigensolver

    @clover_cuda_prec_eigensolver.setter
    def clover_cuda_prec_eigensolver(self, value):
        self.param.clover_cuda_prec_eigensolver = value

    @property
    def clover_order(self):
        return self.param.clover_order

    @clover_order.setter
    def clover_order(self, value):
        self.param.clover_order = value

    @property
    def use_init_guess(self):
        return self.param.use_init_guess

    @use_init_guess.setter
    def use_init_guess(self, value):
        self.param.use_init_guess = value

    @property
    def clover_csw(self):
        return self.param.clover_csw

    @clover_csw.setter
    def clover_csw(self, value):
        self.param.clover_csw = value

    @property
    def clover_coeff(self):
        return self.param.clover_coeff

    @clover_coeff.setter
    def clover_coeff(self, value):
        self.param.clover_coeff = value

    @property
    def clover_rho(self):
        return self.param.clover_rho

    @clover_rho.setter
    def clover_rho(self, value):
        self.param.clover_rho = value

    @property
    def compute_clover_trlog(self):
        return self.param.compute_clover_trlog

    @compute_clover_trlog.setter
    def compute_clover_trlog(self, value):
        self.param.compute_clover_trlog = value

    @property
    def trlogA(self):
        return self.param.trlogA

    @trlogA.setter
    def trlogA(self, value):
        self.param.trlogA = value

    @property
    def compute_clover(self):
        return self.param.compute_clover

    @compute_clover.setter
    def compute_clover(self, value):
        self.param.compute_clover = value

    @property
    def compute_clover_inverse(self):
        return self.param.compute_clover_inverse

    @compute_clover_inverse.setter
    def compute_clover_inverse(self, value):
        self.param.compute_clover_inverse = value

    @property
    def return_clover(self):
        return self.param.return_clover

    @return_clover.setter
    def return_clover(self, value):
        self.param.return_clover = value

    @property
    def return_clover_inverse(self):
        return self.param.return_clover_inverse

    @return_clover_inverse.setter
    def return_clover_inverse(self, value):
        self.param.return_clover_inverse = value

    @property
    def verbosity(self):
        return self.param.verbosity

    @verbosity.setter
    def verbosity(self, value):
        self.param.verbosity = value

    @property
    def iter(self):
        return self.param.iter

    @iter.setter
    def iter(self, value):
        self.param.iter = value

    @property
    def gflops(self):
        return self.param.gflops

    @gflops.setter
    def gflops(self, value):
        self.param.gflops = value

    @property
    def secs(self):
        return self.param.secs

    @secs.setter
    def secs(self, value):
        self.param.secs = value

    @property
    def tune(self):
        return self.param.tune

    @tune.setter
    def tune(self, value):
        self.param.tune = value

    @property
    def Nsteps(self):
        return self.param.Nsteps

    @Nsteps.setter
    def Nsteps(self, value):
        self.param.Nsteps = value

    @property
    def gcrNkrylov(self):
        return self.param.gcrNkrylov

    @gcrNkrylov.setter
    def gcrNkrylov(self, value):
        self.param.gcrNkrylov = value

    @property
    def inv_type_precondition(self):
        return self.param.inv_type_precondition

    @inv_type_precondition.setter
    def inv_type_precondition(self, value):
        self.param.inv_type_precondition = value

    @property
    def preconditioner(self):
        ptr = Pointer("void")
        ptr.set_ptr(self.param.preconditioner)
        return ptr

    @preconditioner.setter
    def preconditioner(self, value):
        self.set_preconditioner(value)

    cdef set_preconditioner(self, Pointer value):
        assert value.dtype == "void"
        self.param.preconditioner = value.ptr

    @property
    def deflation_op(self):
        ptr = Pointer("void")
        ptr.set_ptr(self.param.deflation_op)
        return ptr

    @deflation_op.setter
    def deflation_op(self, value):
        self.set_deflation_op(value)

    cdef set_deflation_op(self, Pointer value):
        assert value.dtype == "void"
        self.param.deflation_op = value.ptr

    @property
    def eig_param(self):
        ptr = Pointer("void")
        ptr.set_ptr(self.param.eig_param)
        return ptr

    @eig_param.setter
    def eig_param(self, value):
        self.set_eig_param(value)

    cdef set_eig_param(self, Pointer value):
        assert value.dtype == "void"
        self.param.eig_param = value.ptr

    @property
    def deflate(self):
        return self.param.deflate

    @deflate.setter
    def deflate(self, value):
        self.param.deflate = value

    @property
    def dslash_type_precondition(self):
        return self.param.dslash_type_precondition

    @dslash_type_precondition.setter
    def dslash_type_precondition(self, value):
        self.param.dslash_type_precondition = value

    @property
    def verbosity_precondition(self):
        return self.param.verbosity_precondition

    @verbosity_precondition.setter
    def verbosity_precondition(self, value):
        self.param.verbosity_precondition = value

    @property
    def tol_precondition(self):
        return self.param.tol_precondition

    @tol_precondition.setter
    def tol_precondition(self, value):
        self.param.tol_precondition = value

    @property
    def maxiter_precondition(self):
        return self.param.maxiter_precondition

    @maxiter_precondition.setter
    def maxiter_precondition(self, value):
        self.param.maxiter_precondition = value

    @property
    def omega(self):
        return self.param.omega

    @omega.setter
    def omega(self, value):
        self.param.omega = value

    @property
    def ca_basis(self):
        return self.param.ca_basis

    @ca_basis.setter
    def ca_basis(self, value):
        self.param.ca_basis = value

    @property
    def ca_lambda_min(self):
        return self.param.ca_lambda_min

    @ca_lambda_min.setter
    def ca_lambda_min(self, value):
        self.param.ca_lambda_min = value

    @property
    def ca_lambda_max(self):
        return self.param.ca_lambda_max

    @ca_lambda_max.setter
    def ca_lambda_max(self, value):
        self.param.ca_lambda_max = value

    @property
    def ca_basis_precondition(self):
        return self.param.ca_basis_precondition

    @ca_basis_precondition.setter
    def ca_basis_precondition(self, value):
        self.param.ca_basis_precondition = value

    @property
    def ca_lambda_min_precondition(self):
        return self.param.ca_lambda_min_precondition

    @ca_lambda_min_precondition.setter
    def ca_lambda_min_precondition(self, value):
        self.param.ca_lambda_min_precondition = value

    @property
    def ca_lambda_max_precondition(self):
        return self.param.ca_lambda_max_precondition

    @ca_lambda_max_precondition.setter
    def ca_lambda_max_precondition(self, value):
        self.param.ca_lambda_max_precondition = value

    @property
    def precondition_cycle(self):
        return self.param.precondition_cycle

    @precondition_cycle.setter
    def precondition_cycle(self, value):
        self.param.precondition_cycle = value

    @property
    def schwarz_type(self):
        return self.param.schwarz_type

    @schwarz_type.setter
    def schwarz_type(self, value):
        self.param.schwarz_type = value

    @property
    def accelerator_type_precondition(self):
        return self.param.accelerator_type_precondition

    @accelerator_type_precondition.setter
    def accelerator_type_precondition(self, value):
        self.param.accelerator_type_precondition = value

    @property
    def madwf_diagonal_suppressor(self):
        return self.param.madwf_diagonal_suppressor

    @madwf_diagonal_suppressor.setter
    def madwf_diagonal_suppressor(self, value):
        self.param.madwf_diagonal_suppressor = value

    @property
    def madwf_ls(self):
        return self.param.madwf_ls

    @madwf_ls.setter
    def madwf_ls(self, value):
        self.param.madwf_ls = value

    @property
    def madwf_null_miniter(self):
        return self.param.madwf_null_miniter

    @madwf_null_miniter.setter
    def madwf_null_miniter(self, value):
        self.param.madwf_null_miniter = value

    @property
    def madwf_null_tol(self):
        return self.param.madwf_null_tol

    @madwf_null_tol.setter
    def madwf_null_tol(self, value):
        self.param.madwf_null_tol = value

    @property
    def madwf_train_maxiter(self):
        return self.param.madwf_train_maxiter

    @madwf_train_maxiter.setter
    def madwf_train_maxiter(self, value):
        self.param.madwf_train_maxiter = value

    @property
    def madwf_param_load(self):
        return self.param.madwf_param_load

    @madwf_param_load.setter
    def madwf_param_load(self, value):
        self.param.madwf_param_load = value

    @property
    def madwf_param_save(self):
        return self.param.madwf_param_save

    @madwf_param_save.setter
    def madwf_param_save(self, value):
        self.param.madwf_param_save = value

    @property
    def madwf_param_infile(self):
        return self.param.madwf_param_infile

    @madwf_param_infile.setter
    def madwf_param_infile(self, value):
        self.param.madwf_param_infile = value

    @property
    def madwf_param_outfile(self):
        return self.param.madwf_param_outfile

    @madwf_param_outfile.setter
    def madwf_param_outfile(self, value):
        self.param.madwf_param_outfile = value

    @property
    def residual_type(self):
        return self.param.residual_type

    @residual_type.setter
    def residual_type(self, value):
        self.param.residual_type = value

    @property
    def cuda_prec_ritz(self):
        return self.param.cuda_prec_ritz

    @cuda_prec_ritz.setter
    def cuda_prec_ritz(self, value):
        self.param.cuda_prec_ritz = value

    @property
    def n_ev(self):
        return self.param.n_ev

    @n_ev.setter
    def n_ev(self, value):
        self.param.n_ev = value

    @property
    def max_search_dim(self):
        return self.param.max_search_dim

    @max_search_dim.setter
    def max_search_dim(self, value):
        self.param.max_search_dim = value

    @property
    def rhs_idx(self):
        return self.param.rhs_idx

    @rhs_idx.setter
    def rhs_idx(self, value):
        self.param.rhs_idx = value

    @property
    def deflation_grid(self):
        return self.param.deflation_grid

    @deflation_grid.setter
    def deflation_grid(self, value):
        self.param.deflation_grid = value

    @property
    def eigenval_tol(self):
        return self.param.eigenval_tol

    @eigenval_tol.setter
    def eigenval_tol(self, value):
        self.param.eigenval_tol = value

    @property
    def eigcg_max_restarts(self):
        return self.param.eigcg_max_restarts

    @eigcg_max_restarts.setter
    def eigcg_max_restarts(self, value):
        self.param.eigcg_max_restarts = value

    @property
    def max_restart_num(self):
        return self.param.max_restart_num

    @max_restart_num.setter
    def max_restart_num(self, value):
        self.param.max_restart_num = value

    @property
    def inc_tol(self):
        return self.param.inc_tol

    @inc_tol.setter
    def inc_tol(self, value):
        self.param.inc_tol = value

    @property
    def make_resident_solution(self):
        return self.param.make_resident_solution

    @make_resident_solution.setter
    def make_resident_solution(self, value):
        self.param.make_resident_solution = value

    @property
    def use_resident_solution(self):
        return self.param.use_resident_solution

    @use_resident_solution.setter
    def use_resident_solution(self, value):
        self.param.use_resident_solution = value

    @property
    def chrono_make_resident(self):
        return self.param.chrono_make_resident

    @chrono_make_resident.setter
    def chrono_make_resident(self, value):
        self.param.chrono_make_resident = value

    @property
    def chrono_replace_last(self):
        return self.param.chrono_replace_last

    @chrono_replace_last.setter
    def chrono_replace_last(self, value):
        self.param.chrono_replace_last = value

    @property
    def chrono_use_resident(self):
        return self.param.chrono_use_resident

    @chrono_use_resident.setter
    def chrono_use_resident(self, value):
        self.param.chrono_use_resident = value

    @property
    def chrono_max_dim(self):
        return self.param.chrono_max_dim

    @chrono_max_dim.setter
    def chrono_max_dim(self, value):
        self.param.chrono_max_dim = value

    @property
    def chrono_index(self):
        return self.param.chrono_index

    @chrono_index.setter
    def chrono_index(self, value):
        self.param.chrono_index = value

    @property
    def chrono_precision(self):
        return self.param.chrono_precision

    @chrono_precision.setter
    def chrono_precision(self, value):
        self.param.chrono_precision = value

    @property
    def extlib_type(self):
        return self.param.extlib_type

    @extlib_type.setter
    def extlib_type(self, value):
        self.param.extlib_type = value

    @property
    def native_blas_lapack(self):
        return self.param.native_blas_lapack

    @native_blas_lapack.setter
    def native_blas_lapack(self, value):
        self.param.native_blas_lapack = value

    @property
    def use_mobius_fused_kernel(self):
        return self.param.use_mobius_fused_kernel

    @use_mobius_fused_kernel.setter
    def use_mobius_fused_kernel(self, value):
        self.param.use_mobius_fused_kernel = value

cdef class QudaMultigridParam:
    cdef quda.QudaMultigridParam param

    def __init__(self):
        self.param = quda.newQudaMultigridParam()

    def __repr__(self):
        buf = io.BytesIO()
        with redirect_stdout(buf):
            quda.printQudaMultigridParam(&self.param)
        ret = buf.getvalue().decode("utf-8")
        return ret

    cdef from_ptr(self, quda.QudaMultigridParam *ptr):
        self.param = cython.operator.dereference(ptr)

    @property
    def struct_size(self):
        return self.param.struct_size

    @struct_size.setter
    def struct_size(self, value):
        self.param.struct_size = value

    @property
    def invert_param(self):
        param = QudaInvertParam()
        param.from_ptr(self.param.invert_param)
        return param

    @invert_param.setter
    def invert_param(self, value):
        self.set_invert_param(value)

    cdef set_invert_param(self, QudaInvertParam value):
        self.param.invert_param = &value.param

    @property
    def eig_param(self):
        params = []
        for i in range(self.param.n_level):
            param = QudaEigParam()
            param.from_ptr(self.param.eig_param[i])
            params.append(param)
        return params

    @eig_param.setter
    def eig_param(self, value):
        for i in range(self.param.n_level):
            self.set_eig_param(value[i], i)

    cdef set_eig_param(self, QudaEigParam value, int i):
        self.param.eig_param[i] = &value.param

    @property
    def n_level(self):
        return self.param.n_level

    @n_level.setter
    def n_level(self, value):
        self.param.n_level = value

    @property
    def geo_block_size(self):
        size = []
        for i in range(self.n_level):
            size.append(self.param.geo_block_size[i])
        return size

    @geo_block_size.setter
    def geo_block_size(self, value):
        for i in range(self.n_level):
            self.param.geo_block_size[i] = value[i]

    @property
    def spin_block_size(self):
        return self.param.spin_block_size

    @spin_block_size.setter
    def spin_block_size(self, value):
        self.param.spin_block_size = value

    @property
    def n_vec(self):
        return self.param.n_vec

    @n_vec.setter
    def n_vec(self, value):
        self.param.n_vec = value

    @property
    def precision_null(self):
        return self.param.precision_null

    @precision_null.setter
    def precision_null(self, value):
        self.param.precision_null = value

    @property
    def n_block_ortho(self):
        return self.param.n_block_ortho

    @n_block_ortho.setter
    def n_block_ortho(self, value):
        self.param.n_block_ortho = value

    @property
    def block_ortho_two_pass(self):
        return self.param.block_ortho_two_pass

    @block_ortho_two_pass.setter
    def block_ortho_two_pass(self, value):
        self.param.block_ortho_two_pass = value

    @property
    def verbosity(self):
        return self.param.verbosity

    @verbosity.setter
    def verbosity(self, value):
        self.param.verbosity = value

    @property
    def setup_use_mma(self):
        return self.param.setup_use_mma

    @setup_use_mma.setter
    def setup_use_mma(self, value):
        self.param.setup_use_mma = value

    @property
    def dslash_use_mma(self):
        return self.param.dslash_use_mma

    @dslash_use_mma.setter
    def dslash_use_mma(self, value):
        self.param.dslash_use_mma = value

    @property
    def setup_inv_type(self):
        return self.param.setup_inv_type

    @setup_inv_type.setter
    def setup_inv_type(self, value):
        self.param.setup_inv_type = value

    @property
    def num_setup_iter(self):
        return self.param.num_setup_iter

    @num_setup_iter.setter
    def num_setup_iter(self, value):
        self.param.num_setup_iter = value

    @property
    def setup_tol(self):
        return self.param.setup_tol

    @setup_tol.setter
    def setup_tol(self, value):
        self.param.setup_tol = value

    @property
    def setup_maxiter(self):
        return self.param.setup_maxiter

    @setup_maxiter.setter
    def setup_maxiter(self, value):
        self.param.setup_maxiter = value

    @property
    def setup_maxiter_refresh(self):
        return self.param.setup_maxiter_refresh

    @setup_maxiter_refresh.setter
    def setup_maxiter_refresh(self, value):
        self.param.setup_maxiter_refresh = value

    @property
    def setup_ca_basis(self):
        return self.param.setup_ca_basis

    @setup_ca_basis.setter
    def setup_ca_basis(self, value):
        self.param.setup_ca_basis = value

    @property
    def setup_ca_basis_size(self):
        return self.param.setup_ca_basis_size

    @setup_ca_basis_size.setter
    def setup_ca_basis_size(self, value):
        self.param.setup_ca_basis_size = value

    @property
    def setup_ca_lambda_min(self):
        return self.param.setup_ca_lambda_min

    @setup_ca_lambda_min.setter
    def setup_ca_lambda_min(self, value):
        self.param.setup_ca_lambda_min = value

    @property
    def setup_ca_lambda_max(self):
        return self.param.setup_ca_lambda_max

    @setup_ca_lambda_max.setter
    def setup_ca_lambda_max(self, value):
        self.param.setup_ca_lambda_max = value

    @property
    def setup_type(self):
        return self.param.setup_type

    @setup_type.setter
    def setup_type(self, value):
        self.param.setup_type = value

    @property
    def pre_orthonormalize(self):
        return self.param.pre_orthonormalize

    @pre_orthonormalize.setter
    def pre_orthonormalize(self, value):
        self.param.pre_orthonormalize = value

    @property
    def post_orthonormalize(self):
        return self.param.post_orthonormalize

    @post_orthonormalize.setter
    def post_orthonormalize(self, value):
        self.param.post_orthonormalize = value

    @property
    def coarse_solver(self):
        return self.param.coarse_solver

    @coarse_solver.setter
    def coarse_solver(self, value):
        self.param.coarse_solver = value

    @property
    def coarse_solver_tol(self):
        return self.param.coarse_solver_tol

    @coarse_solver_tol.setter
    def coarse_solver_tol(self, value):
        self.param.coarse_solver_tol = value

    @property
    def coarse_solver_maxiter(self):
        return self.param.coarse_solver_maxiter

    @coarse_solver_maxiter.setter
    def coarse_solver_maxiter(self, value):
        self.param.coarse_solver_maxiter = value

    @property
    def coarse_solver_ca_basis(self):
        return self.param.coarse_solver_ca_basis

    @coarse_solver_ca_basis.setter
    def coarse_solver_ca_basis(self, value):
        self.param.coarse_solver_ca_basis = value

    @property
    def coarse_solver_ca_basis_size(self):
        return self.param.coarse_solver_ca_basis_size

    @coarse_solver_ca_basis_size.setter
    def coarse_solver_ca_basis_size(self, value):
        self.param.coarse_solver_ca_basis_size = value

    @property
    def coarse_solver_ca_lambda_min(self):
        return self.param.coarse_solver_ca_lambda_min

    @coarse_solver_ca_lambda_min.setter
    def coarse_solver_ca_lambda_min(self, value):
        self.param.coarse_solver_ca_lambda_min = value

    @property
    def coarse_solver_ca_lambda_max(self):
        return self.param.coarse_solver_ca_lambda_max

    @coarse_solver_ca_lambda_max.setter
    def coarse_solver_ca_lambda_max(self, value):
        self.param.coarse_solver_ca_lambda_max = value

    @property
    def smoother(self):
        return self.param.smoother

    @smoother.setter
    def smoother(self, value):
        self.param.smoother = value

    @property
    def smoother_tol(self):
        return self.param.smoother_tol

    @smoother_tol.setter
    def smoother_tol(self, value):
        self.param.smoother_tol = value

    @property
    def nu_pre(self):
        return self.param.nu_pre

    @nu_pre.setter
    def nu_pre(self, value):
        self.param.nu_pre = value

    @property
    def nu_post(self):
        return self.param.nu_post

    @nu_post.setter
    def nu_post(self, value):
        self.param.nu_post = value

    @property
    def smoother_solver_ca_basis(self):
        return self.param.smoother_solver_ca_basis

    @smoother_solver_ca_basis.setter
    def smoother_solver_ca_basis(self, value):
        self.param.smoother_solver_ca_basis = value

    @property
    def smoother_solver_ca_lambda_min(self):
        return self.param.smoother_solver_ca_lambda_min

    @smoother_solver_ca_lambda_min.setter
    def smoother_solver_ca_lambda_min(self, value):
        self.param.smoother_solver_ca_lambda_min = value

    @property
    def smoother_solver_ca_lambda_max(self):
        return self.param.smoother_solver_ca_lambda_max

    @smoother_solver_ca_lambda_max.setter
    def smoother_solver_ca_lambda_max(self, value):
        self.param.smoother_solver_ca_lambda_max = value

    @property
    def omega(self):
        return self.param.omega

    @omega.setter
    def omega(self, value):
        self.param.omega = value

    @property
    def smoother_halo_precision(self):
        return self.param.smoother_halo_precision

    @smoother_halo_precision.setter
    def smoother_halo_precision(self, value):
        self.param.smoother_halo_precision = value

    @property
    def smoother_schwarz_type(self):
        return self.param.smoother_schwarz_type

    @smoother_schwarz_type.setter
    def smoother_schwarz_type(self, value):
        self.param.smoother_schwarz_type = value

    @property
    def smoother_schwarz_cycle(self):
        return self.param.smoother_schwarz_cycle

    @smoother_schwarz_cycle.setter
    def smoother_schwarz_cycle(self, value):
        self.param.smoother_schwarz_cycle = value

    @property
    def coarse_grid_solution_type(self):
        return self.param.coarse_grid_solution_type

    @coarse_grid_solution_type.setter
    def coarse_grid_solution_type(self, value):
        self.param.coarse_grid_solution_type = value

    @property
    def smoother_solve_type(self):
        return self.param.smoother_solve_type

    @smoother_solve_type.setter
    def smoother_solve_type(self, value):
        self.param.smoother_solve_type = value

    @property
    def cycle_type(self):
        return self.param.cycle_type

    @cycle_type.setter
    def cycle_type(self, value):
        self.param.cycle_type = value

    @property
    def global_reduction(self):
        return self.param.global_reduction

    @global_reduction.setter
    def global_reduction(self, value):
        self.param.global_reduction = value

    @property
    def location(self):
        return self.param.location

    @location.setter
    def location(self, value):
        self.param.location = value

    @property
    def setup_location(self):
        return self.param.setup_location

    @setup_location.setter
    def setup_location(self, value):
        self.param.setup_location = value

    @property
    def use_eig_solver(self):
        return self.param.use_eig_solver

    @use_eig_solver.setter
    def use_eig_solver(self, value):
        self.param.use_eig_solver = value

    @property
    def setup_minimize_memory(self):
        return self.param.setup_minimize_memory

    @setup_minimize_memory.setter
    def setup_minimize_memory(self, value):
        self.param.setup_minimize_memory = value

    @property
    def compute_null_vector(self):
        return self.param.compute_null_vector

    @compute_null_vector.setter
    def compute_null_vector(self, value):
        self.param.compute_null_vector = value

    @property
    def generate_all_levels(self):
        return self.param.generate_all_levels

    @generate_all_levels.setter
    def generate_all_levels(self, value):
        self.param.generate_all_levels = value

    @property
    def run_verify(self):
        return self.param.run_verify

    @run_verify.setter
    def run_verify(self, value):
        self.param.run_verify = value

    @property
    def run_low_mode_check(self):
        return self.param.run_low_mode_check

    @run_low_mode_check.setter
    def run_low_mode_check(self, value):
        self.param.run_low_mode_check = value

    @property
    def run_oblique_proj_check(self):
        return self.param.run_oblique_proj_check

    @run_oblique_proj_check.setter
    def run_oblique_proj_check(self, value):
        self.param.run_oblique_proj_check = value

    @property
    def vec_load(self):
        return self.param.vec_load

    @vec_load.setter
    def vec_load(self, value):
        self.param.vec_load = value

    @property
    def vec_infile(self):
        return self.param.vec_infile

    @vec_infile.setter
    def vec_infile(self, value):
        self.param.vec_infile = value

    @property
    def vec_store(self):
        return self.param.vec_store

    @vec_store.setter
    def vec_store(self, value):
        self.param.vec_store = value

    @property
    def vec_outfile(self):
        return self.param.vec_outfile

    @vec_outfile.setter
    def vec_outfile(self, value):
        self.param.vec_outfile = value

    @property
    def mg_vec_partfile(self):
        return self.param.mg_vec_partfile

    @mg_vec_partfile.setter
    def mg_vec_partfile(self, value):
        self.param.mg_vec_partfile = value

    @property
    def coarse_guess(self):
        return self.param.coarse_guess

    @coarse_guess.setter
    def coarse_guess(self, value):
        self.param.coarse_guess = value

    @property
    def preserve_deflation(self):
        return self.param.preserve_deflation

    @preserve_deflation.setter
    def preserve_deflation(self, value):
        self.param.preserve_deflation = value

    @property
    def gflops(self):
        return self.param.gflops

    @gflops.setter
    def gflops(self, value):
        self.param.gflops = value

    @property
    def secs(self):
        return self.param.secs

    @secs.setter
    def secs(self, value):
        self.param.secs = value

    @property
    def mu_factor(self):
        return self.param.mu_factor

    @mu_factor.setter
    def mu_factor(self, value):
        self.param.mu_factor = value

    @property
    def transfer_type(self):
        return self.param.transfer_type

    @transfer_type.setter
    def transfer_type(self, value):
        self.param.transfer_type = value

    @property
    def allow_truncation(self):
        return self.param.allow_truncation

    @allow_truncation.setter
    def allow_truncation(self, value):
        self.param.allow_truncation = value

    @property
    def staggered_kd_dagger_approximation(self):
        return self.param.staggered_kd_dagger_approximation

    @staggered_kd_dagger_approximation.setter
    def staggered_kd_dagger_approximation(self, value):
        self.param.staggered_kd_dagger_approximation = value

    @property
    def thin_update_only(self):
        return self.param.thin_update_only

    @thin_update_only.setter
    def thin_update_only(self, value):
        self.param.thin_update_only = value

cdef class QudaEigParam:
    cdef quda.QudaEigParam param

    def __init__(self):
        self.param = quda.newQudaEigParam()

    def __repr__(self):
        buf = io.BytesIO()
        with redirect_stdout(buf):
            quda.printQudaEigParam(&self.param)
        ret = buf.getvalue().decode("utf-8")
        return ret

    cdef from_ptr(self, quda.QudaEigParam *ptr):
        self.param = cython.operator.dereference(ptr)

    @property
    def struct_size(self):
        return self.param.struct_size

    @struct_size.setter
    def struct_size(self, value):
        self.param.struct_size = value

    @property
    def invert_param(self):
        param = QudaInvertParam()
        param.from_ptr(self.param.invert_param)
        return param

    @invert_param.setter
    def invert_param(self, value):
        self.set_invert_param(value)

    cdef set_invert_param(self, QudaInvertParam value):
        self.param.invert_param = &value.param

    @property
    def eig_type(self):
        return self.param.eig_type

    @eig_type.setter
    def eig_type(self, value):
        self.param.eig_type = value

    @property
    def use_poly_acc(self):
        return self.param.use_poly_acc

    @use_poly_acc.setter
    def use_poly_acc(self, value):
        self.param.use_poly_acc = value

    @property
    def poly_deg(self):
        return self.param.poly_deg

    @poly_deg.setter
    def poly_deg(self, value):
        self.param.poly_deg = value

    @property
    def a_min(self):
        return self.param.a_min

    @a_min.setter
    def a_min(self, value):
        self.param.a_min = value

    @property
    def a_max(self):
        return self.param.a_max

    @a_max.setter
    def a_max(self, value):
        self.param.a_max = value

    @property
    def preserve_deflation(self):
        return self.param.preserve_deflation

    @preserve_deflation.setter
    def preserve_deflation(self, value):
        self.param.preserve_deflation = value

    @property
    def preserve_deflation_space(self):
        ptr = Pointer("void")
        ptr.set_ptr(self.param.preserve_deflation_space)
        return ptr

    @preserve_deflation_space.setter
    def preserve_deflation_space(self, value):
        self.set_preserve_deflation_space(value)

    cdef set_preserve_deflation_space(self, Pointer value):
        assert value.dtype == "void"
        self.param.preserve_deflation_space = value.ptr

    @property
    def preserve_evals(self):
        return self.param.preserve_evals

    @preserve_evals.setter
    def preserve_evals(self, value):
        self.param.preserve_evals = value

    @property
    def use_dagger(self):
        return self.param.use_dagger

    @use_dagger.setter
    def use_dagger(self, value):
        self.param.use_dagger = value

    @property
    def use_norm_op(self):
        return self.param.use_norm_op

    @use_norm_op.setter
    def use_norm_op(self, value):
        self.param.use_norm_op = value

    @property
    def use_pc(self):
        return self.param.use_pc

    @use_pc.setter
    def use_pc(self, value):
        self.param.use_pc = value

    @property
    def use_eigen_qr(self):
        return self.param.use_eigen_qr

    @use_eigen_qr.setter
    def use_eigen_qr(self, value):
        self.param.use_eigen_qr = value

    @property
    def compute_svd(self):
        return self.param.compute_svd

    @compute_svd.setter
    def compute_svd(self, value):
        self.param.compute_svd = value

    @property
    def compute_gamma5(self):
        return self.param.compute_gamma5

    @compute_gamma5.setter
    def compute_gamma5(self, value):
        self.param.compute_gamma5 = value

    @property
    def require_convergence(self):
        return self.param.require_convergence

    @require_convergence.setter
    def require_convergence(self, value):
        self.param.require_convergence = value

    @property
    def spectrum(self):
        return self.param.spectrum

    @spectrum.setter
    def spectrum(self, value):
        self.param.spectrum = value

    @property
    def n_ev(self):
        return self.param.n_ev

    @n_ev.setter
    def n_ev(self, value):
        self.param.n_ev = value

    @property
    def n_kr(self):
        return self.param.n_kr

    @n_kr.setter
    def n_kr(self, value):
        self.param.n_kr = value

    @property
    def nLockedMax(self):
        return self.param.nLockedMax

    @nLockedMax.setter
    def nLockedMax(self, value):
        self.param.nLockedMax = value

    @property
    def n_conv(self):
        return self.param.n_conv

    @n_conv.setter
    def n_conv(self, value):
        self.param.n_conv = value

    @property
    def n_ev_deflate(self):
        return self.param.n_ev_deflate

    @n_ev_deflate.setter
    def n_ev_deflate(self, value):
        self.param.n_ev_deflate = value

    @property
    def tol(self):
        return self.param.tol

    @tol.setter
    def tol(self, value):
        self.param.tol = value

    @property
    def qr_tol(self):
        return self.param.qr_tol

    @qr_tol.setter
    def qr_tol(self, value):
        self.param.qr_tol = value

    @property
    def check_interval(self):
        return self.param.check_interval

    @check_interval.setter
    def check_interval(self, value):
        self.param.check_interval = value

    @property
    def max_restarts(self):
        return self.param.max_restarts

    @max_restarts.setter
    def max_restarts(self, value):
        self.param.max_restarts = value

    @property
    def batched_rotate(self):
        return self.param.batched_rotate

    @batched_rotate.setter
    def batched_rotate(self, value):
        self.param.batched_rotate = value

    @property
    def block_size(self):
        return self.param.block_size

    @block_size.setter
    def block_size(self, value):
        self.param.block_size = value

    @property
    def max_ortho_attempts(self):
        return self.param.max_ortho_attempts

    @max_ortho_attempts.setter
    def max_ortho_attempts(self, value):
        self.param.max_ortho_attempts = value

    @property
    def ortho_block_size(self):
        return self.param.ortho_block_size

    @ortho_block_size.setter
    def ortho_block_size(self, value):
        self.param.ortho_block_size = value

    @property
    def arpack_check(self):
        return self.param.arpack_check

    @arpack_check.setter
    def arpack_check(self, value):
        self.param.arpack_check = value

    @property
    def arpack_logfile(self):
        return self.param.arpack_logfile

    @arpack_logfile.setter
    def arpack_logfile(self, value):
        self.param.arpack_logfile = value

    @property
    def QUDA_logfile(self):
        return self.param.QUDA_logfile

    @QUDA_logfile.setter
    def QUDA_logfile(self, value):
        self.param.QUDA_logfile = value

    @property
    def nk(self):
        return self.param.nk

    @nk.setter
    def nk(self, value):
        self.param.nk = value

    @property
    def np(self):
        return self.param.np

    @np.setter
    def np(self, value):
        self.param.np = value

    @property
    def import_vectors(self):
        return self.param.import_vectors

    @import_vectors.setter
    def import_vectors(self, value):
        self.param.import_vectors = value

    @property
    def cuda_prec_ritz(self):
        return self.param.cuda_prec_ritz

    @cuda_prec_ritz.setter
    def cuda_prec_ritz(self, value):
        self.param.cuda_prec_ritz = value

    @property
    def mem_type_ritz(self):
        return self.param.mem_type_ritz

    @mem_type_ritz.setter
    def mem_type_ritz(self, value):
        self.param.mem_type_ritz = value

    @property
    def location(self):
        return self.param.location

    @location.setter
    def location(self, value):
        self.param.location = value

    @property
    def run_verify(self):
        return self.param.run_verify

    @run_verify.setter
    def run_verify(self, value):
        self.param.run_verify = value

    @property
    def vec_infile(self):
        return self.param.vec_infile

    @vec_infile.setter
    def vec_infile(self, value):
        self.param.vec_infile = value

    @property
    def vec_outfile(self):
        return self.param.vec_outfile

    @vec_outfile.setter
    def vec_outfile(self, value):
        self.param.vec_outfile = value

    @property
    def save_prec(self):
        return self.param.save_prec

    @save_prec.setter
    def save_prec(self, value):
        self.param.save_prec = value

    @property
    def io_parity_inflate(self):
        return self.param.io_parity_inflate

    @io_parity_inflate.setter
    def io_parity_inflate(self, value):
        self.param.io_parity_inflate = value

    @property
    def partfile(self):
        return self.param.partfile

    @partfile.setter
    def partfile(self, value):
        self.param.partfile = value

    @property
    def gflops(self):
        return self.param.gflops

    @gflops.setter
    def gflops(self, value):
        self.param.gflops = value

    @property
    def secs(self):
        return self.param.secs

    @secs.setter
    def secs(self, value):
        self.param.secs = value

    @property
    def extlib_type(self):
        return self.param.extlib_type

    @extlib_type.setter
    def extlib_type(self, value):
        self.param.extlib_type = value

cdef class QudaGaugeObservableParam:
    cdef quda.QudaGaugeObservableParam param

    def __init__(self):
        self.param = quda.newQudaGaugeObservableParam()

    def __repr__(self):
        buf = io.BytesIO()
        with redirect_stdout(buf):
            quda.printQudaGaugeObservableParam(&self.param)
        ret = buf.getvalue().decode("utf-8")
        return ret

    cdef from_ptr(self, quda.QudaGaugeObservableParam *ptr):
        self.param = cython.operator.dereference(ptr)

    @property
    def struct_size(self):
        return self.param.struct_size

    @struct_size.setter
    def struct_size(self, value):
        self.param.struct_size = value

    @property
    def su_project(self):
        return self.param.su_project

    @su_project.setter
    def su_project(self, value):
        self.param.su_project = value

    @property
    def compute_plaquette(self):
        return self.param.compute_plaquette

    @compute_plaquette.setter
    def compute_plaquette(self, value):
        self.param.compute_plaquette = value

    @property
    def plaquette(self):
        return self.param.plaquette

    @plaquette.setter
    def plaquette(self, value):
        self.param.plaquette = value

    @property
    def compute_polyakov_loop(self):
        return self.param.compute_polyakov_loop

    @compute_polyakov_loop.setter
    def compute_polyakov_loop(self, value):
        self.param.compute_polyakov_loop = value

    @property
    def ploop(self):
        return self.param.ploop

    @ploop.setter
    def ploop(self, value):
        self.param.ploop = value

    @property
    def compute_gauge_loop_trace(self):
        return self.param.compute_gauge_loop_trace

    @compute_gauge_loop_trace.setter
    def compute_gauge_loop_trace(self, value):
        self.param.compute_gauge_loop_trace = value

    @property
    def traces(self):
        ptr = Pointer("double_complex")
        ptr.set_ptr(self.param.traces)
        return ptr

    @traces.setter
    def traces(self, value):
        self.set_traces(value)

    cdef set_traces(self, Pointer value):
        assert value.dtype == "double_complex"
        self.param.traces = <double complex *>value.ptr

    @property
    def input_path_buff(self):
        ptr = Pointers("int", self.param.num_paths)
        ptr.set_ptrs(<void **>self.param.input_path_buff)
        return ptr

    @input_path_buff.setter
    def input_path_buff(self, value):
        self.set_input_path_buff(value)

    cdef set_input_path_buff(self, Pointers value):
        assert value.dtype == "int"
        self.param.input_path_buff = <int **>value.ptrs

    @property
    def path_length(self):
        ptr = Pointer("int")
        ptr.set_ptr(self.param.path_length)
        return ptr

    @path_length.setter
    def path_length(self, value):
        self.set_path_length(value)

    cdef set_path_length(self, Pointer value):
        assert value.dtype == "int"
        self.param.path_length = <int *>value.ptr

    @property
    def loop_coeff(self):
        ptr = Pointer("double")
        ptr.set_ptr(self.param.loop_coeff)
        return ptr

    @loop_coeff.setter
    def loop_coeff(self, value):
        self.set_loop_coeff(value)

    cdef set_loop_coeff(self, Pointer value):
        assert value.dtype == "double"
        self.param.loop_coeff = <double *>value.ptr

    @property
    def num_paths(self):
        return self.param.num_paths

    @num_paths.setter
    def num_paths(self, value):
        self.param.num_paths = value

    @property
    def max_length(self):
        return self.param.max_length

    @max_length.setter
    def max_length(self, value):
        self.param.max_length = value

    @property
    def factor(self):
        return self.param.factor

    @factor.setter
    def factor(self, value):
        self.param.factor = value

    @property
    def compute_qcharge(self):
        return self.param.compute_qcharge

    @compute_qcharge.setter
    def compute_qcharge(self, value):
        self.param.compute_qcharge = value

    @property
    def qcharge(self):
        return self.param.qcharge

    @qcharge.setter
    def qcharge(self, value):
        self.param.qcharge = value

    @property
    def energy(self):
        return self.param.energy

    @energy.setter
    def energy(self, value):
        self.param.energy = value

    @property
    def compute_qcharge_density(self):
        return self.param.compute_qcharge_density

    @compute_qcharge_density.setter
    def compute_qcharge_density(self, value):
        self.param.compute_qcharge_density = value

    @property
    def qcharge_density(self):
        ptr = Pointer("void")
        ptr.set_ptr(self.param.qcharge_density)
        return ptr

    @qcharge_density.setter
    def qcharge_density(self, value):
        self.set_qcharge_density(value)

    cdef set_qcharge_density(self, Pointer value):
        assert value.dtype == "void"
        self.param.qcharge_density = value.ptr

    @property
    def remove_staggered_phase(self):
        return self.param.remove_staggered_phase

    @remove_staggered_phase.setter
    def remove_staggered_phase(self, value):
        self.param.remove_staggered_phase = value

cdef class QudaGaugeSmearParam:
    cdef quda.QudaGaugeSmearParam param

    def __init__(self):
        self.param = quda.newQudaGaugeSmearParam()

    # def __repr__(self):
    #     buf = io.BytesIO()
    #     with redirect_stdout(buf):
    #         quda.printQudaGaugeSmearParam(&self.param)
    #     ret = buf.getvalue().decode("utf-8")
    #     return ret

    cdef from_ptr(self, quda.QudaGaugeSmearParam *ptr):
        self.param = cython.operator.dereference(ptr)

    @property
    def struct_size(self):
        return self.param.struct_size

    @struct_size.setter
    def struct_size(self, value):
        self.param.struct_size = value

    @property
    def n_steps(self):
        return self.param.n_steps

    @n_steps.setter
    def n_steps(self, value):
        self.param.n_steps = value

    @property
    def epsilon(self):
        return self.param.epsilon

    @epsilon.setter
    def epsilon(self, value):
        self.param.epsilon = value

    @property
    def alpha(self):
        return self.param.alpha

    @alpha.setter
    def alpha(self, value):
        self.param.alpha = value

    @property
    def rho(self):
        return self.param.rho

    @rho.setter
    def rho(self, value):
        self.param.rho = value

    @property
    def alpha1(self):
        return self.param.alpha1

    @alpha1.setter
    def alpha1(self, value):
        self.param.alpha1 = value

    @property
    def alpha2(self):
        return self.param.alpha2

    @alpha2.setter
    def alpha2(self, value):
        self.param.alpha2 = value

    @property
    def alpha3(self):
        return self.param.alpha3

    @alpha3.setter
    def alpha3(self, value):
        self.param.alpha3 = value

    @property
    def meas_interval(self):
        return self.param.meas_interval

    @meas_interval.setter
    def meas_interval(self, value):
        self.param.meas_interval = value

    @property
    def smear_type(self):
        return self.param.smear_type

    @smear_type.setter
    def smear_type(self, value):
        self.param.smear_type = value

cdef class QudaBLASParam:
    cdef quda.QudaBLASParam param

    def __init__(self):
        self.param = quda.newQudaBLASParam()

    def __repr__(self):
        buf = io.BytesIO()
        with redirect_stdout(buf):
            quda.printQudaBLASParam(&self.param)
        ret = buf.getvalue().decode("utf-8")
        return ret

    cdef from_ptr(self, quda.QudaBLASParam *ptr):
        self.param = cython.operator.dereference(ptr)

    @property
    def struct_size(self):
        return self.param.struct_size

    @struct_size.setter
    def struct_size(self, value):
        self.param.struct_size = value

    @property
    def blas_type(self):
        return self.param.blas_type

    @blas_type.setter
    def blas_type(self, value):
        self.param.blas_type = value

    @property
    def trans_a(self):
        return self.param.trans_a

    @trans_a.setter
    def trans_a(self, value):
        self.param.trans_a = value

    @property
    def trans_b(self):
        return self.param.trans_b

    @trans_b.setter
    def trans_b(self, value):
        self.param.trans_b = value

    @property
    def m(self):
        return self.param.m

    @m.setter
    def m(self, value):
        self.param.m = value

    @property
    def n(self):
        return self.param.n

    @n.setter
    def n(self, value):
        self.param.n = value

    @property
    def k(self):
        return self.param.k

    @k.setter
    def k(self, value):
        self.param.k = value

    @property
    def lda(self):
        return self.param.lda

    @lda.setter
    def lda(self, value):
        self.param.lda = value

    @property
    def ldb(self):
        return self.param.ldb

    @ldb.setter
    def ldb(self, value):
        self.param.ldb = value

    @property
    def ldc(self):
        return self.param.ldc

    @ldc.setter
    def ldc(self, value):
        self.param.ldc = value

    @property
    def a_offset(self):
        return self.param.a_offset

    @a_offset.setter
    def a_offset(self, value):
        self.param.a_offset = value

    @property
    def b_offset(self):
        return self.param.b_offset

    @b_offset.setter
    def b_offset(self, value):
        self.param.b_offset = value

    @property
    def c_offset(self):
        return self.param.c_offset

    @c_offset.setter
    def c_offset(self, value):
        self.param.c_offset = value

    @property
    def a_stride(self):
        return self.param.a_stride

    @a_stride.setter
    def a_stride(self, value):
        self.param.a_stride = value

    @property
    def b_stride(self):
        return self.param.b_stride

    @b_stride.setter
    def b_stride(self, value):
        self.param.b_stride = value

    @property
    def c_stride(self):
        return self.param.c_stride

    @c_stride.setter
    def c_stride(self, value):
        self.param.c_stride = value

    @property
    def alpha(self):
        return self.param.alpha

    @alpha.setter
    def alpha(self, value):
        self.param.alpha = value

    @property
    def beta(self):
        return self.param.beta

    @beta.setter
    def beta(self, value):
        self.param.beta = value

    @property
    def inv_mat_size(self):
        return self.param.inv_mat_size

    @inv_mat_size.setter
    def inv_mat_size(self, value):
        self.param.inv_mat_size = value

    @property
    def batch_count(self):
        return self.param.batch_count

    @batch_count.setter
    def batch_count(self, value):
        self.param.batch_count = value

    @property
    def data_type(self):
        return self.param.data_type

    @data_type.setter
    def data_type(self, value):
        self.param.data_type = value

    @property
    def data_order(self):
        return self.param.data_order

    @data_order.setter
    def data_order(self, value):
        self.param.data_order = value


def setVerbosityQuda(quda.QudaVerbosity verbosity, const char prefix[]):
    quda.setVerbosityQuda(verbosity, prefix, stdout)

def initCommsGridQuda(int nDim, list dims):
    assert nDim == 4 and len(dims) >= 4
    cdef int c_dims[4]
    c_dims = dims
    quda.initCommsGridQuda(nDim, c_dims, NULL, NULL)

def initQudaDevice(int device):
    quda.initQudaDevice(device)

def initQudaMemory():
    quda.initQudaMemory()

def initQuda(int device):
    quda.initQuda(device)

def endQuda():
    quda.endQuda()

def updateR():
    quda.updateR()

def loadGaugeQuda(Pointers h_gauge, QudaGaugeParam param):
    assert h_gauge.dtype == "void"
    quda.loadGaugeQuda(h_gauge.ptr, &param.param)

def freeGaugeQuda():
    quda.freeGaugeQuda()

def freeUniqueGaugeQuda(quda.QudaLinkType link_type):
    quda.freeUniqueGaugeQuda(link_type)

def freeGaugeSmearedQuda():
    quda.freeGaugeSmearedQuda()

def saveGaugeQuda(Pointers h_gauge, QudaGaugeParam param):
    assert h_gauge.dtype == "void"
    quda.saveGaugeQuda(h_gauge.ptr, &param.param)

def loadCloverQuda(Pointer h_clover, Pointer h_clovinv, QudaInvertParam inv_param):
    assert h_clover.dtype == "void"
    assert h_clovinv.dtype == "void"
    quda.loadCloverQuda(h_clover.ptr, h_clovinv.ptr, &inv_param.param)

def freeCloverQuda():
    quda.freeCloverQuda()

# def lanczosQuda(int k0, int m, Pointer hp_Apsi, Pointer hp_r, Pointer hp_V, Pointer hp_alpha, Pointer hp_beta, QudaEigParam eig_param)
# def eigensolveQuda(Pointers h_evecs, Pointer<double_complex> h_evals, QudaEigParam param)

def invertQuda(Pointer h_x, Pointer h_b, QudaInvertParam param):
    assert h_x.dtype == "void"
    assert h_b.dtype == "void"
    quda.invertQuda(h_x.ptr, h_b.ptr, &param.param)

# def invertMultiSrcQuda(Pointers _hp_x, Pointers _hp_b, QudaInvertParam param, Pointer h_gauge, QudaGaugeParam gauge_param)
# def invertMultiSrcStaggeredQuda(Pointers _hp_x, Pointers _hp_b, QudaInvertParam param, Pointer milc_fatlinks, Pointer milc_longlinks, QudaGaugeParam gauge_param)
# def invertMultiSrcCloverQuda(Pointers _hp_x, Pointers _hp_b, QudaInvertParam param, Pointer h_gauge, QudaGaugeParam gauge_param, Pointer h_clover, Pointer h_clovinv)

def invertMultiShiftQuda(Pointers _hp_x, Pointer _hp_b, QudaInvertParam param):
    assert _hp_x.dtype == "void"
    assert _hp_b.dtype == "void"
    quda.invertMultiShiftQuda(_hp_x.ptrs, _hp_b.ptr, &param.param)

def newMultigridQuda(QudaMultigridParam param) -> Pointer:
    mg_instance = Pointer("void")
    cdef void *ptr = quda.newMultigridQuda(&param.param)
    mg_instance.set_ptr(ptr)
    return mg_instance

def destroyMultigridQuda(Pointer mg_instance):
    quda.destroyMultigridQuda(mg_instance.ptr)

def updateMultigridQuda(Pointer mg_instance, QudaMultigridParam param):
    quda.updateMultigridQuda(mg_instance.ptr, &param.param)

def dumpMultigridQuda(Pointer mg_instance, QudaMultigridParam param):
    quda.dumpMultigridQuda(mg_instance.ptr, &param.param)

def dslashQuda(Pointer h_out, Pointer h_in, QudaInvertParam inv_param, quda.QudaParity parity):
    assert h_out.dtype == "void"
    assert h_in.dtype == "void"
    quda.dslashQuda(h_out.ptr, h_in.ptr, &inv_param.param, parity)

# def dslashMultiSrcQuda(Pointers _hp_x, Pointers _hp_b, QudaInvertParam param, QudaParity parity, Pointer h_gauge, QudaGaugeParam gauge_param)
# def dslashMultiSrcStaggeredQuda(Pointers _hp_x, Pointers _hp_b, QudaInvertParam param, QudaParity parity, Pointers milc_fatlinks, Pointers milc_longlinks, QudaGaugeParam gauge_param)
# def dslashMultiSrcCloverQuda(Pointers_hp_x, Pointers_hp_b, QudaInvertParam param, QudaParity parity, Pointer h_gauge, QudaGaugeParam gauge_param, Pointer h_clover, Pointer h_clovinv)

def cloverQuda(Pointer h_out, Pointer h_in, QudaInvertParam inv_param, quda.QudaParity parity, int inverse):
    assert h_out.dtype == "void"
    assert h_in.dtype == "void"
    quda.cloverQuda(h_out.ptr, h_in.ptr, &inv_param.param, parity, inverse)

def MatQuda(Pointer h_out, Pointer h_in, QudaInvertParam inv_param):
    assert h_out.dtype == "void"
    assert h_in.dtype == "void"
    quda.MatQuda(h_out.ptr, h_in.ptr, &inv_param.param)

def MatDagMatQuda(Pointer h_out, Pointer h_in, QudaInvertParam inv_param):
    assert h_out.dtype == "void"
    assert h_in.dtype == "void"
    quda.MatDagMatQuda(h_out.ptr, h_in.ptr, &inv_param.param)

# void set_dim(int *)
# void pack_ghost(void **cpuLink, void **cpuGhost, int nFace, QudaPrecision precision)

def computeKSLinkQuda(Pointers fatlink, Pointers longlink, Pointers ulink, Pointers inlink, Pointer path_coeff, QudaGaugeParam param):
    assert fatlink.dtype == "void"
    assert longlink.dtype == "void"
    assert ulink.dtype == "void"
    assert inlink.dtype == "void"
    assert path_coeff.dtype == "double"
    quda.computeKSLinkQuda(fatlink.ptr, longlink.ptr, ulink.ptr, inlink.ptr, <double *>path_coeff.ptr, &param.param)

# void computeTwoLinkQuda(void *twolink, void *inlink, QudaGaugeParam *param)

def momResidentQuda(Pointers mom, QudaGaugeParam param):
    assert mom.dtype == "void"
    quda.momResidentQuda(mom.ptr, &param.param)

def computeGaugeForceQuda(Pointers mom, Pointers sitelink, Pointer input_path_buf, Pointer path_length, Pointer loop_coeff, int num_paths, int max_length, double dt, QudaGaugeParam qudaGaugeParam):
    assert mom.dtype == "void"
    assert sitelink.dtype == "void"
    return quda.computeGaugeForceQuda(mom.ptr, sitelink.ptr, <int ***>input_path_buf.ptr, <int *>path_length.ptr, <double *>loop_coeff.ptr, num_paths, max_length, dt, &qudaGaugeParam.param)

def computeGaugePathQuda(Pointers out, Pointers sitelink, Pointer input_path_buf, Pointer path_length, Pointer loop_coeff, int num_paths, int max_length, double dt, QudaGaugeParam qudaGaugeParam):
    assert out.dtype == "void"
    assert sitelink.dtype == "void"
    return quda.computeGaugePathQuda(out.ptr, sitelink.ptr, <int ***>input_path_buf.ptr, <int *>path_length.ptr, <double *>loop_coeff.ptr, num_paths, max_length, dt, &qudaGaugeParam.param)

def computeGaugeLoopTraceQuda(Pointer traces, Pointers input_path_buf, Pointer path_length, Pointer loop_coeff, int num_paths, int max_length, double factor):
    assert traces.dtype == "double_complex"
    assert input_path_buf.dtype == "int"
    assert path_length.dtype == "int"
    assert loop_coeff.dtype == "double"
    quda.computeGaugeLoopTraceQuda(<double complex *>traces.ptr, <int **>input_path_buf.ptr, <int *>path_length.ptr, <double *>loop_coeff.ptr, num_paths, max_length, factor)


def updateGaugeFieldQuda(Pointers gauge, Pointers momentum, double dt, int conj_mom, int exact, QudaGaugeParam param):
    assert gauge.dtype == "void"
    assert momentum.dtype == "void"
    quda.updateGaugeFieldQuda(gauge.ptr, momentum.ptr, dt, conj_mom, exact, &param.param)

def staggeredPhaseQuda(Pointers gauge_h, QudaGaugeParam param):
    assert gauge_h.dtype == "void"
    quda.staggeredPhaseQuda(gauge_h.ptr, &param.param)

def projectSU3Quda(Pointers gauge_h, double tol, QudaGaugeParam param):
    assert gauge_h.dtype == "void"
    quda.projectSU3Quda(gauge_h.ptr, tol, &param.param)

def momActionQuda(Pointers momentum, QudaGaugeParam param):
    assert momentum.dtype == "void"
    return quda.momActionQuda(momentum.ptr, &param.param)

# void* createGaugeFieldQuda(void* gauge, int geometry, QudaGaugeParam* param)
# void saveGaugeFieldQuda(void* outGauge, void* inGauge, QudaGaugeParam* param)
# void destroyGaugeFieldQuda(void* gauge)

def createCloverQuda(QudaInvertParam param):
    quda.createCloverQuda(&param.param)

def computeCloverForceQuda(Pointers mom, double dt, Pointers x, Pointers p, Pointer coeff, double kappa2, double ck, int nvector, double multiplicity, Pointers gauge, QudaGaugeParam gauge_param, QudaInvertParam inv_param):
    assert mom.dtype == "void"
    assert x.dtype == "void"
    assert coeff.dtype == "double"
    quda.computeCloverForceQuda(mom.ptr, dt, x.ptrs, NULL, <double *>coeff.ptr, kappa2, ck, nvector, multiplicity, NULL, &gauge_param.param, &inv_param.param)

# void computeStaggeredForceQuda(void *mom, double dt, double delta, void *gauge, void **x, QudaGaugeParam *gauge_param, QudaInvertParam *invert_param)
# void computeHISQForceQuda(void* momentum, double dt, const double level2_coeff[6], const double fat7_coeff[6], const void* const w_link, const void* const v_link, const void* const u_link, void** quark, int num, int num_naik, double** coeff, QudaGaugeParam* param)

def gaussGaugeQuda(unsigned long long seed, double sigma):
    quda.gaussGaugeQuda(seed, sigma)

def gaussMomQuda(unsigned long long seed, double sigma):
    quda.gaussMomQuda(seed, sigma)

def plaqQuda(list plaq):
    assert len(plaq) >= 3
    cdef double c_plaq[3]
    quda.plaqQuda(c_plaq)
    for i in range(3):
        plaq[i] = c_plaq[i]

# void polyakovLoopQuda(double ploop[2], int dir)

# void copyExtendedResidentGaugeQuda(void *resident_gauge)

# void performWuppertalnStep(void *h_out, void *h_in, QudaInvertParam *param, unsigned int n_steps, double alpha)

def performGaugeSmearQuda(QudaGaugeSmearParam smear_param, QudaGaugeObservableParam obs_param):
    quda.performGaugeSmearQuda(&smear_param.param, &obs_param.param)

def performWFlowQuda(QudaGaugeSmearParam smear_param, QudaGaugeObservableParam obs_param):
    quda.performWFlowQuda(&smear_param.param, &obs_param.param)

def gaugeObservablesQuda(QudaGaugeObservableParam param):
    quda.gaugeObservablesQuda(&param.param)

# void contractQuda(const void *x, const void *y, void *result, const QudaContractType cType, QudaInvertParam *param,
#                     const int *X)

def computeGaugeFixingOVRQuda(Pointers gauge, unsigned int gauge_dir, unsigned int Nsteps, unsigned int verbose_interval, double relax_boost, double tolerance, unsigned int reunit_interval, unsigned int stopWtheta, QudaGaugeParam param, list timeinfo):
    assert len(timeinfo) >= 3
    assert gauge.dtype == "void"
    cdef double c_timeinfo[3]
    ret = quda.computeGaugeFixingOVRQuda(gauge.ptr, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta, &param.param, c_timeinfo)
    for i in range(3):
        timeinfo[i] = c_timeinfo[i]
    return ret

def computeGaugeFixingFFTQuda(Pointers gauge, unsigned int gauge_dir, unsigned int Nsteps, unsigned int verbose_interval, double alpha, unsigned int autotune, double tolerance, unsigned int stopWtheta, QudaGaugeParam param, list timeinfo):
    assert len(timeinfo) >= 3
    assert gauge.dtype == "void"
    cdef double c_timeinfo[3]
    ret = quda.computeGaugeFixingFFTQuda(gauge.ptr, gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta, &param.param, c_timeinfo)
    for i in range(3):
        timeinfo[i] = c_timeinfo[i]
    return ret

# void blasGEMMQuda(void *arrayA, void *arrayB, void *arrayC, QudaBoolean native, QudaBLASParam *param)
# void blasLUInvQuda(void *Ainv, void *A, QudaBoolean use_native, QudaBLASParam *param)

# void flushChronoQuda(int index)

# void* newDeflationQuda(QudaEigParam *param)
# void destroyDeflationQuda(void *df_instance)

cdef class QudaQuarkSmearParam:
    cdef quda.QudaQuarkSmearParam param

    def __init__(self):
        # self.param = quda.QudaQuarkSmearParam()
        pass

    # def __repr__(self):
    #     buf = io.BytesIO()
    #     with redirect_stdout(buf):
    #         quda.printQudaQuarkSmearParam(&self.param)
    #     ret = buf.getvalue().decode("utf-8")
    #     return ret

    cdef from_ptr(self, quda.QudaQuarkSmearParam *ptr):
        self.param = cython.operator.dereference(ptr)

    @property
    def inv_param(self):
        param = QudaInvertParam()
        param.from_ptr(self.param.inv_param)
        return param

    @inv_param.setter
    def inv_param(self, value):
        self.set_inv_param(value)

    cdef set_inv_param(self, QudaInvertParam value):
        self.param.inv_param = &value.param


    @property
    def n_steps(self):
        return self.param.n_steps

    @n_steps.setter
    def n_steps(self, value):
        self.param.n_steps = value

    @property
    def width(self):
        return self.param.width

    @width.setter
    def width(self, value):
        self.param.width = value

    @property
    def compute_2link(self):
        return self.param.compute_2link

    @compute_2link.setter
    def compute_2link(self, value):
        self.param.compute_2link = value

    @property
    def delete_2link(self):
        return self.param.delete_2link

    @delete_2link.setter
    def delete_2link(self, value):
        self.param.delete_2link = value

    @property
    def t0(self):
        return self.param.t0

    @t0.setter
    def t0(self, value):
        self.param.t0 = value

    @property
    def gflops(self):
        return self.param.gflops

    @gflops.setter
    def gflops(self, value):
        self.param.gflops = value

# void performTwoLinkGaussianSmearNStep(void *h_in, QudaQuarkSmearParam *smear_param)
