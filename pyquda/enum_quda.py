from enum import IntEnum

QUDA_INVALID_ENUM = -0x7FFFFFFF - 1

QUDA_VERSION_MAJOR = 1
QUDA_VERSION_MINOR = 1
QUDA_VERSION_SUBMINOR = 0

QUDA_VERSION = (QUDA_VERSION_MAJOR << 16) | (QUDA_VERSION_MINOR << 8) | QUDA_VERSION_SUBMINOR
"""
@def   QUDA_VERSION
@brief This macro is deprecated.  Use QUDA_VERSION_MAJOR, etc., instead.
"""

QUDA_MAX_DIM = 6
"""
@def   QUDA_MAX_DIM
@brief Maximum number of dimensions supported by QUDA.  In practice, no
routines make use of more than 5.
"""

QUDA_MAX_GEOMETRY = 8
"""
@def   QUDA_MAX_GEOMETRY
@brief Maximum geometry supported by a field.  This essentially is
the maximum number of dimensions supported per lattice site.
"""

QUDA_MAX_MULTI_SHIFT = 32
"""
@def QUDA_MAX_MULTI_SHIFT
@brief Maximum number of shifts supported by the multi-shift solver.
    This number may be changed if need be.
"""

QUDA_MAX_BLOCK_SRC = 64
"""
@def QUDA_MAX_BLOCK_SRC
@brief Maximum number of sources that can be supported by the block solver
"""

QUDA_MAX_ARRAY_SIZE = QUDA_MAX_MULTI_SHIFT if QUDA_MAX_MULTI_SHIFT > QUDA_MAX_BLOCK_SRC else QUDA_MAX_BLOCK_SRC
"""
@def QUDA_MAX_ARRAY
@brief Maximum array length used in QudaInvertParam arrays
"""

QUDA_MAX_DWF_LS = 32
"""
@def   QUDA_MAX_DWF_LS
@brief Maximum length of the Ls dimension for domain-wall fermions
"""

QUDA_MAX_MG_LEVEL = 5
"""
@def QUDA_MAX_MG_LEVEL
@brief Maximum number of multi-grid levels.  This number may be
increased if needed.
"""


class qudaError_t(IntEnum):
    QUDA_SUCCESS = 0
    QUDA_ERROR = 1
    QUDA_ERROR_UNINITIALIZED = 2


class QudaMemoryType(IntEnum):
    QUDA_MEMORY_DEVICE = 0
    QUDA_MEMORY_PINNED = 1
    QUDA_MEMORY_MAPPED = 2
    QUDA_MEMORY_INVALID = QUDA_INVALID_ENUM


#
# Types used in QudaGaugeParam
#


class QudaLinkType(IntEnum):
    QUDA_SU3_LINKS = 0
    QUDA_GENERAL_LINKS = 1
    QUDA_THREE_LINKS = 2
    QUDA_MOMENTUM_LINKS = 3
    QUDA_COARSE_LINKS = 4  # used for coarse-gauge field with multigrid
    QUDA_SMEARED_LINKS = 5  # used for loading and saving gaugeSmeared in the interface
    QUDA_WILSON_LINKS = QUDA_SU3_LINKS  # used by wilson, clover, twisted mass, and domain wall
    QUDA_ASQTAD_FAT_LINKS = QUDA_GENERAL_LINKS
    QUDA_ASQTAD_LONG_LINKS = QUDA_THREE_LINKS
    QUDA_ASQTAD_MOM_LINKS = QUDA_MOMENTUM_LINKS
    QUDA_ASQTAD_GENERAL_LINKS = QUDA_GENERAL_LINKS
    QUDA_INVALID_LINKS = QUDA_INVALID_ENUM


class QudaGaugeFieldOrder(IntEnum):
    QUDA_FLOAT_GAUGE_ORDER = 1
    QUDA_FLOAT2_GAUGE_ORDER = 2  # no reconstruct and double precision
    QUDA_FLOAT4_GAUGE_ORDER = 4  # 8 reconstruct single, and 12 reconstruct single, half, quarter
    QUDA_FLOAT8_GAUGE_ORDER = 8  # 8 reconstruct half and quarter
    QUDA_NATIVE_GAUGE_ORDER = 9  # used to denote one of the above types in a trait, not used directly
    QUDA_QDP_GAUGE_ORDER = 10  # expect *gauge[mu], even-odd, spacetime, row-column color
    QUDA_QDPJIT_GAUGE_ORDER = 11  # expect *gauge[mu], even-odd, complex-column-row-spacetime
    QUDA_CPS_WILSON_GAUGE_ORDER = 12  # expect *gauge, even-odd, mu, spacetime, column-row color
    QUDA_MILC_GAUGE_ORDER = 13  # expect *gauge, even-odd, mu, spacetime, row-column order
    QUDA_MILC_SITE_GAUGE_ORDER = 14  # packed into MILC site AoS [even-odd][spacetime] array, and [dir][row][col] inside
    QUDA_BQCD_GAUGE_ORDER = 15  # expect *gauge, mu, even-odd, spacetime+halos, column-row order
    QUDA_TIFR_GAUGE_ORDER = 16  # expect *gauge, mu, even-odd, spacetime, column-row order
    QUDA_TIFR_PADDED_GAUGE_ORDER = 17  # expect *gauge, mu, parity, t, z+halo, y, x/2, column-row order
    QUDA_INVALID_GAUGE_ORDER = QUDA_INVALID_ENUM


class QudaTboundary(IntEnum):
    QUDA_ANTI_PERIODIC_T = -1
    QUDA_PERIODIC_T = 1
    QUDA_INVALID_T_BOUNDARY = QUDA_INVALID_ENUM


class QudaPrecision(IntEnum):
    QUDA_QUARTER_PRECISION = 1
    QUDA_HALF_PRECISION = 2
    QUDA_SINGLE_PRECISION = 4
    QUDA_DOUBLE_PRECISION = 8
    QUDA_INVALID_PRECISION = QUDA_INVALID_ENUM


class QudaReconstructType(IntEnum):
    QUDA_RECONSTRUCT_NO = 18  # store all 18 real numbers explicitly
    QUDA_RECONSTRUCT_12 = 12  # reconstruct from 12 real numbers
    QUDA_RECONSTRUCT_8 = 8  # reconstruct from 8 real numbers
    QUDA_RECONSTRUCT_9 = 9  # used for storing HISQ long-link variables
    QUDA_RECONSTRUCT_13 = 13  # used for storing HISQ long-link variables
    QUDA_RECONSTRUCT_10 = 10  # 10-number parameterization used for storing the momentum field
    QUDA_RECONSTRUCT_INVALID = QUDA_INVALID_ENUM


class QudaGaugeFixed(IntEnum):
    QUDA_GAUGE_FIXED_NO = 0  # no gauge fixing
    QUDA_GAUGE_FIXED_YES = 1  # gauge field stored in temporal gauge
    QUDA_GAUGE_FIXED_INVALID = QUDA_INVALID_ENUM


#
# Types used in QudaInvertParam
#


class QudaDslashType(IntEnum):
    QUDA_WILSON_DSLASH = 0
    QUDA_CLOVER_WILSON_DSLASH = 1
    QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH = 2
    QUDA_DOMAIN_WALL_DSLASH = 3
    QUDA_DOMAIN_WALL_4D_DSLASH = 4
    QUDA_MOBIUS_DWF_DSLASH = 5
    QUDA_MOBIUS_DWF_EOFA_DSLASH = 6
    QUDA_STAGGERED_DSLASH = 7
    QUDA_ASQTAD_DSLASH = 8
    QUDA_TWISTED_MASS_DSLASH = 9
    QUDA_TWISTED_CLOVER_DSLASH = 10
    QUDA_LAPLACE_DSLASH = 11
    QUDA_COVDEV_DSLASH = 12
    QUDA_INVALID_DSLASH = QUDA_INVALID_ENUM


class QudaInverterType(IntEnum):
    QUDA_CG_INVERTER = 0
    QUDA_BICGSTAB_INVERTER = 1
    QUDA_GCR_INVERTER = 2
    QUDA_MR_INVERTER = 3
    QUDA_SD_INVERTER = 4
    QUDA_PCG_INVERTER = 5
    QUDA_EIGCG_INVERTER = 6
    QUDA_INC_EIGCG_INVERTER = 7
    QUDA_GMRESDR_INVERTER = 8
    QUDA_GMRESDR_PROJ_INVERTER = 9
    QUDA_GMRESDR_SH_INVERTER = 10
    QUDA_FGMRESDR_INVERTER = 11
    QUDA_MG_INVERTER = 12
    QUDA_BICGSTABL_INVERTER = 13
    QUDA_CGNE_INVERTER = 14
    QUDA_CGNR_INVERTER = 15
    QUDA_CG3_INVERTER = 16
    QUDA_CG3NE_INVERTER = 17
    QUDA_CG3NR_INVERTER = 18
    QUDA_CA_CG_INVERTER = 19
    QUDA_CA_CGNE_INVERTER = 20
    QUDA_CA_CGNR_INVERTER = 21
    QUDA_CA_GCR_INVERTER = 22
    QUDA_INVALID_INVERTER = QUDA_INVALID_ENUM


class QudaEigType(IntEnum):
    QUDA_EIG_TR_LANCZOS = 0  # Thick restarted lanczos solver
    QUDA_EIG_BLK_TR_LANCZOS = 1  # Block Thick restarted lanczos solver
    QUDA_EIG_IR_ARNOLDI = 2  # Implicitly Restarted Arnoldi solver
    QUDA_EIG_BLK_IR_ARNOLDI = 3  # Block Implicitly Restarted Arnoldi solver
    QUDA_EIG_INVALID = QUDA_INVALID_ENUM


# S=smallest L=largest
# R=real M=modulus I=imaniary
class QudaEigSpectrumType(IntEnum):
    QUDA_SPECTRUM_LM_EIG = 0
    QUDA_SPECTRUM_SM_EIG = 1
    QUDA_SPECTRUM_LR_EIG = 2
    QUDA_SPECTRUM_SR_EIG = 3
    QUDA_SPECTRUM_LI_EIG = 4
    QUDA_SPECTRUM_SI_EIG = 5
    QUDA_SPECTRUM_INVALID = QUDA_INVALID_ENUM


class QudaSolutionType(IntEnum):
    QUDA_MAT_SOLUTION = 0
    QUDA_MATDAG_MAT_SOLUTION = 1
    QUDA_MATPC_SOLUTION = 2
    QUDA_MATPC_DAG_SOLUTION = 3
    QUDA_MATPCDAG_MATPC_SOLUTION = 4
    QUDA_MATPCDAG_MATPC_SHIFT_SOLUTION = 5
    QUDA_INVALID_SOLUTION = QUDA_INVALID_ENUM


class QudaSolveType(IntEnum):
    QUDA_DIRECT_SOLVE = 0
    QUDA_NORMOP_SOLVE = 1
    QUDA_DIRECT_PC_SOLVE = 2
    QUDA_NORMOP_PC_SOLVE = 3
    QUDA_NORMERR_SOLVE = 4
    QUDA_NORMERR_PC_SOLVE = 5
    QUDA_NORMEQ_SOLVE = QUDA_NORMOP_SOLVE  # deprecated
    QUDA_NORMEQ_PC_SOLVE = QUDA_NORMOP_PC_SOLVE  # deprecated
    QUDA_INVALID_SOLVE = QUDA_INVALID_ENUM


class QudaMultigridCycleType(IntEnum):
    QUDA_MG_CYCLE_VCYCLE = 0
    QUDA_MG_CYCLE_FCYCLE = 1
    QUDA_MG_CYCLE_WCYCLE = 2
    QUDA_MG_CYCLE_RECURSIVE = 3
    QUDA_MG_CYCLE_INVALID = QUDA_INVALID_ENUM


class QudaSchwarzType(IntEnum):
    QUDA_ADDITIVE_SCHWARZ = 0
    QUDA_MULTIPLICATIVE_SCHWARZ = 1
    QUDA_INVALID_SCHWARZ = QUDA_INVALID_ENUM


class QudaAcceleratorType(IntEnum):
    QUDA_MADWF_ACCELERATOR = 0  # Use the MADWF accelerator
    QUDA_INVALID_ACCELERATOR = QUDA_INVALID_ENUM


class QudaResidualType(IntEnum):
    QUDA_L2_RELATIVE_RESIDUAL = 1  # L2 relative residual (default)
    QUDA_L2_ABSOLUTE_RESIDUAL = 2  # L2 absolute residual
    QUDA_HEAVY_QUARK_RESIDUAL = 4  # Fermilab heavy quark residual
    QUDA_INVALID_RESIDUAL = QUDA_INVALID_ENUM


# Which basis to use for CA algorithms
class QudaCABasis(IntEnum):
    QUDA_POWER_BASIS = 0
    QUDA_CHEBYSHEV_BASIS = 1
    QUDA_INVALID_BASIS = QUDA_INVALID_ENUM


# Whether the preconditioned matrix is (1-k^2 Deo Doe) or (1-k^2 Doe Deo)
#
# For the clover-improved Wilson Dirac operator, QUDA_MATPC_EVEN_EVEN
# defaults to the "symmetric" form, (1 - k^2 A_ee^-1 D_eo A_oo^-1 D_oe)
# and likewise for QUDA_MATPC_ODD_ODD.
#
# For the "asymmetric" form, (A_ee - k^2 D_eo A_oo^-1 D_oe), select
# QUDA_MATPC_EVEN_EVEN_ASYMMETRIC.
#
class QudaMatPCType(IntEnum):
    QUDA_MATPC_EVEN_EVEN = 0
    QUDA_MATPC_ODD_ODD = 1
    QUDA_MATPC_EVEN_EVEN_ASYMMETRIC = 2
    QUDA_MATPC_ODD_ODD_ASYMMETRIC = 3
    QUDA_MATPC_INVALID = QUDA_INVALID_ENUM


class QudaDagType(IntEnum):
    QUDA_DAG_NO = 0
    QUDA_DAG_YES = 1
    QUDA_DAG_INVALID = QUDA_INVALID_ENUM


class QudaMassNormalization(IntEnum):
    QUDA_KAPPA_NORMALIZATION = 0
    QUDA_MASS_NORMALIZATION = 1
    QUDA_ASYMMETRIC_MASS_NORMALIZATION = 2
    QUDA_INVALID_NORMALIZATION = QUDA_INVALID_ENUM


class QudaSolverNormalization(IntEnum):
    QUDA_DEFAULT_NORMALIZATION = 0  # leave source and solution untouched
    QUDA_SOURCE_NORMALIZATION = 1  # normalize such that || src || = 1


class QudaPreserveSource(IntEnum):
    QUDA_PRESERVE_SOURCE_NO = 0  # use the source for the residual
    QUDA_PRESERVE_SOURCE_YES = 1  # keep the source intact
    QUDA_PRESERVE_SOURCE_INVALID = QUDA_INVALID_ENUM


class QudaDiracFieldOrder(IntEnum):
    QUDA_INTERNAL_DIRAC_ORDER = 0  # internal dirac order used, varies on precision and dslash type
    QUDA_DIRAC_ORDER = 1  # even-odd, color inside spin
    QUDA_QDP_DIRAC_ORDER = 2  # even-odd, spin inside color
    QUDA_QDPJIT_DIRAC_ORDER = 3  # even-odd, complex-color-spin-spacetime
    QUDA_CPS_WILSON_DIRAC_ORDER = 4  # odd-even, color inside spin
    QUDA_LEX_DIRAC_ORDER = 5  # lexicographical order, color inside spin
    QUDA_TIFR_PADDED_DIRAC_ORDER = 6  # padded z dimension for TIFR RHMC code
    QUDA_INVALID_DIRAC_ORDER = QUDA_INVALID_ENUM


class QudaCloverFieldOrder(IntEnum):
    QUDA_FLOAT_CLOVER_ORDER = 1  # even-odd float ordering
    QUDA_FLOAT2_CLOVER_ORDER = 2  # even-odd float2 ordering
    QUDA_FLOAT4_CLOVER_ORDER = 4  # even-odd float4 ordering
    QUDA_FLOAT8_CLOVER_ORDER = 8  # even-odd float8 ordering
    QUDA_PACKED_CLOVER_ORDER = 9  # even-odd, QDP packed
    QUDA_QDPJIT_CLOVER_ORDER = 10  # (diagonal / off-diagonal)-chirality-spacetime
    QUDA_BQCD_CLOVER_ORDER = 11  # even-odd, super-diagonal packed and reordered
    QUDA_INVALID_CLOVER_ORDER = QUDA_INVALID_ENUM


class QudaVerbosity(IntEnum):
    QUDA_SILENT = 0
    QUDA_SUMMARIZE = 1
    QUDA_VERBOSE = 2
    QUDA_DEBUG_VERBOSE = 3
    QUDA_INVALID_VERBOSITY = QUDA_INVALID_ENUM


class QudaTune(IntEnum):
    QUDA_TUNE_NO = 0
    QUDA_TUNE_YES = 1
    QUDA_TUNE_INVALID = QUDA_INVALID_ENUM


class QudaPreserveDirac(IntEnum):
    QUDA_PRESERVE_DIRAC_NO = 0
    QUDA_PRESERVE_DIRAC_YES = 1
    QUDA_PRESERVE_DIRAC_INVALID = QUDA_INVALID_ENUM


#
# Type used for "parity" argument to dslashQuda()
#


class QudaParity(IntEnum):
    QUDA_EVEN_PARITY = 0
    QUDA_ODD_PARITY = 1
    QUDA_INVALID_PARITY = QUDA_INVALID_ENUM


#
# Types used only internally
#


class QudaDiracType(IntEnum):
    QUDA_WILSON_DIRAC = 0
    QUDA_WILSONPC_DIRAC = 1
    QUDA_CLOVER_DIRAC = 2
    QUDA_CLOVERPC_DIRAC = 3
    QUDA_CLOVER_HASENBUSCH_TWIST_DIRAC = 4
    QUDA_CLOVER_HASENBUSCH_TWISTPC_DIRAC = 5
    QUDA_DOMAIN_WALL_DIRAC = 6
    QUDA_DOMAIN_WALLPC_DIRAC = 7
    QUDA_DOMAIN_WALL_4D_DIRAC = 8
    QUDA_DOMAIN_WALL_4DPC_DIRAC = 9
    QUDA_MOBIUS_DOMAIN_WALL_DIRAC = 10
    QUDA_MOBIUS_DOMAIN_WALLPC_DIRAC = 11
    QUDA_MOBIUS_DOMAIN_WALL_EOFA_DIRAC = 12
    QUDA_MOBIUS_DOMAIN_WALLPC_EOFA_DIRAC = 13
    QUDA_STAGGERED_DIRAC = 14
    QUDA_STAGGEREDPC_DIRAC = 15
    QUDA_STAGGEREDKD_DIRAC = 16
    QUDA_ASQTAD_DIRAC = 17
    QUDA_ASQTADPC_DIRAC = 18
    QUDA_ASQTADKD_DIRAC = 19
    QUDA_TWISTED_MASS_DIRAC = 20
    QUDA_TWISTED_MASSPC_DIRAC = 21
    QUDA_TWISTED_CLOVER_DIRAC = 22
    QUDA_TWISTED_CLOVERPC_DIRAC = 23
    QUDA_COARSE_DIRAC = 24
    QUDA_COARSEPC_DIRAC = 25
    QUDA_GAUGE_LAPLACE_DIRAC = 26
    QUDA_GAUGE_LAPLACEPC_DIRAC = 27
    QUDA_GAUGE_COVDEV_DIRAC = 28
    QUDA_INVALID_DIRAC = QUDA_INVALID_ENUM


# Where the field is stored
class QudaFieldLocation(IntEnum):
    QUDA_CPU_FIELD_LOCATION = 1
    QUDA_CUDA_FIELD_LOCATION = 2
    QUDA_INVALID_FIELD_LOCATION = QUDA_INVALID_ENUM


# Which sites are included
class QudaSiteSubset(IntEnum):
    QUDA_PARITY_SITE_SUBSET = 1
    QUDA_FULL_SITE_SUBSET = 2
    QUDA_INVALID_SITE_SUBSET = QUDA_INVALID_ENUM


# Site ordering (always t-z-y-x, with rightmost varying fastest)
class QudaSiteOrder(IntEnum):
    QUDA_LEXICOGRAPHIC_SITE_ORDER = 0  # lexicographic ordering
    QUDA_EVEN_ODD_SITE_ORDER = 1  # QUDA and QDP use this
    QUDA_ODD_EVEN_SITE_ORDER = 2  # CPS uses this
    QUDA_INVALID_SITE_ORDER = QUDA_INVALID_ENUM


# Degree of freedom ordering
class QudaFieldOrder(IntEnum):
    QUDA_FLOAT_FIELD_ORDER = 1  # spin-color-complex-space
    QUDA_FLOAT2_FIELD_ORDER = 2  # (spin-color-complex)/2-space-(spin-color-complex)%2
    QUDA_FLOAT4_FIELD_ORDER = 4  # (spin-color-complex)/4-space-(spin-color-complex)%4
    QUDA_FLOAT8_FIELD_ORDER = 8  # (spin-color-complex)/8-space-(spin-color-complex)%8
    QUDA_SPACE_SPIN_COLOR_FIELD_ORDER = 9  # CPS/QDP++ ordering
    QUDA_SPACE_COLOR_SPIN_FIELD_ORDER = 10  # QLA ordering (spin inside color)
    QUDA_QDPJIT_FIELD_ORDER = 11  # QDP field ordering (complex-color-spin-spacetime)
    QUDA_QOP_DOMAIN_WALL_FIELD_ORDER = 12  # QOP domain-wall ordering
    QUDA_PADDED_SPACE_SPIN_COLOR_FIELD_ORDER = 13  # TIFR RHMC ordering
    QUDA_INVALID_FIELD_ORDER = QUDA_INVALID_ENUM


class QudaFieldCreate(IntEnum):
    QUDA_NULL_FIELD_CREATE = 0  # new field
    QUDA_ZERO_FIELD_CREATE = 1  # new field and zero it
    QUDA_COPY_FIELD_CREATE = 2  # copy to field
    QUDA_REFERENCE_FIELD_CREATE = 3  # reference to field
    QUDA_GHOST_FIELD_CREATE = 4  # dummy field used only for ghost storage
    QUDA_INVALID_FIELD_CREATE = QUDA_INVALID_ENUM


class QudaGammaBasis(IntEnum):
    QUDA_DEGRAND_ROSSI_GAMMA_BASIS = 0
    QUDA_UKQCD_GAMMA_BASIS = 1
    QUDA_CHIRAL_GAMMA_BASIS = 2
    QUDA_INVALID_GAMMA_BASIS = QUDA_INVALID_ENUM


class QudaSourceType(IntEnum):
    QUDA_POINT_SOURCE = 0
    QUDA_RANDOM_SOURCE = 1
    QUDA_CONSTANT_SOURCE = 2
    QUDA_SINUSOIDAL_SOURCE = 3
    QUDA_CORNER_SOURCE = 4
    QUDA_INVALID_SOURCE = QUDA_INVALID_ENUM


class QudaNoiseType(IntEnum):
    QUDA_NOISE_GAUSS = 0
    QUDA_NOISE_UNIFORM = 1
    QUDA_NOISE_INVALID = QUDA_INVALID_ENUM


class QudaDilutionType(IntEnum):
    QUDA_DILUTION_SPIN = 0
    QUDA_DILUTION_COLOR = 1
    QUDA_DILUTION_SPIN_COLOR = 2
    QUDA_DILUTION_SPIN_COLOR_EVEN_ODD = 3
    QUDA_DILUTION_INVALID = QUDA_INVALID_ENUM


# used to select projection method for deflated solvers
class QudaProjectionType(IntEnum):
    QUDA_MINRES_PROJECTION = 0
    QUDA_GALERKIN_PROJECTION = 1
    QUDA_INVALID_PROJECTION = QUDA_INVALID_ENUM


# used to select checkerboard preconditioning method
class QudaPCType(IntEnum):
    QUDA_4D_PC = 4
    QUDA_5D_PC = 5
    QUDA_PC_INVALID = QUDA_INVALID_ENUM


class QudaTwistFlavorType(IntEnum):
    QUDA_TWIST_SINGLET = 1
    QUDA_TWIST_NONDEG_DOUBLET = +2
    QUDA_TWIST_NO = 0
    QUDA_TWIST_INVALID = QUDA_INVALID_ENUM


class QudaTwistDslashType(IntEnum):
    QUDA_DEG_TWIST_INV_DSLASH = 0
    QUDA_DEG_DSLASH_TWIST_INV = 1
    QUDA_DEG_DSLASH_TWIST_XPAY = 2
    QUDA_NONDEG_DSLASH = 3
    QUDA_DSLASH_INVALID = QUDA_INVALID_ENUM


class QudaTwistCloverDslashType(IntEnum):
    QUDA_DEG_CLOVER_TWIST_INV_DSLASH = 0
    QUDA_DEG_DSLASH_CLOVER_TWIST_INV = 1
    QUDA_DEG_DSLASH_CLOVER_TWIST_XPAY = 2
    QUDA_TC_DSLASH_INVALID = QUDA_INVALID_ENUM


class QudaTwistGamma5Type(IntEnum):
    QUDA_TWIST_GAMMA5_DIRECT = 0
    QUDA_TWIST_GAMMA5_INVERSE = 1
    QUDA_TWIST_GAMMA5_INVALID = QUDA_INVALID_ENUM


class QudaUseInitGuess(IntEnum):
    QUDA_USE_INIT_GUESS_NO = 0
    QUDA_USE_INIT_GUESS_YES = 1
    QUDA_USE_INIT_GUESS_INVALID = QUDA_INVALID_ENUM


class QudaDeflatedGuess(IntEnum):
    QUDA_DEFLATED_GUESS_NO = 0
    QUDA_DEFLATED_GUESS_YES = 1
    QUDA_DEFLATED_GUESS_INVALID = QUDA_INVALID_ENUM


class QudaComputeNullVector(IntEnum):
    QUDA_COMPUTE_NULL_VECTOR_NO = 0
    QUDA_COMPUTE_NULL_VECTOR_YES = 1
    QUDA_COMPUTE_NULL_VECTOR_INVALID = QUDA_INVALID_ENUM


class QudaSetupType(IntEnum):
    QUDA_NULL_VECTOR_SETUP = 0
    QUDA_TEST_VECTOR_SETUP = 1
    QUDA_INVALID_SETUP_TYPE = QUDA_INVALID_ENUM


class QudaTransferType(IntEnum):
    QUDA_TRANSFER_AGGREGATE = 0
    QUDA_TRANSFER_COARSE_KD = 1
    QUDA_TRANSFER_OPTIMIZED_KD = 2
    QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG = 3
    QUDA_TRANSFER_INVALID = QUDA_INVALID_ENUM


class QudaBoolean(IntEnum):
    QUDA_BOOLEAN_FALSE = 0
    QUDA_BOOLEAN_TRUE = 1
    QUDA_BOOLEAN_INVALID = QUDA_INVALID_ENUM


# define these for backwards compatibility
QUDA_BOOLEAN_NO = QudaBoolean.QUDA_BOOLEAN_FALSE
QUDA_BOOLEAN_YES = QudaBoolean.QUDA_BOOLEAN_TRUE


class QudaBLASType(IntEnum):
    QUDA_BLAS_GEMM = 0
    QUDA_BLAS_LU_INV = 1
    QUDA_BLAS_INVALID = QUDA_INVALID_ENUM


class QudaBLASOperation(IntEnum):
    QUDA_BLAS_OP_N = 0  # No transpose
    QUDA_BLAS_OP_T = 1  # Transpose only
    QUDA_BLAS_OP_C = 2  # Conjugate transpose
    QUDA_BLAS_OP_INVALID = QUDA_INVALID_ENUM


class QudaBLASDataType(IntEnum):
    QUDA_BLAS_DATATYPE_S = 0  # Single
    QUDA_BLAS_DATATYPE_D = 1  # Double
    QUDA_BLAS_DATATYPE_C = 2  # Complex(single)
    QUDA_BLAS_DATATYPE_Z = 3  # Complex(double)
    QUDA_BLAS_DATATYPE_INVALID = QUDA_INVALID_ENUM


class QudaBLASDataOrder(IntEnum):
    QUDA_BLAS_DATAORDER_ROW = 0
    QUDA_BLAS_DATAORDER_COL = 1
    QUDA_BLAS_DATAORDER_INVALID = QUDA_INVALID_ENUM


class QudaDirection(IntEnum):
    QUDA_BACKWARDS = -1
    QUDA_IN_PLACE = 0
    QUDA_FORWARDS = +1
    QUDA_BOTH_DIRS = 2


class QudaLinkDirection(IntEnum):
    QUDA_LINK_BACKWARDS = 0
    QUDA_LINK_FORWARDS = 1
    QUDA_LINK_BIDIRECTIONAL = 2


class QudaFieldGeometry(IntEnum):
    QUDA_SCALAR_GEOMETRY = 1
    QUDA_VECTOR_GEOMETRY = 4
    QUDA_TENSOR_GEOMETRY = 6
    QUDA_COARSE_GEOMETRY = 8
    QUDA_KDINVERSE_GEOMETRY = 16  # Decomposition of Kahler-Dirac block
    QUDA_INVALID_GEOMETRY = QUDA_INVALID_ENUM


class QudaGhostExchange(IntEnum):
    QUDA_GHOST_EXCHANGE_NO = 0
    QUDA_GHOST_EXCHANGE_PAD = 1
    QUDA_GHOST_EXCHANGE_EXTENDED = 2
    QUDA_GHOST_EXCHANGE_INVALID = QUDA_INVALID_ENUM


class QudaStaggeredPhase(IntEnum):
    QUDA_STAGGERED_PHASE_NO = 0
    QUDA_STAGGERED_PHASE_MILC = 1
    QUDA_STAGGERED_PHASE_CHROMA = 2
    QUDA_STAGGERED_PHASE_TIFR = 3
    QUDA_STAGGERED_PHASE_INVALID = QUDA_INVALID_ENUM


class QudaContractType(IntEnum):
    QUDA_CONTRACT_TYPE_OPEN = 0  # Open spin elementals
    QUDA_CONTRACT_TYPE_DR = 1  # DegrandRossi
    QUDA_CONTRACT_TYPE_INVALID = QUDA_INVALID_ENUM


class QudaContractGamma(IntEnum):
    QUDA_CONTRACT_GAMMA_I = 0
    QUDA_CONTRACT_GAMMA_G1 = 1
    QUDA_CONTRACT_GAMMA_G2 = 2
    QUDA_CONTRACT_GAMMA_G3 = 3
    QUDA_CONTRACT_GAMMA_G4 = 4
    QUDA_CONTRACT_GAMMA_G5 = 5
    QUDA_CONTRACT_GAMMA_G1G5 = 6
    QUDA_CONTRACT_GAMMA_G2G5 = 7
    QUDA_CONTRACT_GAMMA_G3G5 = 8
    QUDA_CONTRACT_GAMMA_G4G5 = 9
    QUDA_CONTRACT_GAMMA_S12 = 10
    QUDA_CONTRACT_GAMMA_S13 = 11
    QUDA_CONTRACT_GAMMA_S14 = 12
    QUDA_CONTRACT_GAMMA_S21 = 13
    QUDA_CONTRACT_GAMMA_S23 = 14
    QUDA_CONTRACT_GAMMA_S34 = 15
    QUDA_CONTRACT_GAMMA_INVALID = QUDA_INVALID_ENUM


class QudaGaugeSmearType(IntEnum):
    QUDA_GAUGE_SMEAR_APE = 0
    QUDA_GAUGE_SMEAR_STOUT = 1
    QUDA_GAUGE_SMEAR_OVRIMP_STOUT = 2
    QUDA_GAUGE_SMEAR_WILSON_FLOW = 3
    QUDA_GAUGE_SMEAR_SYMANZIK_FLOW = 4
    QUDA_GAUGE_SMEAR_HYP = 5
    QUDA_GAUGE_SMEAR_INVALID = QUDA_INVALID_ENUM


class QudaFermionSmearType(IntEnum):
    QUDA_FERMION_SMEAR_TYPE_GAUSSIAN = 0
    QUDA_FERMION_SMEAR_TYPE_WUPPERTAL = 1
    QUDA_FERMION_SMEAR_TYPE_INVALID = QUDA_INVALID_ENUM


# Allows to choose an appropriate external library
class QudaExtLibType(IntEnum):
    QUDA_CUSOLVE_EXTLIB = 0
    QUDA_EIGEN_EXTLIB = 1
    QUDA_EXTLIB_INVALID = QUDA_INVALID_ENUM
