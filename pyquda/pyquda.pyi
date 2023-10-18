from typing import List

size_t = int
double = float
double_complex = complex

from .enum_quda import (  # noqa: F401
    QUDA_INVALID_ENUM,
    QUDA_VERSION_MAJOR,
    QUDA_VERSION_MINOR,
    QUDA_VERSION_SUBMINOR,
    QUDA_VERSION,
    QUDA_MAX_DIM,
    QUDA_MAX_GEOMETRY,
    QUDA_MAX_MULTI_SHIFT,
    QUDA_MAX_BLOCK_SRC,
    QUDA_MAX_ARRAY_SIZE,
    QUDA_MAX_DWF_LS,
    QUDA_MAX_MG_LEVEL,
    qudaError_t,
    QudaMemoryType,
    QudaLinkType,
    QudaGaugeFieldOrder,
    QudaTboundary,
    QudaPrecision,
    QudaReconstructType,
    QudaGaugeFixed,
    QudaDslashType,
    QudaInverterType,
    QudaEigType,
    QudaEigSpectrumType,
    QudaSolutionType,
    QudaSolveType,
    QudaMultigridCycleType,
    QudaSchwarzType,
    QudaAcceleratorType,
    QudaResidualType,
    QudaCABasis,
    QudaMatPCType,
    QudaDagType,
    QudaMassNormalization,
    QudaSolverNormalization,
    QudaPreserveSource,
    QudaDiracFieldOrder,
    QudaCloverFieldOrder,
    QudaVerbosity,
    QudaTune,
    QudaPreserveDirac,
    QudaParity,
    QudaDiracType,
    QudaFieldLocation,
    QudaSiteSubset,
    QudaSiteOrder,
    QudaFieldOrder,
    QudaFieldCreate,
    QudaGammaBasis,
    QudaSourceType,
    QudaNoiseType,
    QudaDilutionType,
    QudaProjectionType,
    QudaPCType,
    QudaTwistFlavorType,
    QudaTwistDslashType,
    QudaTwistCloverDslashType,
    QudaTwistGamma5Type,
    QudaUseInitGuess,
    QudaDeflatedGuess,
    QudaComputeNullVector,
    QudaSetupType,
    QudaTransferType,
    QudaBoolean,
    QUDA_BOOLEAN_NO,
    QUDA_BOOLEAN_YES,
    QudaBLASType,
    QudaBLASOperation,
    QudaBLASDataType,
    QudaBLASDataOrder,
    QudaDirection,
    QudaLinkDirection,
    QudaFieldGeometry,
    QudaGhostExchange,
    QudaStaggeredPhase,
    QudaContractType,
    QudaContractGamma,
    QudaGaugeSmearType,
    QudaExtLibType,
)

from .pointer import Pointer, Pointers

class QudaGaugeParam:
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

    struct_size: size_t
    location: QudaFieldLocation
    X: List[int, 4]
    anisotropy: double
    tadpole_coeff: double
    scale: double
    type: QudaLinkType
    gauge_order: QudaGaugeFieldOrder
    t_boundary: QudaTboundary
    cpu_prec: QudaPrecision
    cuda_prec: QudaPrecision
    reconstruct: QudaReconstructType
    cuda_prec_sloppy: QudaPrecision
    reconstruct_sloppy: QudaReconstructType
    cuda_prec_refinement_sloppy: QudaPrecision
    reconstruct_refinement_sloppy: QudaReconstructType
    cuda_prec_precondition: QudaPrecision
    reconstruct_precondition: QudaReconstructType
    cuda_prec_eigensolver: QudaPrecision
    reconstruct_eigensolver: QudaReconstructType
    gauge_fix: QudaGaugeFixed
    ga_pad: int
    site_ga_pad: int
    staple_pad: int
    llfat_ga_pad: int
    mom_ga_pad: int
    staggered_phase_type: QudaStaggeredPhase
    staggered_phase_applied: int
    i_mu: double
    overlap: int
    overwrite_gauge: int
    overwrite_mom: int
    use_resident_gauge: int
    use_resident_mom: int
    make_resident_gauge: int
    make_resident_mom: int
    return_result_gauge: int
    return_result_mom: int
    gauge_offset: size_t
    mom_offset: size_t
    site_size: size_t

class QudaInvertParam:
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

    struct_size: size_t
    input_location: QudaFieldLocation
    output_location: QudaFieldLocation
    dslash_type: QudaDslashType
    inv_type: QudaInverterType
    mass: double
    kappa: double
    m5: double
    Ls: int
    b_5: List[double_complex, QUDA_MAX_DWF_LS]
    c_5: List[double_complex, QUDA_MAX_DWF_LS]
    eofa_shift: double
    eofa_pm: int
    mq1: double
    mq2: double
    mq3: double
    mu: double
    tm_rho: double
    epsilon: double
    twist_flavor: QudaTwistFlavorType
    laplace3D: int
    tol: double
    tol_restart: double
    tol_hq: double
    compute_true_res: int
    true_res: double
    true_res_hq: double
    maxiter: int
    reliable_delta: double
    reliable_delta_refinement: double
    use_alternative_reliable: int
    use_sloppy_partial_accumulator: int
    solution_accumulator_pipeline: int
    max_res_increase: int
    max_res_increase_total: int
    max_hq_res_increase: int
    max_hq_res_restart_total: int
    heavy_quark_check: int
    pipeline: int
    num_offset: int
    num_src: int
    num_src_per_sub_partition: int
    split_grid: List[int, QUDA_MAX_DIM]
    overlap: int
    offset: List[double, QUDA_MAX_MULTI_SHIFT]
    tol_offset: List[double, QUDA_MAX_MULTI_SHIFT]
    tol_hq_offset: List[double, QUDA_MAX_MULTI_SHIFT]
    true_res_offset: List[double, QUDA_MAX_MULTI_SHIFT]
    iter_res_offset: List[double, QUDA_MAX_MULTI_SHIFT]
    true_res_hq_offset: List[double, QUDA_MAX_MULTI_SHIFT]
    residue: List[double, QUDA_MAX_MULTI_SHIFT]
    compute_action: int
    action: List[double, 2]
    solution_type: QudaSolutionType
    solve_type: QudaSolveType
    matpc_type: QudaMatPCType
    dagger: QudaDagType
    mass_normalization: QudaMassNormalization
    solver_normalization: QudaSolverNormalization
    preserve_source: QudaPreserveSource
    cpu_prec: QudaPrecision
    cuda_prec: QudaPrecision
    cuda_prec_sloppy: QudaPrecision
    cuda_prec_refinement_sloppy: QudaPrecision
    cuda_prec_precondition: QudaPrecision
    cuda_prec_eigensolver: QudaPrecision
    dirac_order: QudaDiracFieldOrder
    gamma_basis: QudaGammaBasis
    clover_location: QudaFieldLocation
    clover_cpu_prec: QudaPrecision
    clover_cuda_prec: QudaPrecision
    clover_cuda_prec_sloppy: QudaPrecision
    clover_cuda_prec_refinement_sloppy: QudaPrecision
    clover_cuda_prec_precondition: QudaPrecision
    clover_cuda_prec_eigensolver: QudaPrecision
    clover_order: QudaCloverFieldOrder
    use_init_guess: QudaUseInitGuess
    clover_csw: double
    clover_coeff: double
    clover_rho: double
    compute_clover_trlog: int
    trlogA: List[double, 2]
    compute_clover: int
    compute_clover_inverse: int
    return_clover: int
    return_clover_inverse: int
    verbosity: QudaVerbosity
    iter: int
    gflops: double
    secs: double
    tune: QudaTune
    Nsteps: int
    gcrNkrylov: int
    inv_type_precondition: QudaInverterType
    preconditioner: Pointer
    deflation_op: Pointer
    eig_param: Pointer
    deflate: QudaBoolean
    dslash_type_precondition: QudaDslashType
    verbosity_precondition: QudaVerbosity
    tol_precondition: double
    maxiter_precondition: int
    omega: double
    ca_basis: QudaCABasis
    ca_lambda_min: double
    ca_lambda_max: double
    ca_basis_precondition: QudaCABasis
    ca_lambda_min_precondition: double
    ca_lambda_max_precondition: double
    precondition_cycle: int
    schwarz_type: QudaSchwarzType
    accelerator_type_precondition: QudaAcceleratorType
    madwf_diagonal_suppressor: double
    madwf_ls: int
    madwf_null_miniter: int
    madwf_null_tol: double
    madwf_train_maxiter: int
    madwf_param_load: QudaBoolean
    madwf_param_save: QudaBoolean
    madwf_param_infile: bytes[256]
    madwf_param_outfile: bytes[256]
    residual_type: QudaResidualType
    cuda_prec_ritz: QudaPrecision
    n_ev: int
    max_search_dim: int
    rhs_idx: int
    deflation_grid: int
    eigenval_tol: double
    eigcg_max_restarts: int
    max_restart_num: int
    inc_tol: double
    make_resident_solution: int
    use_resident_solution: int
    chrono_make_resident: int
    chrono_replace_last: int
    chrono_use_resident: int
    chrono_max_dim: int
    chrono_index: int
    chrono_precision: QudaPrecision
    extlib_type: QudaExtLibType
    native_blas_lapack: QudaBoolean
    use_mobius_fused_kernel: QudaBoolean

class QudaEigParam:
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

    struct_size: size_t
    invert_param: QudaInvertParam
    eig_type: QudaEigType
    use_poly_acc: QudaBoolean
    poly_deg: int
    a_min: double
    a_max: double
    preserve_deflation: QudaBoolean
    preserve_deflation_space: Pointer
    preserve_evals: QudaBoolean
    use_dagger: QudaBoolean
    use_norm_op: QudaBoolean
    use_eigen_qr: QudaBoolean
    use_pc: QudaBoolean
    compute_svd: QudaBoolean
    compute_gamma5: QudaBoolean
    require_convergence: QudaBoolean
    spectrum: QudaEigSpectrumType
    n_ev: int
    n_kr: int
    nLockedMax: int
    n_conv: int
    n_ev_deflate: int
    tol: double
    qr_tol: double
    check_interval: int
    max_restarts: int
    batched_rotate: int
    block_size: int
    max_ortho_attempts: int
    ortho_block_size: int
    arpack_check: QudaBoolean
    arpack_logfile: bytes[512]
    QUDA_logfile: bytes[512]
    nk: int
    np: int
    import_vectors: QudaBoolean
    cuda_prec_ritz: QudaPrecision
    mem_type_ritz: QudaMemoryType
    location: QudaFieldLocation
    run_verify: QudaBoolean
    vec_infile: bytes[256]
    vec_outfile: bytes[256]
    save_prec: QudaPrecision
    io_parity_inflate: QudaBoolean
    partfile: QudaBoolean
    gflops: double
    secs: double
    extlib_type: QudaExtLibType

class QudaMultigridParam:
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

    struct_size: size_t
    invert_param: QudaInvertParam
    eig_param: List[QudaEigParam, QUDA_MAX_MG_LEVEL]
    n_level: int
    geo_block_size: List[List[int, QUDA_MAX_DIM], QUDA_MAX_MG_LEVEL]
    spin_block_size: List[int, QUDA_MAX_MG_LEVEL]
    n_vec: List[int, QUDA_MAX_MG_LEVEL]
    precision_null: List[QudaPrecision, QUDA_MAX_MG_LEVEL]
    n_block_ortho: List[int, QUDA_MAX_MG_LEVEL]
    block_ortho_two_pass: List[QudaBoolean, QUDA_MAX_MG_LEVEL]
    verbosity: List[QudaVerbosity, QUDA_MAX_MG_LEVEL]
    setup_use_mma: List[QudaBoolean, QUDA_MAX_MG_LEVEL]
    dslash_use_mma: List[QudaBoolean, QUDA_MAX_MG_LEVEL]
    setup_inv_type: List[QudaInverterType, QUDA_MAX_MG_LEVEL]
    num_setup_iter: List[int, QUDA_MAX_MG_LEVEL]
    setup_tol: List[double, QUDA_MAX_MG_LEVEL]
    setup_maxiter: List[int, QUDA_MAX_MG_LEVEL]
    setup_maxiter_refresh: List[int, QUDA_MAX_MG_LEVEL]
    setup_ca_basis: List[QudaCABasis, QUDA_MAX_MG_LEVEL]
    setup_ca_basis_size: List[int, QUDA_MAX_MG_LEVEL]
    setup_ca_lambda_min: List[double, QUDA_MAX_MG_LEVEL]
    setup_ca_lambda_max: List[double, QUDA_MAX_MG_LEVEL]
    setup_type: QudaSetupType
    pre_orthonormalize: QudaBoolean
    post_orthonormalize: QudaBoolean
    coarse_solver: List[QudaInverterType, QUDA_MAX_MG_LEVEL]
    coarse_solver_tol: List[double, QUDA_MAX_MG_LEVEL]
    coarse_solver_maxiter: List[int, QUDA_MAX_MG_LEVEL]
    coarse_solver_ca_basis: List[QudaCABasis, QUDA_MAX_MG_LEVEL]
    coarse_solver_ca_basis_size: List[int, QUDA_MAX_MG_LEVEL]
    coarse_solver_ca_lambda_min: List[double, QUDA_MAX_MG_LEVEL]
    coarse_solver_ca_lambda_max: List[double, QUDA_MAX_MG_LEVEL]
    smoother: List[QudaInverterType, QUDA_MAX_MG_LEVEL]
    smoother_tol: List[double, QUDA_MAX_MG_LEVEL]
    nu_pre: List[int, QUDA_MAX_MG_LEVEL]
    nu_post: List[int, QUDA_MAX_MG_LEVEL]
    smoother_solver_ca_basis: List[QudaCABasis, QUDA_MAX_MG_LEVEL]
    smoother_solver_ca_lambda_min: List[double, QUDA_MAX_MG_LEVEL]
    smoother_solver_ca_lambda_max: List[double, QUDA_MAX_MG_LEVEL]
    omega: List[double, QUDA_MAX_MG_LEVEL]
    smoother_halo_precision: List[QudaPrecision, QUDA_MAX_MG_LEVEL]
    smoother_schwarz_type: List[QudaSchwarzType, QUDA_MAX_MG_LEVEL]
    smoother_schwarz_cycle: List[int, QUDA_MAX_MG_LEVEL]
    coarse_grid_solution_type: List[QudaSolutionType, QUDA_MAX_MG_LEVEL]
    smoother_solve_type: List[QudaSolveType, QUDA_MAX_MG_LEVEL]
    cycle_type: List[QudaMultigridCycleType, QUDA_MAX_MG_LEVEL]
    global_reduction: List[QudaBoolean, QUDA_MAX_MG_LEVEL]
    location: List[QudaFieldLocation, QUDA_MAX_MG_LEVEL]
    setup_location: List[QudaFieldLocation, QUDA_MAX_MG_LEVEL]
    use_eig_solver: List[QudaBoolean, QUDA_MAX_MG_LEVEL]
    setup_minimize_memory: QudaBoolean
    compute_null_vector: QudaComputeNullVector
    generate_all_levels: QudaBoolean
    run_verify: QudaBoolean
    run_low_mode_check: QudaBoolean
    run_oblique_proj_check: QudaBoolean
    vec_load: List[QudaBoolean, QUDA_MAX_MG_LEVEL]
    vec_infile: List[bytes[256], QUDA_MAX_MG_LEVEL]
    vec_store: List[QudaBoolean, QUDA_MAX_MG_LEVEL]
    vec_outfile: List[bytes[256], QUDA_MAX_MG_LEVEL]
    mg_vec_partfile: List[QudaBoolean, QUDA_MAX_MG_LEVEL]
    coarse_guess: QudaBoolean
    preserve_deflation: QudaBoolean
    gflops: double
    secs: double
    mu_factor: List[double, QUDA_MAX_MG_LEVEL]
    transfer_type: List[QudaTransferType, QUDA_MAX_MG_LEVEL]
    allow_truncation: QudaBoolean
    staggered_kd_dagger_approximation: QudaBoolean
    thin_update_only: QudaBoolean

class QudaGaugeObservableParam:
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

    struct_size: size_t
    su_project: QudaBoolean
    compute_plaquette: QudaBoolean
    plaquette: List[double, 3]
    compute_gauge_loop_trace: QudaBoolean
    traces: Pointer
    input_path_buff: Pointers
    path_length: Pointer
    loop_coeff: Pointer
    num_paths: int
    max_length: int
    factor: double
    compute_qcharge: QudaBoolean
    qcharge: double
    energy: List[double, 3]
    compute_qcharge_density: QudaBoolean
    qcharge_density: Pointer

class QudaGaugeSmearParam:
    def __init__(self) -> None: ...
    # def __repr__(self) -> str: ...

    struct_size: size_t
    n_steps: int
    epsilon: double
    alpha: double
    rho: double
    alpha1: double
    alpha2: double
    alpha3: double
    meas_interval: int
    smear_type: QudaGaugeSmearType

class QudaBLASParam:
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

    struct_size: size_t
    trans_a: QudaBLASOperation
    trans_b: QudaBLASOperation
    m: int
    n: int
    k: int
    lda: int
    ldb: int
    ldc: int
    a_offset: int
    b_offset: int
    c_offset: int
    a_stride: int
    b_stride: int
    c_stride: int
    alpha: double_complex
    beta: double_complex
    batch_count: int
    data_type: QudaBLASDataType
    data_order: QudaBLASDataOrder

def setVerbosityQuda(verbosity: QudaVerbosity, prefix: bytes) -> None:
    """
    Set parameters related to status reporting.

    In typical usage, this function will be called once (or not at
    all) just before the call to initQuda(), but it's valid to call
    it any number of times at any point during execution.  Prior to
    the first time it's called, the parameters take default values
    as indicated below.

    @param verbosity:
        Default verbosity, ranging from QUDA_SILENT to
        QUDA_DEBUG_VERBOSE.  Within a solver, this
        parameter is overridden by the "verbosity"
        member of QudaInvertParam.  The default value
        is QUDA_SUMMARIZE.

    @param prefix:
        String to prepend to all messages from QUDA.  This
        defaults to the empty string (""), but you may
        wish to specify something like "QUDA: " to
        distinguish QUDA's output from that of your
        application.
    """
    ...

def initCommsGridQuda(nDim: int, dims: List[int, 4]):
    """
    Declare the grid mapping ("logical topology" in QMP parlance)
    used for communications in a multi-GPU grid.  This function
    should be called prior to initQuda().  The only case in which
    it's optional is when QMP is used for communication and the
    logical topology has already been declared by the application.

    @param nDim:
        Number of grid dimensions.  "4" is the only supported
        value currently.

    @param dims:
        Array of grid dimensions.  dims[0]*dims[1]*dims[2]*dims[3]
        must equal the total number of MPI ranks or QMP nodes.
    """
    ...

def initQudaDevice(device: int) -> None:
    """
    Initialize the library.  This is a low-level interface that is
    called by initQuda.  Calling initQudaDevice requires that the
    user also call initQudaMemory before using QUDA.

    @param device:
        CUDA device number to use.  In a multi-GPU build,
        this parameter may either be set explicitly on a
        per-process basis or set to -1 to enable a default
        allocation of devices to processes.
    """
    ...

def initQudaMemory() -> None:
    """
    Initialize the library persistant memory allocations (both host
    and device).  This is a low-level interface that is called by
    initQuda.  Calling initQudaMemory requires that the user has
    previously called initQudaDevice.
    """
    ...

def initQuda(device: int) -> None:
    """
    Initialize the library.  This function is actually a wrapper
    around calls to initQudaDevice() and initQudaMemory().

    @param device:
        CUDA device number to use.  In a multi-GPU build,
        this parameter may either be set explicitly on a
        per-process basis or set to -1 to enable a default
        allocation of devices to processes.
    """
    ...

def endQuda() -> None:
    """
    Finalize the library.
    """
    ...

def loadGaugeQuda(h_gauge: Pointers, param: QudaGaugeParam) -> None:
    """
    Load the gauge field from the host.
    @param h_gauge:
        Base pointer to host gauge field (regardless of dimensionality)
    @param param:
        Contains all metadata regarding host and device storage
    """
    ...

def freeGaugeQuda() -> None:
    """
    Free QUDA's internal copy of the gauge field.
    """
    ...

def freeUniqueGaugeQuda(link_type: QudaLinkType) -> None:
    """
    Free a unique type (Wilson, HISQ fat, HISQ long, smeared) of internal gauge field.
    @param link_type[in]:
        Type of link type to free up
    """
    ...

def freeGaugeSmearedQuda() -> None:
    """
    Free QUDA's internal smeared gauge field.
    """
    ...

def saveGaugeQuda(h_gauge: Pointers, param: QudaGaugeParam) -> None:
    """
    Save the gauge field to the host.
    @param h_gauge:
        Base pointer to host gauge field (regardless of dimensionality)
    @param param:
        Contains all metadata regarding host and device storage
    """
    ...

def loadCloverQuda(h_clover: Pointer, h_clovinv: Pointer, inv_param: QudaInvertParam) -> None:
    """
    Load the clover term and/or the clover inverse from the host.
    Either h_clover or h_clovinv may be set to NULL.
    @param h_clover:
        Base pointer to host clover field
    @param h_cloverinv:
        Base pointer to host clover inverse field
    @param inv_param:
        Contains all metadata regarding host and device storage
    """
    ...

def freeCloverQuda() -> None:
    """
    Free QUDA's internal copy of the clover term and/or clover inverse.
    """
    ...

def invertQuda(h_x: Pointer, h_b: Pointer, param: QudaInvertParam) -> None:
    """
    Perform the solve, according to the parameters set in param.  It
    is assumed that the gauge field has already been loaded via
    loadGaugeQuda().
    @param h_x:
        Solution spinor field
    @param h_b:
        Source spinor field
    @param param:
        Contains all metadata regarding host and device
        storage and solver parameters
    """
    ...

def invertMultiShiftQuda(_hp_x: Pointers, _hp_b: Pointer, param: QudaInvertParam) -> None:
    """
    Perform the solve like @invertQuda but for multiple rhs by spliting the comm grid into sub-partitions:
    each sub-partition invert one or more rhs'.
    The QudaInvertParam object specifies how the solve should be performed on each sub-partition.
    Unlike @invertQuda, the interface also takes the host side gauge as input. The gauge pointer and
    gauge_param are used if for inv_param split_grid[0] * split_grid[1] * split_grid[2] * split_grid[3]
    is larger than 1, in which case gauge field is not required to be loaded beforehand; otherwise
    this interface would just work as @invertQuda, which requires gauge field to be loaded beforehand,
    and the gauge field pointer and gauge_param are not used.
    @param _hp_x:
        Array of solution spinor fields
    @param _hp_b:
        Array of source spinor fields
    @param param:
        Contains all metadata regarding host and device storage and solver parameters
    @param h_gauge:
        Base pointer to host gauge field (regardless of dimensionality)
    @param gauge_param:
        Contains all metadata regarding host and device storage for gauge field
    """
    ...

def newMultigridQuda(param: QudaMultigridParam) -> Pointer:
    """
    Setup the multigrid solver, according to the parameters set in param.  It
    is assumed that the gauge field has already been loaded via
    loadGaugeQuda().
    @param param:
        Contains all metadata regarding host and device
        storage and solver parameters
    """
    ...

def destroyMultigridQuda(mg_instance: Pointer):
    """
    Free resources allocated by the multigrid solver
    @param mg_instance:
        Pointer to instance of multigrid_solver
    @param param:
        Contains all metadata regarding host and device
        storage and solver parameters
    """
    ...

def updateMultigridQuda(mg_instance: Pointer, param: QudaMultigridParam):
    """
    Updates the multigrid preconditioner for the new gauge / clover field
    @param mg_instance:
        Pointer to instance of multigrid_solver
    @param param:
        Contains all metadata regarding host and device
        storage and solver parameters, of note contains a flag specifying whether
        to do a full update or a thin update.
    """
    ...

def dumpMultigridQuda(mg_instance: Pointer, param: QudaMultigridParam):
    """
    Dump the null-space vectors to disk
    @param[in] mg_instance:
        Pointer to the instance of multigrid_solver
    @param[in] param:
        Contains all metadata regarding host and device
        storage and solver parameters (QudaMultigridParam::vec_outfile
        sets the output filename prefix).
    """
    ...

def dslashQuda(h_out: Pointer, h_in: Pointer, inv_param: QudaInvertParam, parity: QudaParity) -> None:
    """
    Apply the Dslash operator (D_{eo} or D_{oe}).
    @param h_out:
        Result spinor field
    @param h_in:
        Input spinor field
    @param param:
        Contains all metadata regarding host and device
        storage
    @param parity:
        The destination parity of the field
    """
    ...

def cloverQuda(h_out: Pointer, h_in: Pointer, inv_param: QudaInvertParam, parity: QudaParity, inverse: int) -> None:
    """
    Apply the clover operator or its inverse.
    @param h_out:
        Result spinor field
    @param h_in:
        Input spinor field
    @param param:
        Contains all metadata regarding host and device
        storage
    @param parity:
        The source and destination parity of the field
    @param inverse:
        Whether to apply the inverse of the clover term
    """
    ...

def MatQuda(h_out: Pointer, h_in: Pointer, inv_param: QudaInvertParam) -> None:
    """
    Apply the full Dslash matrix, possibly even/odd preconditioned.
    @param h_out:
        Result spinor field
    @param h_in:
        Input spinor field
    @param param:
        Contains all metadata regarding host and device
        storage
    """
    ...

def MatDagMatQuda(h_out: Pointer, h_in: Pointer, inv_param: QudaInvertParam) -> None:
    r"""
    Apply M^{\dag}M, possibly even/odd preconditioned.
    @param h_out:
        Result spinor field
    @param h_in:
        Input spinor field
    @param param:
        Contains all metadata regarding host and device
        storage
    """
    ...

def computeKSLinkQuda(
    fatlink: Pointers, longlink: Pointers, ulink: Pointers, inlink: Pointers, path_coeff: Pointer, param: QudaGaugeParam
) -> None:
    """ """
    ...

def momResidentQuda(mom: Pointer, param: QudaGaugeParam) -> None:
    """
    Either downloads and sets the resident momentum field, or uploads
    and returns the resident momentum field

    @param[in,out] mom:
        The external momentum field
    @param[in] param:
        The parameters of the external field
    """
    ...

def computeGaugeForceQuda(
    mom: Pointers,
    sitelink: Pointers,
    input_path_buf: Pointer,
    path_length: Pointer,
    loop_coeff: Pointer,
    num_paths: int,
    max_length: int,
    dt: double,
    qudaGaugeParam: QudaGaugeParam,
) -> None:
    """
    Compute the gauge force and update the momentum field

    @param[in,out] mom:
        The momentum field to be updated
    @param[in] sitelink:
        The gauge field from which we compute the force
    @param[in] input_path_buf:
        [dim][num_paths][path_length]
    @param[in] path_length:
        One less that the number of links in a loop (e.g., 3 for a staple)
    @param[in] loop_coeff:
        Coefficients of the different loops in the Symanzik action
    @param[in] num_paths:
        How many contributions from path_length different "staples"
    @param[in] max_length:
        The maximum number of non-zero of links in any path in the action
    @param[in] dt:
        The integration step size (for MILC this is dt*beta/3)
    @param[in] param:
        The parameters of the external fields and the computation settings
    """
    ...

def computeGaugePathQuda(
    out: Pointers,
    sitelink: Pointers,
    input_path_buf: Pointer,
    path_length: Pointer,
    loop_coeff: Pointer,
    num_paths: int,
    max_length: int,
    dt: double,
    qudaGaugeParam: QudaGaugeParam,
) -> None:
    """
    Compute the product of gauge links along a path and add to/overwrite the output field

    @param[in,out] out:
        The output field to be updated
    @param[in] sitelink:
        The gauge field from which we compute the products of gauge links
    @param[in] input_path_buf:
        [dim][num_paths][path_length]
    @param[in] path_length:
        One less that the number of links in a loop (e.g., 3 for a staple)
    @param[in] loop_coeff:
        Coefficients of the different loops in the Symanzik action
    @param[in] num_paths:
        How many contributions from path_length different "staples"
    @param[in] max_length
        The maximum number of non-zero of links in any path in the action
    @param[in] dt:
        The integration step size (for MILC this is dt*beta/3)
    @param[in] param:
        The parameters of the external fields and the computation settings
    """
    ...

def computeGaugeLoopTraceQuda(
    traces: Pointer,
    input_path_buf: Pointers,
    path_length: Pointer,
    loop_coeff: Pointer,
    num_paths: int,
    max_length: int,
    factor: double,
) -> None:
    """
    Compute the traces of products of gauge links along paths using the resident field

    @param[in,out] traces:
        The computed traces
    @param[in] sitelink:
        The gauge field from which we compute the products of gauge links
    @param[in] path_length:
        The number of links in each loop
    @param[in] loop_coeff:
        Multiplicative coefficients for each loop
    @param[in] num_paths:
        Total number of loops
    @param[in] max_length:
        The maximum number of non-zero of links in any path in the action
    @param[in] factor:
        An overall normalization factor
    """
    ...

def updateGaugeFieldQuda(
    gauge: Pointers, momentum: Pointers, dt: double, conj_mom: int, exact: int, param: QudaGaugeParam
) -> None:
    """
    Evolve the gauge field by step size dt, using the momentum field
    I.e., Evalulate U(t+dt) = e(dt pi) U(t)

    @param gauge:
        The gauge field to be updated
    @param momentum:
        The momentum field
    @param dt:
        The integration step size step
    @param conj_mom:
        Whether to conjugate the momentum matrix
    @param exact:
        Whether to use an exact exponential or Taylor expand
    @param param:
        The parameters of the external fields and the computation settings
    """
    ...

def staggeredPhaseQuda(gauge_h: Pointers, param: QudaGaugeParam) -> None:
    """
    Apply the staggered phase factors to the gauge field.  If the
    imaginary chemical potential is non-zero then the phase factor
    exp(imu/T) will be applied to the links in the temporal
    direction.

    @param gauge_h:
        The gauge field
    @param param:
        The parameters of the gauge field
    """
    ...

def projectSU3Quda(gauge_h: Pointers, tol: double, param: QudaGaugeParam):
    """
    Project the input field on the SU(3) group.  If the target
    tolerance is not met, this routine will give a runtime error.

    @param gauge_h:
        The gauge field to be updated
    @param tol:
        The tolerance to which we iterate
    @param param:
        The parameters of the gauge field
    """
    ...

def momActionQuda(momentum: Pointer, param: QudaGaugeParam) -> double:
    """
    Evaluate the momentum contribution to the Hybrid Monte Carlo
    action.

    @param momentum:
        The momentum field
    @param param:
        The parameters of the external fields and the computation settings
    @return:
        momentum action
    """
    ...

def createCloverQuda(param: QudaInvertParam) -> None:
    """
    Compute the clover field and its inverse from the resident gauge field.

    @param param:
        The parameters of the clover field to create
    """
    ...

def computeCloverForceQuda(
    mom: Pointers,
    dt: double,
    x: Pointers,
    p: Pointers,
    coeff: Pointer,
    kappa2: double,
    ck: double,
    nvector: int,
    multiplicity: double,
    gauge: Pointers,
    gauge_param: QudaGaugeParam,
    inv_param: QudaInvertParam,
):
    """
    Compute the clover force contributions in each dimension mu given
    the array of solution fields, and compute the resulting momentum
    field.

    @param mom:
        Force matrix
    @param dt:
        Integrating step size
    @param x:
        Array of solution vectors
    @param[deprecated] p:
        Array of intermediate vectors
    @param coeff:
        Array of residues for each contribution (multiplied by stepsize)
    @param kappa2:
        -kappa*kappa parameter
    @param ck:
        -clover_coefficient * kappa / 8
    @param nvec:
        Number of vectors
    @param multiplicity:
        Number fermions this bilinear reresents
    @param[deprecated] gauge:
        Gauge Field
    @param gauge_param:
        Gauge field meta data
    @param inv_param:
        Dirac and solver meta data
    """
    ...

def gaussGaugeQuda(seed: int, sigma: double) -> None:
    """
    Generate Gaussian distributed fields and store in the
    resident gauge field.  We create a Gaussian-distributed su(n)
    field and exponentiate it, e.g., U = exp(sigma * H), where H is
    the distributed su(n) field and sigma is the width of the
    distribution (sigma = 0 results in a free field, and sigma = 1 has
    maximum disorder).

    @param seed:
        The seed used for the RNG
    @param sigma:
        Width of Gaussian distrubution
    """
    ...

def gaussMomQuda(seed: int, sigma: double) -> None:
    """
    Generate Gaussian distributed fields and store in the
    resident momentum field. We create a Gaussian-distributed su(n)
    field, e.g., sigma * H, where H is the distributed su(n) field
    and sigma is the width of the distribution (sigma = 0 results
    in a free field, and sigma = 1 has maximum disorder).

    @param seed:
        The seed used for the RNG
    @param sigma:
        Width of Gaussian distrubution
    """
    ...

def plaqQuda(plaq: List[double, 3]) -> None:
    """
    Computes the total, spatial and temporal plaquette averages of the loaded gauge configuration.
    @param[out] plaq:
        Array for storing the averages (total, spatial, temporal)
    """
    ...

def performGaugeSmearQuda(smear_param: QudaGaugeSmearParam, obs_param: QudaGaugeObservableParam) -> None:
    """
    Performs APE, Stout, or Over Imroved STOUT smearing on gaugePrecise and stores it in gaugeSmeared
    @param[in] smear_param:
        Parameter struct that defines the computation parameters
    @param[in,out] obs_param:
        Parameter struct that defines which
        observables we are making and the resulting observables.
    """
    ...

def performWFlowQuda(smear_param: QudaGaugeSmearParam, obs_param: QudaGaugeObservableParam) -> None:
    """
    Performs Wilson Flow on gaugePrecise and stores it in gaugeSmeared
    @param[in] smear_param:
        Parameter struct that defines the computation parameters
    @param[in,out] obs_param:
        Parameter struct that defines which
        observables we are making and the resulting observables.
    """
    ...

def gaugeObservablesQuda(param: QudaGaugeObservableParam):
    """
    Calculates a variety of gauge-field observables.  If a
    smeared gauge field is presently loaded (in gaugeSmeared) the
    observables are computed on this, else the resident gauge field
    will be used.
    @param[in,out] param:
        Parameter struct that defines which
        observables we are making and the resulting observables.
    """
    ...

def computeGaugeFixingOVRQuda(
    gauge: Pointers,
    gauge_dir: int,
    Nsteps: int,
    verbose_interval: int,
    relax_boost: double,
    tolerance: double,
    reunit_interval: int,
    stopWtheta: int,
    param: QudaGaugeParam,
    timeinfo: List[double, 3],
) -> int:
    """
    Gauge fixing with overrelaxation with support for single and multi GPU.
    @param[in,out] gauge:
        gauge field to be fixed
    @param[in] gauge_dir:
        3 for Coulomb gauge fixing, other for Landau gauge fixing
    @param[in] Nsteps:
        maximum number of steps to perform gauge fixing
    @param[in] verbose_interval:
        print gauge fixing info when iteration count is a multiple of this
    @param[in] relax_boost:
        gauge fixing parameter of the overrelaxation method, most common value is 1.5 or 1.7.
    @param[in] tolerance:
        torelance value to stop the method, if this value is zero then the method stops when
        iteration reachs the maximum number of steps defined by Nsteps
    @param[in] reunit_interval:
        reunitarize gauge field when iteration count is a multiple of this
    @param[in] stopWtheta:
        0 for MILC criterion and 1 to use the theta value
    @param[in] param:
        The parameters of the external fields and the computation settings
    @param[out] timeinfo:
    """
    ...

def computeGaugeFixingFFTQuda(
    gauge: Pointers,
    gauge_dir: int,
    Nsteps: int,
    verbose_interval: int,
    alpha: double,
    autotune: int,
    tolerance: double,
    stopWtheta: int,
    param: QudaGaugeParam,
    timeinfo: List[double, 3],
) -> int:
    """
    Gauge fixing with Steepest descent method with FFTs with support for single GPU only.
    @param[in,out] gauge:
        gauge field to be fixed
    @param[in] gauge_dir:
        3 for Coulomb gauge fixing, other for Landau gauge fixing
    @param[in] Nsteps:
        maximum number of steps to perform gauge fixing
    @param[in] verbose_interval:
        print gauge fixing info when iteration count is a multiple of this
    @param[in] alpha:
        gauge fixing parameter of the method, most common value is 0.08
    @param[in] autotune:
        1 to autotune the method, i.e., if the Fg inverts its tendency we decrease the alpha value
    @param[in] tolerance:
        torelance value to stop the method, if this value is zero then the method stops when
        iteration reachs the maximum number of steps defined by Nsteps
    @param[in] stopWtheta:
        0 for MILC criterion and 1 to use the theta value
    @param[in] param:
        The parameters of the external fields and the computation settings
    @param[out] timeinfo:
    """
    ...

class QudaQuarkSmearParam:
    def __init__(self) -> None: ...
    # def __repr__(self) -> str: ...

    inv_param: QudaInvertParam
    n_steps: int
    width: double
    compute_2link: int
    delete_2link: int
    t0: int
    gflops: int
