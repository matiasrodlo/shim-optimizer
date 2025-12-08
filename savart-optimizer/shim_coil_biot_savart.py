"""
2D Shim-Coil Optimizer using Biot-Savart Law

This is a 2D shim-coil optimizer that uses REAL B0 field map data from a BIDS dataset.
It uses the Biot-Savart formula to compute magnetic fields from circular shim loops
and optimizes loop currents to minimize field variance within the ROI.

The script:
1. Loads real B0 field map data from a BIDS dataset
2. Places circular shim loops around an imaging ROI
3. Optimizes loop currents to minimize variance of (B0 + shim) within the ROI
   using Tikhonov regularization

Note: This is a 2D model (loops in xy-plane, field computed at z=0) and omits
coil coupling and full 3D effects for simplicity. The dataset is REQUIRED.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
from pathlib import Path

# Import dependencies with error handling
try:
    from scipy import optimize
    from scipy import special
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Suggested: pip install numpy scipy matplotlib")
    sys.exit(1)

try:
    from skimage import transform
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    from bids import BIDSLayout
    HAS_PYBIDS = True
except ImportError:
    HAS_PYBIDS = False

# Import validation utilities
try:
    from validation_utils import validate_optimization_result, generate_validation_summary
    HAS_VALIDATION = True
except ImportError:
    HAS_VALIDATION = False

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(verbose=False):
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTDIR = "analysis"  # output folder
GRID_N = 200  # grid resolution (max 300 recommended)
GRID_FOV_MM = 200.0  # field-of-view in mm
N_LOOPS = 8  # number of shim loops placed around the imaging ROI
R_COIL_MM = 80.0  # radius (mm) of coil ring where loops are mounted
LOOP_RADIUS_MM = 10.0  # physical radius of each circular loop conductor (mm)
ROI_RADIUS_MM = 25.0  # radius of central ROI to optimize (mm)
INITIAL_WEIGHT = 0.2  # initial current amplitude for each loop (arbitrary units)
BOUNDS = (-1.0, 1.0)  # allowed current weight bounds for optimizer
ALPHA = 1e-3  # Tikhonov regularization on weights
OPT_METHOD = "L-BFGS-B"  # scipy.optimize method
MAXITER = 500
DOWNSAMPLE_MAX = 300
USE_REPO_B0 = True  # optional: if True and dataset B0 provided, compare improvement on repo B0 (cautious)
RANDOM_SEED = 42

# Dataset directory - automatically detected from script location
# Try relative path first, then environment variable, then current directory
def find_dataset_directory():
    """Find dataset directory using multiple strategies."""
    # Strategy 1: Relative to script location (one level up)
    relative_dataset = os.path.join(SCRIPT_DIR, "..", "dataset")
    if os.path.exists(relative_dataset):
        return os.path.abspath(relative_dataset)
    
    # Strategy 2: Two levels up (for backward compatibility)
    relative_dataset2 = os.path.join(SCRIPT_DIR, "..", "..", "dataset")
    if os.path.exists(relative_dataset2):
        return os.path.abspath(relative_dataset2)
    
    # Strategy 3: Environment variable
    env_dataset = os.environ.get('BIDS_DATASET_DIR')
    if env_dataset and os.path.exists(env_dataset):
        return os.path.abspath(env_dataset)
    
    # Strategy 4: Current directory
    current_dataset = os.path.join(os.getcwd(), "dataset")
    if os.path.exists(current_dataset):
        return os.path.abspath(current_dataset)
    
    # Strategy 5: Parent of current directory
    parent_dataset = os.path.join(os.path.dirname(os.getcwd()), "dataset")
    if os.path.exists(parent_dataset):
        return os.path.abspath(parent_dataset)
    
    return None

DATASET_DIR = find_dataset_directory()

# ============================================================================
# INPUT VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration parameters."""
    errors = []
    
    if GRID_N <= 0:
        errors.append(f"GRID_N must be positive, got {GRID_N}")
    if GRID_N > DOWNSAMPLE_MAX:
        errors.append(f"GRID_N ({GRID_N}) exceeds DOWNSAMPLE_MAX ({DOWNSAMPLE_MAX})")
    
    if GRID_FOV_MM <= 0:
        errors.append(f"GRID_FOV_MM must be positive, got {GRID_FOV_MM}")
    
    if ROI_RADIUS_MM >= GRID_FOV_MM / 2:
        errors.append(f"ROI_RADIUS_MM ({ROI_RADIUS_MM}) must be < GRID_FOV_MM/2 ({GRID_FOV_MM/2})")
    
    if N_LOOPS <= 0:
        errors.append(f"N_LOOPS must be positive, got {N_LOOPS}")
    
    if R_COIL_MM <= ROI_RADIUS_MM:
        errors.append(f"R_COIL_MM ({R_COIL_MM}) should be > ROI_RADIUS_MM ({ROI_RADIUS_MM})")
    
    if ALPHA < 0:
        errors.append(f"ALPHA must be non-negative, got {ALPHA}")
    
    if len(BOUNDS) != 2 or BOUNDS[0] >= BOUNDS[1]:
        errors.append(f"BOUNDS must be (min, max) with min < max, got {BOUNDS}")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True


def load_bids_fieldmap(dataset_dir, subject='01', acq=None, fmap_type='anat', logger=None):
    """
    Load BIDS-compliant field map with metadata.
    
    Parameters
    ----------
    dataset_dir : str
        Path to BIDS dataset
    subject : str
        Subject ID (e.g., '01')
    acq : str, optional
        Acquisition type (e.g., 'CP', 'CoV')
    fmap_type : str
        'anat' or 'famp' for field map type
    logger : logging.Logger, optional
        Logger instance
    
    Returns
    -------
    data : ndarray
        Field map data
    metadata : dict
        BIDS metadata from JSON sidecar
    affine : ndarray
        Affine transformation matrix
    nii_file : str
        Path to NIfTI file
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if not HAS_NIBABEL:
        raise ImportError("nibabel is required for loading field maps")
    
    # Try using pybids if available
    if HAS_PYBIDS:
        try:
            layout = BIDSLayout(dataset_dir, validate=False)
            
            # Build query
            query = {
                'subject': subject,
                'datatype': 'fmap',
                'suffix': 'TB1TFL',
                'extension': '.nii.gz',
                'return_type': 'filename'
            }
            
            if acq:
                query['acquisition'] = acq
            
            # Filter by fmap_type (anat or famp)
            files = layout.get(**query)
            if files:
                # Filter by acquisition name pattern
                pattern = f"acq-{fmap_type}" if not acq else f"acq-{fmap_type}{acq}"
                files = [f for f in files if pattern in os.path.basename(f)]
            
            if not files:
                raise FileNotFoundError(
                    f"No field map found for subject {subject}, "
                    f"acq {acq}, type {fmap_type}"
                )
            
            nii_file = files[0]
            logger.info(f"Using pybids to load: {nii_file}")
            
        except Exception as e:
            logger.warning(f"pybids failed ({e}), falling back to glob pattern")
            nii_file = None
    else:
        nii_file = None
    
    # Fallback to glob pattern if pybids not available or failed
    if nii_file is None:
        import glob
        subject_str = f"sub-{subject:02d}" if isinstance(subject, int) else f"sub-{subject}"
        pattern = os.path.join(dataset_dir, subject_str, "fmap", f"*_acq-{fmap_type}*_TB1TFL.nii.gz")
        if acq:
            pattern = os.path.join(dataset_dir, subject_str, "fmap", f"*_acq-{fmap_type}{acq}_TB1TFL.nii.gz")
        
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(
                f"No field map found matching pattern: {pattern}"
            )
        nii_file = files[0]
        logger.info(f"Using glob pattern to load: {nii_file}")
    
    # Load NIfTI file
    try:
        img = nib.load(nii_file)
        data = img.get_fdata()
        affine = img.affine
    except Exception as e:
        raise RuntimeError(f"Error loading NIfTI file {nii_file}: {e}")
    
    # Load JSON metadata
    json_file = nii_file.replace('.nii.gz', '.json')
    if not os.path.exists(json_file):
        logger.warning(f"JSON sidecar not found: {json_file}, using empty metadata")
        metadata = {}
    else:
        try:
            with open(json_file, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata from: {json_file}")
        except Exception as e:
            logger.warning(f"Error loading JSON metadata: {e}, using empty metadata")
            metadata = {}
    
    return data, metadata, affine, nii_file


def load_and_resample_b0(dataset_dir, grid_x, grid_y, subject='01', acq=None, logger=None):
    """
    Load B0 field map from dataset and resample to match optimization grid.
    
    Parameters
    ----------
    dataset_dir : str
        Path to BIDS dataset directory
    grid_x : ndarray, shape (Ny, Nx)
        X coordinates of optimization grid (in mm)
    grid_y : ndarray, shape (Ny, Nx)
        Y coordinates of optimization grid (in mm)
    subject : str
        Subject ID (e.g., '01')
    acq : str, optional
        Acquisition type (e.g., 'CP', 'CoV', 'patient')
    logger : logging.Logger, optional
        Logger instance
    
    Returns
    -------
    b0_resampled : ndarray, shape (Ny, Nx)
        B0 field map resampled to match grid
    metadata : dict
        BIDS metadata
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if not HAS_NIBABEL:
        raise ImportError("nibabel is required for loading B0 field maps")
    
    # Load B0 field map
    logger.info(f"\nLoading B0 field map from dataset...")
    logger.info(f"  Subject: {subject}, Acquisition: {acq}")
    b0_data, metadata, affine, nii_file = load_bids_fieldmap(
        dataset_dir, subject=subject, acq=acq, fmap_type='anat', logger=logger
    )
    
    logger.info(f"Loaded B0 map: {nii_file}")
    logger.info(f"  Original shape: {b0_data.shape}")
    
    # Handle 3D/4D data - select central slice
    if len(b0_data.shape) == 3:
        # 3D: select central slice
        central_slice = b0_data.shape[2] // 2
        b0_slice = b0_data[:, :, central_slice]
        logger.info(f"  Selected central slice: {central_slice}")
    elif len(b0_data.shape) == 4:
        # 4D: select central slice and first volume
        central_slice = b0_data.shape[2] // 2
        b0_slice = b0_data[:, :, central_slice, 0]
        logger.info(f"  Selected central slice: {central_slice}, volume 0")
    else:
        b0_slice = b0_data
    
    # Get voxel size from metadata or affine
    if metadata and 'SliceThickness' in metadata:
        voxel_size_z = metadata.get('SliceThickness', 1.0)
        voxel_size_xy = np.abs(np.diag(affine[:2, :2]))
        if len(voxel_size_xy) == 2:
            voxel_size_x, voxel_size_y = voxel_size_xy
        else:
            voxel_size_x = voxel_size_y = voxel_size_xy[0] if len(voxel_size_xy) > 0 else 1.0
    else:
        voxel_sizes = np.abs(np.diag(affine[:3, :3]))
        if len(voxel_sizes) >= 2:
            voxel_size_x, voxel_size_y = voxel_sizes[0], voxel_sizes[1]
        else:
            voxel_size_x = voxel_size_y = 1.0
    
    logger.info(f"  Estimated voxel size: {voxel_size_x:.2f} x {voxel_size_y:.2f} mm")
    
    # Resample to match grid
    Ny, Nx = grid_x.shape
    logger.info(f"  Resampling to grid size: {Ny} x {Nx}")
    
    if HAS_SKIMAGE:
        b0_resampled = transform.resize(b0_slice, (Ny, Nx), order=1, anti_aliasing=True)
    else:
        # Simple downsampling using scipy
        from scipy.ndimage import zoom
        zoom_factors = (Ny / b0_slice.shape[0], Nx / b0_slice.shape[1])
        b0_resampled = zoom(b0_slice, zoom_factors, order=1)
    
    logger.info(f"  Resampled shape: {b0_resampled.shape}")
    logger.info(f"  B0 range: [{np.min(b0_resampled):.6f}, {np.max(b0_resampled):.6f}]")
    
    return b0_resampled, metadata


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def make_loop_positions(n_loops, R_coil_mm, loop_radius_mm):
    """
    Generate positions and geometry for shim loops placed evenly around a circle.
    
    Parameters
    ----------
    n_loops : int
        Number of loops
    R_coil_mm : float
        Radius of coil ring (mm)
    loop_radius_mm : float
        Physical radius of each loop (mm)
    
    Returns
    -------
    loop_centers : ndarray, shape (n_loops, 2)
        (x, y) positions of loop centers in mm
    loop_radius : float
        Physical loop radius (same for all)
    """
    angles = np.linspace(0, 2 * np.pi, n_loops, endpoint=False)
    loop_centers = np.zeros((n_loops, 2))
    loop_centers[:, 0] = R_coil_mm * np.cos(angles)
    loop_centers[:, 1] = R_coil_mm * np.sin(angles)
    return loop_centers, loop_radius_mm


def compute_bz_grid_for_loop(loop_center, loop_radius_mm, grid_x, grid_y, Nseg=64):
    """
    Numerically compute Bz field on the plane from a circular loop using Biot-Savart.
    
    Approximates the loop as Nseg straight segments and sums Biot-Savart contributions.
    This is a 2D model: loops are in the xy-plane (z=0) with normal along z.
    Field is computed at z=0 (imaging plane).
    
    Uses a numerically stable Biot-Savart formula for straight wire segments:
    Bz = (μ₀ I / 4π) * segment_length * (r × dl) / |r|³
    
    where:
    - r is vector from segment midpoint to observation point
    - dl is segment direction vector
    - Cross product gives field direction (right-hand rule)
    
    This formulation is more numerically stable than angle-based formulas,
    especially for points near the wire.
    
    Parameters
    ----------
    loop_center : array-like, shape (2,)
        (x, y) center of loop in mm
    loop_radius_mm : float
        Radius of loop in mm
    grid_x : ndarray, shape (Ny, Nx)
        X coordinates of grid points in mm
    grid_y : ndarray, shape (Ny, Nx)
        Y coordinates of grid points in mm
    Nseg : int
        Number of segments to discretize the loop
    
    Returns
    -------
    Bz : ndarray, shape (Ny, Nx)
        Bz field component (arbitrary units proportional to current)
    """
    # Discretize loop into segments
    seg_angles = np.linspace(0, 2 * np.pi, Nseg, endpoint=False)
    seg_start = np.zeros((Nseg, 2))
    seg_start[:, 0] = loop_center[0] + loop_radius_mm * np.cos(seg_angles)
    seg_start[:, 1] = loop_center[1] + loop_radius_mm * np.sin(seg_angles)
    
    # End points (next segment start)
    seg_end = np.zeros((Nseg, 2))
    seg_end[:, 0] = loop_center[0] + loop_radius_mm * np.cos(seg_angles + 2 * np.pi / Nseg)
    seg_end[:, 1] = loop_center[1] + loop_radius_mm * np.sin(seg_angles + 2 * np.pi / Nseg)
    
    # Segment vectors
    seg_vec = seg_end - seg_start
    seg_length = np.linalg.norm(seg_vec, axis=1)
    
    # Initialize Bz field
    Bz = np.zeros_like(grid_x)
    Ny, Nx = grid_x.shape
    
    # Biot-Savart constant (arbitrary units, will be scaled later to match B0)
    # Use larger value to reduce extreme scaling factors later
    mu0_over_4pi = 1000.0  # Arbitrary scaling to get reasonable field magnitudes
    
    # For each segment, compute field using numerically stable Biot-Savart formula
    # For a straight wire segment in 2D (xy-plane), Bz is computed as:
    # Bz = (μ₀ I / 4π) * segment_length * (r × dl) / |r|³
    # where r is vector from segment to observation point
    for i in range(Nseg):
        # Unit vector along segment direction
        seg_dir = seg_vec[i] / (seg_length[i] + 1e-12)
        
        # Vector from segment midpoint to each grid point (more stable than endpoints)
        seg_mid = (seg_start[i] + seg_end[i]) / 2
        r = np.stack([
            grid_x - seg_mid[0],
            grid_y - seg_mid[1]
        ], axis=-1)  # Shape: (Ny, Nx, 2)
        
        # Distance from segment midpoint to observation points
        r_mag = np.linalg.norm(r, axis=-1)  # Shape: (Ny, Nx)
        # Use larger minimum distance to avoid extreme values near the wire
        r_mag = np.maximum(r_mag, loop_radius_mm * 0.1)  # Min distance = 10% of loop radius
        
        # Cross product (r × seg_dir) for z-component
        # For 2D: cross_z = r_x * seg_dir_y - r_y * seg_dir_x
        cross_z = r[..., 0] * seg_dir[1] - r[..., 1] * seg_dir[0]
        
        # Biot-Savart contribution: Bz ∝ (r × dl) / |r|³
        # Scale by segment length and use r³ for proper field decay
        r_cubed = r_mag**3
        contribution = seg_length[i] * cross_z / r_cubed
        
        # Clip to avoid numerical overflow and replace non-finite values
        contribution = np.nan_to_num(contribution, nan=0.0, posinf=0.0, neginf=0.0)
        contribution = np.clip(contribution, -10.0, 10.0)  # More conservative clipping
        
        # Apply scaling factor
        Bz += contribution * mu0_over_4pi
    
    # Final cleanup of any remaining non-finite values
    Bz = np.nan_to_num(Bz, nan=0.0, posinf=0.0, neginf=0.0)
    
    return Bz


def compute_field_matrix(loops, grid_x, grid_y):
    """
    Compute Bz field maps for all loops and create design matrix.
    
    Parameters
    ----------
    loops : tuple
        (loop_centers, loop_radius) from make_loop_positions
    grid_x : ndarray, shape (Ny, Nx)
        X coordinates
    grid_y : ndarray, shape (Ny, Nx)
        Y coordinates
    
    Returns
    -------
    M : ndarray, shape (n_loops, Ny, Nx)
        Stacked field maps for each loop
    A : ndarray, shape (Npix, n_loops)
        Flattened design matrix (Npix = Ny * Nx)
    """
    loop_centers, loop_radius = loops
    n_loops = len(loop_centers)
    Ny, Nx = grid_x.shape
    
    M = np.zeros((n_loops, Ny, Nx))
    
    logging.getLogger(__name__).info(f"Computing field maps for {n_loops} loops...")
    for k in range(n_loops):
        M[k] = compute_bz_grid_for_loop(loop_centers[k], loop_radius, grid_x, grid_y)
        
        # Clean up non-finite values immediately
        n_bad = np.sum(~np.isfinite(M[k]))
        if n_bad > 0:
            logging.getLogger(__name__).warning(f"  Loop {k}: {n_bad} non-finite values, replacing with zeros")
            M[k] = np.nan_to_num(M[k], nan=0.0, posinf=0.0, neginf=0.0)
        
        norm_k = np.linalg.norm(M[k])
        logging.getLogger(__name__).debug(f"  Loop {k}: L2 norm = {norm_k:.4f}")
    
    # Flatten to design matrix
    Npix = Ny * Nx
    A = M.reshape(n_loops, Npix).T  # Shape: (Npix, n_loops)
    
    return M, A


def make_roi_mask(grid_x, grid_y, roi_radius_mm):
    """
    Create boolean mask for central circular ROI.
    
    Parameters
    ----------
    grid_x : ndarray
        X coordinates
    grid_y : ndarray
        Y coordinates
    roi_radius_mm : float
        ROI radius in mm
    
    Returns
    -------
    mask : ndarray, bool
        True inside ROI
    """
    r = np.sqrt(grid_x**2 + grid_y**2)
    return r <= roi_radius_mm


def baseline_field_and_metrics(A, weights0, roi_mask, baseline_field=None):
    """
    Compute combined field and metrics inside ROI.
    
    Parameters
    ----------
    A : ndarray, shape (Npix, n_loops)
        Design matrix
    weights0 : ndarray, shape (n_loops,)
        Current weights
    roi_mask : ndarray, bool
        ROI mask
    baseline_field : ndarray, shape (Ny, Nx), optional
        Baseline B0 field map. If provided, total field = baseline + shim.
    
    Returns
    -------
    field : ndarray, shape (Ny, Nx)
        Combined field map (baseline + shim if baseline provided, else just shim)
    metrics : dict
        Dictionary with 'mean', 'std', 'CV' inside ROI
    """
    Ny, Nx = roi_mask.shape
    # Use error handling for numerical stability
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        shim_flat = A @ weights0
    
    # Clean up non-finite values
    shim_flat = np.nan_to_num(shim_flat, nan=0.0, posinf=0.0, neginf=0.0)
    shim_field = shim_flat.reshape(Ny, Nx)
    
    # Clip shim field to prevent extreme values
    # This is critical for visualization and physical reasonableness
    if baseline_field is not None:
        # Use B0 std instead of max for more conservative clipping
        b0_std = np.std(baseline_field)
        # Allow shim field up to 3x the B0 std (reasonable for correction)
        # This is more conservative than using B0 range
        max_shim = b0_std * 3
        shim_field = np.clip(shim_field, -max_shim, max_shim)
    
    if baseline_field is not None:
        field = baseline_field + shim_field
    else:
        field = shim_field
    
    # Final cleanup
    field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)
    
    roi_field = field[roi_mask]
    mean_val = np.mean(roi_field)
    std_val = np.std(roi_field)
    cv_val = std_val / (np.abs(mean_val) + 1e-10)
    
    metrics = {
        'mean': mean_val,
        'std': std_val,
        'CV': cv_val
    }
    
    return field, metrics


def optimize_weights_tikhonov(A, roi_mask, alpha, bounds, w0, method, maxiter, baseline_field=None):
    """
    Optimize loop weights to minimize ROI variance with Tikhonov regularization.
    
    If baseline_field is provided, optimizes: minimize variance of (baseline + shim) in ROI
    Otherwise, optimizes: minimize variance of shim field in ROI
    
    Objective: minimize sum((f_total_roi - mean(f_total_roi))^2) + alpha * ||w||^2
    
    Parameters
    ----------
    A : ndarray, shape (Npix, n_loops)
        Design matrix
    roi_mask : ndarray, bool
        ROI mask
    alpha : float
        Regularization strength
    bounds : tuple
        (min, max) bounds for weights
    w0 : ndarray
        Initial weights
    method : str
        Optimization method
    maxiter : int
        Maximum iterations
    baseline_field : ndarray, shape (Ny, Nx), optional
        Baseline B0 field map (e.g., from dataset). If provided, optimization
        minimizes variance of (baseline + shim) instead of just shim.
    
    Returns
    -------
    w_opt : ndarray
        Optimized weights
    success : bool
        Optimizer success flag
    obj_value : float
        Final objective value
    history : dict
        Optimization history (objective, variance, regularization, iterations)
    """
    Ny, Nx = roi_mask.shape
    roi_flat = roi_mask.flatten()
    n_loops = A.shape[1]
    
    # Extract ROI rows from design matrix
    A_roi = A[roi_flat]
    
    # Extract baseline field in ROI if provided
    if baseline_field is not None:
        baseline_roi = baseline_field.flatten()[roi_flat]
    else:
        baseline_roi = np.zeros(len(A_roi))
    
    # Track optimization history
    history = {
        'objective': [],
        'variance': [],
        'regularization': [],
        'iteration': [],
        'grad_norm': []
    }
    
    def objective(w):
        """Objective function: variance in ROI + regularization."""
        # Use error handling for numerical stability
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            shim_roi = A_roi @ w
            f_total_roi = baseline_roi + shim_roi
            mean_f = np.mean(f_total_roi)
            # Normalize variance by number of pixels for better numerical stability
            variance = np.mean((f_total_roi - mean_f)**2)
            reg = alpha * np.mean(w**2)
            obj = variance + reg
            
            # Track history (only every 10th iteration to avoid overhead)
            if len(history['objective']) == 0 or len(history['objective']) % 10 == 0:
                history['objective'].append(float(obj))
                history['variance'].append(float(variance))
                history['regularization'].append(float(reg))
                history['iteration'].append(len(history['objective']) * 10)
        
        return obj
    
    def gradient(w):
        """Analytic gradient of objective."""
        # Use error handling for numerical stability
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            shim_roi = A_roi @ w
            f_total_roi = baseline_roi + shim_roi
            mean_f = np.mean(f_total_roi)
            
            # Gradient of variance term: d/dw mean((f_total - mean)^2)
            # = 2 * A_roi.T @ (f_total - mean) / n_roi
            # Since we normalized objective, gradient is also normalized
            grad_var = 2 * A_roi.T @ (f_total_roi - mean_f) / len(f_total_roi)
            
            # Gradient of regularization term (normalized)
            grad_reg = 2 * alpha * w / len(w)
            
            grad = grad_var + grad_reg
            
            # Track gradient norm (only every 10th call)
            if len(history['grad_norm']) < len(history['objective']):
                history['grad_norm'].append(float(np.linalg.norm(grad)))
        
        return grad
    
    # Optimize
    result = optimize.minimize(
        objective,
        w0,
        method=method,
        jac=gradient,
        bounds=[bounds] * n_loops,
        options={'maxiter': maxiter, 'disp': False}
    )
    
    # Add final result info to history
    history['n_iterations'] = result.nit
    history['n_function_evals'] = result.nfev if hasattr(result, 'nfev') else len(history['objective'])
    history['success'] = bool(result.success)
    history['message'] = str(result.message)
    history['final_objective'] = float(result.fun)
    history['initial_objective'] = float(history['objective'][0]) if history['objective'] else float('nan')
    
    return result.x, result.success, result.fun, history


def plot_before_after(field_before, field_after, roi_mask, loops, weights_before, weights_after, outpath):
    """
    Create multi-panel before/after comparison figure.
    
    Parameters
    ----------
    field_before : ndarray
        Baseline field map
    field_after : ndarray
        Optimized field map
    roi_mask : ndarray
        ROI mask
    loops : tuple
        (loop_centers, loop_radius)
    weights_before : ndarray
        Initial weights
    weights_after : ndarray
        Optimized weights
    outpath : str
        Output file path
    """
    loop_centers, _ = loops
    Ny, Nx = field_before.shape
    
    # Create grid for plotting
    x_plot = np.linspace(-GRID_FOV_MM/2, GRID_FOV_MM/2, Nx)
    y_plot = np.linspace(-GRID_FOV_MM/2, GRID_FOV_MM/2, Ny)
    X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
    
    # Check for NaN/Inf values and clean them
    n_bad_before = np.sum(~np.isfinite(field_before))
    n_bad_after = np.sum(~np.isfinite(field_after))
    
    if n_bad_before > 0:
        print(f"Warning: field_before contains {n_bad_before} non-finite values, cleaning...")
        field_before = np.nan_to_num(field_before, nan=0.0, posinf=0.0, neginf=0.0)
    
    if n_bad_after > 0:
        print(f"Warning: field_after contains {n_bad_after} non-finite values, cleaning...")
        field_after = np.nan_to_num(field_after, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Aggressive clipping based on ROI values for better visualization
    # Compute reasonable limits from ROI
    field_before_roi = field_before[roi_mask]
    field_after_roi = field_after[roi_mask]
    
    before_min, before_max = np.percentile(field_before_roi, [0.1, 99.9])
    after_min, after_max = np.percentile(field_after_roi, [0.1, 99.9])
    
    # Use same color scale for both panels for better comparison
    # This makes visual comparison more meaningful
    global_min = min(before_min, after_min)
    global_max = max(before_max, after_max)
    
    # Small margin for visualization
    margin = 0.2  # 20% margin (more conservative)
    value_range = global_max - global_min
    
    clip_min = global_min - margin * value_range
    clip_max = global_max + margin * value_range
    
    # Apply same clipping to both fields
    field_before_plot = np.clip(field_before, clip_min, clip_max)
    field_after_plot = np.clip(field_after, clip_min, clip_max)
    
    # Additionally, mask out areas very close to loops to avoid artifacts
    # Create exclusion zones around each loop position
    loop_exclusion_radius = 15.0  # mm - don't visualize field within this radius of loops
    exclusion_mask = np.ones_like(field_after, dtype=bool)
    
    # Recreate grid for distance calculation
    Ny, Nx = field_after.shape
    fov_mm = 100.0  # Assume 200mm FOV centered at origin
    x_coords = np.linspace(-fov_mm, fov_mm, Nx)
    y_coords = np.linspace(-fov_mm, fov_mm, Ny)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    
    for loop_pos in loop_centers:
        dist_to_loop = np.sqrt((X_grid - loop_pos[0])**2 + (Y_grid - loop_pos[1])**2)
        exclusion_mask &= (dist_to_loop > loop_exclusion_radius)
    
    # Apply exclusion mask to after field (set excluded areas to NaN for white space)
    field_after_plot = np.where(exclusion_mask, field_after_plot, np.nan)
    
    # Debug: print field ranges
    print(f"Plot debug:")
    print(f"  field_before: original range=[{np.min(field_before):.1f}, {np.max(field_before):.1f}]")
    print(f"  field_before: clipped range=[{np.min(field_before_plot):.1f}, {np.max(field_before_plot):.1f}]")
    print(f"  field_after: original range=[{np.min(field_after):.1f}, {np.max(field_after):.1f}]")
    print(f"  field_after: clipped range=[{np.min(field_after_plot):.1f}, {np.max(field_after_plot):.1f}]")
    print(f"  ROI mask: {np.sum(roi_mask)} pixels")
    
    # Create 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Top-left: before (B0 baseline with no shim)
    ax = axes[0, 0]
    
    # Use clipped field for plotting
    vmin_before = np.min(field_before_plot)
    vmax_before = np.max(field_before_plot)
    
    print(f"  Before panel: vmin={vmin_before:.1f}, vmax={vmax_before:.1f}")
    
    # Create contour plot with explicit levels
    try:
        levels_before = np.linspace(vmin_before, vmax_before, 21)
        im = ax.contourf(X_plot, Y_plot, field_before_plot, levels=levels_before, cmap='RdBu_r')
        ax.contour(X_plot, Y_plot, roi_mask.astype(float), levels=[0.5], colors='black', linewidths=2)
        ax.scatter(loop_centers[:, 0], loop_centers[:, 1], c='gray', 
                   s=100, edgecolors='black', linewidths=1, alpha=0.5)
        ax.set_title(f'Before: B0 Baseline (no shim)\nROI std = {np.std(field_before[roi_mask]):.1f}')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Bz (arb. units)')
    except Exception as e:
        print(f"  Error creating before panel: {e}")
        ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, ha='center')
    
    # Top-right: after (B0 + optimized shim)
    ax = axes[0, 1]
    
    # Use clipped field for plotting
    vmin_after = np.min(field_after_plot)
    vmax_after = np.max(field_after_plot)
    
    print(f"  After panel: vmin={vmin_after:.1f}, vmax={vmax_after:.1f}")
    
    # Create contour plot with explicit levels
    try:
        levels_after = np.linspace(vmin_after, vmax_after, 21)
        im = ax.contourf(X_plot, Y_plot, field_after_plot, levels=levels_after, cmap='RdBu_r')
        ax.contour(X_plot, Y_plot, roi_mask.astype(float), levels=[0.5], colors='black', linewidths=2)
        ax.scatter(loop_centers[:, 0], loop_centers[:, 1], c=weights_after, 
                   s=100, cmap='RdBu_r', edgecolors='black', linewidths=1, vmin=-1, vmax=1)
        for k, (x, y) in enumerate(loop_centers):
            ax.annotate(f'{weights_after[k]:.2f}', (x, y), fontsize=8, ha='center', va='center')
        ax.set_title(f'After: B0 + Optimized Shim\nROI std = {np.std(field_after[roi_mask]):.1f}')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Bz (arb. units)')
    except Exception as e:
        print(f"  Error creating after panel: {e}")
        ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, ha='center')
    
    # Bottom-left: difference (improvement)
    ax = axes[1, 0]
    # Show std reduction in ROI
    std_before = np.std(field_before[roi_mask])
    std_after = np.std(field_after[roi_mask])
    improvement = 100 * (1 - std_after / std_before)
    
    # Plot field difference (use clipped fields)
    field_diff = field_after_plot - field_before_plot
    im = ax.contourf(X_plot, Y_plot, field_diff, levels=20, cmap='RdBu_r')
    ax.contour(X_plot, Y_plot, roi_mask.astype(float), levels=[0.5], colors='black', linewidths=2)
    ax.set_title(f'Shim Field Applied\n{improvement:.1f}% std reduction')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Shim Bz (arb. units)')
    
    # Bottom-right: weights bar chart
    ax = axes[1, 1]
    x_pos = np.arange(len(weights_before))
    width = 0.35
    
    # Plot bars with distinct colors
    bars_before = ax.bar(x_pos - width/2, weights_before, width, label='Before (no shim)', 
                         alpha=0.8, color='lightblue', edgecolor='blue', linewidth=1.5)
    bars_after = ax.bar(x_pos + width/2, weights_after, width, label='After (optimized)', 
                        alpha=0.8, color='lightcoral', edgecolor='red', linewidth=1.5)
    
    # Add horizontal line at zero for reference
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    ax.set_xlabel('Loop Index')
    ax.set_ylabel('Weight')
    ax.set_title('Loop Weights Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(i) for i in x_pos])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([-1.1, 1.1])
    
    # Add text annotation for zero weights
    if np.allclose(weights_before, 0):
        ax.text(0.5, 0.95, 'Before: All weights = 0 (no shim applied)', 
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                fontsize=9)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {outpath}")


def save_weights_csv(weights, loops, fname):
    """
    Save weights to CSV with loop positions.
    
    Parameters
    ----------
    weights : ndarray
        Loop weights
    loops : tuple
        (loop_centers, loop_radius)
    fname : str
        Output filename
    """
    loop_centers, _ = loops
    data = {
        'loop_index': np.arange(len(weights)),
        'x_mm': loop_centers[:, 0],
        'y_mm': loop_centers[:, 1],
        'weight': weights
    }
    
    # Simple CSV writing
    with open(fname, 'w') as f:
        f.write('loop_index,x_mm,y_mm,weight\n')
        for i in range(len(weights)):
            f.write(f'{i},{loop_centers[i,0]:.4f},{loop_centers[i,1]:.4f},{weights[i]:.6f}\n')
    
    logging.getLogger(__name__).info(f"Saved weights CSV: {fname}")


def maybe_compare_on_repo_b0(DATASET_DIR, grid_x, grid_y, roi_mask, weights, loops, fname, 
                              subject='01', acq=None, logger=None):
    """
    Optional comparison with repository B0 data (if available).
    
    WARNING: This is illustrative only. The comparison should be interpreted with caution.
    
    Parameters
    ----------
    DATASET_DIR : str
        Dataset directory path
    grid_x : ndarray
        Grid X coordinates
    grid_y : ndarray
        Grid Y coordinates
    roi_mask : ndarray
        ROI mask
    weights : ndarray
        Optimized weights
    loops : tuple
        (loop_centers, loop_radius)
    fname : str
        Output CSV filename
    subject : str
        Subject ID (e.g., '01')
    acq : str, optional
        Acquisition type (e.g., 'CP', 'CoV')
    logger : logging.Logger, optional
        Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if not USE_REPO_B0 or DATASET_DIR is None:
        logger.debug("Skipping repo B0 comparison: USE_REPO_B0=False or DATASET_DIR=None")
        return
    
    if not HAS_NIBABEL:
        logger.warning("nibabel not available, skipping repo B0 comparison")
        return
    
    logger.info("\n" + "="*60)
    logger.info("WARNING: Comparing with repository B0 data")
    logger.info("This is illustrative only.")
    logger.info("Results should be interpreted with extreme caution.")
    logger.info("="*60 + "\n")
    
    try:
        # Use BIDS-compliant loading
        b0_data, metadata, affine, nii_file = load_bids_fieldmap(
            DATASET_DIR, subject=subject, acq=acq, fmap_type='anat', logger=logger
        )
        
        logger.info(f"Loaded B0 map: {nii_file}")
        if metadata:
            logger.info(f"  SliceThickness: {metadata.get('SliceThickness', 'N/A')} mm")
            logger.info(f"  SpacingBetweenSlices: {metadata.get('SpacingBetweenSlices', 'N/A')} mm")
        
        # Handle 3D/4D data - select central slice
        if b0_data.ndim == 3:
            # Find slice with largest non-zero area
            slice_areas = [np.sum(np.abs(b0_data[:, :, z]) > 0) for z in range(b0_data.shape[2])]
            z_slice = np.argmax(slice_areas) if max(slice_areas) > 0 else b0_data.shape[2] // 2
            b0_slice = b0_data[:, :, z_slice]
            logger.info(f"Selected slice {z_slice} from 3D data")
        elif b0_data.ndim == 4:
            z_slice = b0_data.shape[2] // 2
            t_slice = 0  # take first time point
            b0_slice = b0_data[:, :, z_slice, t_slice]
            logger.info(f"Selected slice {z_slice}, time {t_slice} from 4D data")
        elif b0_data.ndim == 2:
            b0_slice = b0_data
            logger.info("Using 2D data directly")
        else:
            logger.warning(f"Unsupported data dimensions: {b0_data.ndim}D")
            return
        
        # Get voxel size from metadata or affine
        if metadata and 'SliceThickness' in metadata:
            # Use metadata if available
            voxel_size_z = metadata.get('SliceThickness', 1.0)
            # Estimate in-plane voxel size from affine
            voxel_size_xy = np.abs(np.diag(affine[:2, :2]))
            if len(voxel_size_xy) == 2:
                voxel_size_x, voxel_size_y = voxel_size_xy
            else:
                voxel_size_x = voxel_size_y = voxel_size_xy[0] if len(voxel_size_xy) > 0 else 1.0
        else:
            # Fallback: estimate from affine
            voxel_sizes = np.abs(np.diag(affine[:3, :3]))
            if len(voxel_sizes) >= 2:
                voxel_size_x, voxel_size_y = voxel_sizes[0], voxel_sizes[1]
            else:
                voxel_size_x = voxel_size_y = 1.0
        
        logger.info(f"Estimated voxel size: {voxel_size_x:.2f} x {voxel_size_y:.2f} mm")
        
        # Downsample to match grid
        Ny, Nx = grid_x.shape
        if HAS_SKIMAGE:
            b0_downsampled = transform.resize(b0_slice, (Ny, Nx), order=1, anti_aliasing=True)
        else:
            # Simple downsampling
            from scipy.ndimage import zoom
            zoom_factors = (Ny / b0_slice.shape[0], Nx / b0_slice.shape[1])
            b0_downsampled = zoom(b0_slice, zoom_factors, order=1)
        
        # Compute simulated shim field
        _, A = compute_field_matrix(loops, grid_x, grid_y)
        shim_field_flat = A @ weights
        shim_field = shim_field_flat.reshape(Ny, Nx)
        
        # Normalize shim field to have similar scale (arbitrary scaling)
        # This is an illustrative comparison
        b0_roi = b0_downsampled[roi_mask]
        shim_roi = shim_field[roi_mask]
        if np.std(shim_roi) > 1e-10:
            scale = np.std(b0_roi) / np.std(shim_roi) * 0.1  # Arbitrary scaling factor
            shim_field_scaled = shim_field * scale
            logger.info(f"Applied scaling factor: {scale:.6f} (arbitrary, for illustration only)")
        else:
            shim_field_scaled = shim_field
        
        # Apply correction (subtract shim field)
        b0_corrected = b0_downsampled - shim_field_scaled
        
        # Compute metrics
        b0_roi_before = b0_downsampled[roi_mask]
        b0_roi_after = b0_corrected[roi_mask]
        
        std_before = np.std(b0_roi_before)
        std_after = np.std(b0_roi_after)
        percent_reduction = 100 * (1 - std_after / (std_before + 1e-10))
        
        # Save comparison
        import csv
        with open(fname, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['b0_std_before', std_before])
            writer.writerow(['b0_std_after', std_after])
            writer.writerow(['percent_reduction', percent_reduction])
            writer.writerow(['subject', subject])
            writer.writerow(['acquisition', acq if acq else 'default'])
            writer.writerow(['voxel_size_x_mm', voxel_size_x])
            writer.writerow(['voxel_size_y_mm', voxel_size_y])
            writer.writerow(['note', 'Illustrative comparison only'])
        
        logger.info(f"B0 comparison saved: {fname}")
        logger.info(f"  B0 std before: {std_before:.4f}")
        logger.info(f"  B0 std after: {std_after:.4f}")
        logger.info(f"  Percent reduction: {percent_reduction:.2f}%")
        
    except FileNotFoundError as e:
        logger.warning(f"Field map not found: {e}")
        logger.info("Skipping repo B0 comparison")
    except ImportError as e:
        logger.warning(f"Import error: {e}")
        logger.info("Skipping repo B0 comparison")
    except Exception as e:
        logger.error(f"Error loading/comparing B0 data: {e}", exc_info=True)
        logger.info("Skipping repo B0 comparison")


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='2D Shim-Coil Optimizer using Biot-Savart Law - Uses REAL B0 data from dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default=None,
        help='Path to BIDS dataset directory (REQUIRED - overrides auto-detection)'
    )
    
    parser.add_argument(
        '--subject',
        type=str,
        default='01',
        help='Subject ID for B0 field map (e.g., "01", "02")'
    )
    
    parser.add_argument(
        '--acq',
        type=str,
        default=None,
        help='Acquisition type for B0 field map (e.g., "CP", "CoV", "patient")'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (overrides OUTDIR config)'
    )
    
    parser.add_argument(
        '--no-repo-b0',
        action='store_true',
        help='Disable repository B0 comparison'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main script execution."""
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(verbose=args.verbose)
    
    # Override dataset directory if provided
    global DATASET_DIR
    if args.dataset_dir:
        if os.path.exists(args.dataset_dir):
            DATASET_DIR = os.path.abspath(args.dataset_dir)
            logger.info(f"Using dataset directory from command line: {DATASET_DIR}")
        else:
            logger.error(f"Dataset directory does not exist: {args.dataset_dir}")
            sys.exit(1)
    
    # Override USE_REPO_B0 if requested
    global USE_REPO_B0
    if args.no_repo_b0:
        USE_REPO_B0 = False
        logger.info("Repository B0 comparison disabled by command line")
    
    # Require dataset directory
    if DATASET_DIR is None:
        logger.error("Dataset directory not found. Please provide --dataset-dir or ensure dataset/ exists.")
        logger.error("The optimizer requires B0 field map data from the dataset.")
        sys.exit(1)
    
    # Validate configuration
    try:
        validate_config()
        logger.debug("Configuration validation passed")
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    
    # Create output directory
    if args.output_dir:
        outdir_full = os.path.abspath(args.output_dir)
    else:
        outdir_full = os.path.abspath(os.path.join(SCRIPT_DIR, OUTDIR))
    os.makedirs(outdir_full, exist_ok=True)
    logger.info(f"Output directory: {outdir_full}")
    
    # Log configuration
    logger.info("="*60)
    logger.info("2D Shim-Coil Optimizer (Biot-Savart)")
    logger.info("Using REAL B0 data from dataset (not synthetic)")
    logger.info("="*60)
    logger.info(f"OUTDIR: {outdir_full}")
    logger.info(f"GRID_N: {GRID_N}")
    logger.info(f"GRID_FOV_MM: {GRID_FOV_MM}")
    logger.info(f"N_LOOPS: {N_LOOPS}")
    logger.info(f"R_COIL_MM: {R_COIL_MM}")
    logger.info(f"LOOP_RADIUS_MM: {LOOP_RADIUS_MM}")
    logger.info(f"ROI_RADIUS_MM: {ROI_RADIUS_MM}")
    logger.info(f"INITIAL_WEIGHT: {INITIAL_WEIGHT}")
    logger.info(f"BOUNDS: {BOUNDS}")
    logger.info(f"ALPHA: {ALPHA}")
    logger.info(f"OPT_METHOD: {OPT_METHOD}")
    logger.info(f"MAXITER: {MAXITER}")
    logger.info(f"DATASET_DIR: {DATASET_DIR}")
    logger.info(f"Subject: {args.subject}, Acquisition: {args.acq}")
    logger.info("="*60 + "\n")
    
    # Check grid size
    grid_n = GRID_N
    if grid_n > DOWNSAMPLE_MAX:
        logger.warning(f"GRID_N ({grid_n}) > DOWNSAMPLE_MAX ({DOWNSAMPLE_MAX})")
        logger.warning(f"Reducing to {DOWNSAMPLE_MAX}")
        grid_n = DOWNSAMPLE_MAX
    
    # Build imaging grid
    logger.info(f"Creating {grid_n}x{grid_n} imaging grid...")
    x = np.linspace(-GRID_FOV_MM/2, GRID_FOV_MM/2, grid_n)
    y = np.linspace(-GRID_FOV_MM/2, GRID_FOV_MM/2, grid_n)
    grid_x, grid_y = np.meshgrid(x, y)
    
    # Load B0 field map from dataset
    try:
        baseline_b0, b0_metadata = load_and_resample_b0(
            DATASET_DIR, grid_x, grid_y, 
            subject=args.subject, acq=args.acq, logger=logger
        )
    except Exception as e:
        logger.error(f"Failed to load B0 field map from dataset: {e}")
        logger.error("The optimizer requires B0 field map data from the dataset.")
        sys.exit(1)
    
    # Generate loop positions
    logger.info(f"\nGenerating {N_LOOPS} loop positions...")
    loops = make_loop_positions(N_LOOPS, R_COIL_MM, LOOP_RADIUS_MM)
    loop_centers, loop_radius = loops
    logger.info("Loop centers (mm):")
    for k, (x, y) in enumerate(loop_centers):
        logger.info(f"  Loop {k}: ({x:.2f}, {y:.2f})")
    
    # Compute field matrix
    logger.info("\nComputing field matrix...")
    M, A = compute_field_matrix(loops, grid_x, grid_y)
    
    # Check field matrix quality
    logger.info(f"  Field matrix shape: {A.shape}")
    logger.debug(f"  Field matrix range: [{np.min(A):.6e}, {np.max(A):.6e}]")
    n_bad_before = np.sum(~np.isfinite(A))
    if n_bad_before > 0:
        logger.warning(f"  Field matrix contains {n_bad_before} non-finite values, will be cleaned")
    
    # Create ROI mask
    logger.info("\nCreating ROI mask...")
    roi_mask = make_roi_mask(grid_x, grid_y, ROI_RADIUS_MM)
    n_roi_pixels = np.sum(roi_mask)
    logger.info(f"ROI contains {n_roi_pixels} pixels")
    
    # Normalize and scale field matrix for optimal performance
    # Professional approach: normalize each loop, then scale globally
    logger.info("\nNormalizing and scaling field matrix...")
    b0_roi = baseline_b0[roi_mask]
    b0_std = np.std(b0_roi)
    b0_range = np.ptp(b0_roi)  # Peak-to-peak range
    
    # Extract ROI portion of field matrix
    A_roi = A[roi_mask.flatten()]
    
    # Step 1: Normalize each loop's field to unit standard deviation in ROI
    # This makes all loops contribute equally and prevents numerical issues
    logger.info("  Step 1: Normalizing each loop to unit std in ROI...")
    field_stds = np.std(A_roi, axis=0)
    
    # Check for zero-variance loops
    n_zero_loops = np.sum(field_stds < 1e-12)
    if n_zero_loops > 0:
        logger.warning(f"  {n_zero_loops} loops have near-zero field in ROI")
    
    # Normalize (avoid division by zero)
    normalization = np.where(field_stds > 1e-12, field_stds, 1.0)
    A_normalized = A / normalization[np.newaxis, :]
    M_normalized = M / normalization[:, np.newaxis, np.newaxis]
    
    # Step 2: Scale globally to match B0 field magnitude
    # Use standard deviation as it's more robust than max
    logger.info("  Step 2: Scaling to match B0 field magnitude...")
    A_roi_normalized = A_normalized[roi_mask.flatten()]
    
    # Compute expected shim field std with unit weights after normalization
    unit_weights = np.ones(N_LOOPS)
    shim_roi_normalized = A_roi_normalized @ unit_weights
    shim_std_normalized = np.std(shim_roi_normalized)
    
    if shim_std_normalized > 1e-10:
        # Scale so that unit weights produce field with std similar to B0 std
        # This ensures shim can effectively correct B0 variations
        global_scale = b0_std / shim_std_normalized
        
        # Apply reasonable limits to prevent numerical issues
        # Allow shim field up to 10x B0 std (strong correction capability)
        max_reasonable_scale = 10.0
        if global_scale > max_reasonable_scale:
            logger.warning(f"  Global scale ({global_scale:.2f}) exceeds reasonable limit")
            logger.warning(f"  Capping at {max_reasonable_scale:.2f} for numerical stability")
            global_scale = max_reasonable_scale
        
        A = A_normalized * global_scale
        M = M_normalized * global_scale
        
        logger.info(f"  B0 std in ROI: {b0_std:.2f}")
        logger.info(f"  B0 range in ROI: {b0_range:.2f}")
        logger.info(f"  Shim std (normalized, unit weights): {shim_std_normalized:.6f}")
        logger.info(f"  Global scaling factor: {global_scale:.2f}")
        logger.info(f"  Expected shim std with unit weights: {shim_std_normalized * global_scale:.2f}")
        
        # Verify final field matrix quality
        A_roi_final = A[roi_mask.flatten()]
        cond_num = np.linalg.cond(A_roi_final)
        logger.info(f"  Final condition number: {cond_num:.2e}")
        
        if cond_num > 1e10:
            logger.warning("  Field matrix is ill-conditioned, optimization may be challenging")
        
        # Check for any remaining numerical issues
        n_bad = np.sum(~np.isfinite(A))
        if n_bad > 0:
            logger.warning(f"  {n_bad} non-finite values in field matrix, replacing with zeros")
            A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
            M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info(f"  Final field matrix range: [{np.min(A):.2e}, {np.max(A):.2e}]")
    else:
        logger.error("  Shim field too weak even after normalization!")
        logger.error("  Check coil geometry and Biot-Savart computation")
        sys.exit(1)
    
    # Baseline field (using real B0 data, no shim initially)
    logger.info("\nComputing baseline field (real B0 data, no shim)...")
    weights0 = np.zeros(N_LOOPS)  # Start with zero shim
    field_before, metrics_before = baseline_field_and_metrics(A, weights0, roi_mask, baseline_field=baseline_b0)
    logger.info("Baseline metrics (ROI):")
    logger.info(f"  Mean: {metrics_before['mean']:.6f}")
    logger.info(f"  Std:  {metrics_before['std']:.6f}")
    logger.info(f"  CV:   {metrics_before['CV']:.6f}")
    
    # Save baseline map (optional)
    baseline_path = os.path.join(outdir_full, "biot_savart_baseline.png")
    plt.figure(figsize=(8, 8))
    plt.contourf(grid_x, grid_y, field_before, levels=20, cmap='RdBu_r')
    plt.contour(grid_x, grid_y, roi_mask.astype(float), levels=[0.5], colors='black', linewidths=2)
    plt.scatter(loop_centers[:, 0], loop_centers[:, 1], c='red', s=50, marker='o', edgecolors='black')
    plt.colorbar(label='Bz (arb. units)')
    plt.title('Baseline Field')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.axis('equal')
    plt.savefig(baseline_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved baseline map: {baseline_path}")
    
    # Optimization
    logger.info("\nOptimizing weights to minimize variance of (B0 + shim)...")
    logger.info(f"  Method: {OPT_METHOD}")
    logger.info(f"  Regularization (α): {ALPHA}")
    logger.info(f"  Weight bounds: {BOUNDS}")
    logger.info(f"  Max iterations: {MAXITER}")
    
    w_opt, success, obj_value, opt_history = optimize_weights_tikhonov(
        A, roi_mask, ALPHA, BOUNDS, weights0, OPT_METHOD, MAXITER, baseline_field=baseline_b0
    )
    
    # Log optimization results
    logger.info("\nOptimization Results:")
    logger.info(f"  Success: {success}")
    logger.info(f"  Message: {opt_history['message']}")
    logger.info(f"  Iterations: {opt_history['n_iterations']}")
    logger.info(f"  Function evaluations: {opt_history['n_function_evals']}")
    logger.info(f"  Initial objective: {opt_history['initial_objective']:.6f}")
    logger.info(f"  Final objective: {obj_value:.6f}")
    
    if opt_history['initial_objective'] > 0:
        obj_reduction = 100 * (1 - obj_value / opt_history['initial_objective'])
        logger.info(f"  Objective reduction: {obj_reduction:.2f}%")
    
    logger.info("\nOptimized weights:")
    for k, w in enumerate(w_opt):
        logger.info(f"  Loop {k}: {w:.6f}")
    
    # Optimized field
    logger.info("\nComputing optimized field (B0 + optimized shim)...")
    field_after, metrics_after = baseline_field_and_metrics(A, w_opt, roi_mask, baseline_field=baseline_b0)
    logger.info("Optimized metrics (ROI):")
    logger.info(f"  Mean: {metrics_after['mean']:.6f}")
    logger.info(f"  Std:  {metrics_after['std']:.6f}")
    logger.info(f"  CV:   {metrics_after['CV']:.6f}")
    
    # Percent reduction
    percent_reduction = 100 * (1 - metrics_after['std'] / (metrics_before['std'] + 1e-10))
    logger.info("\nImprovement:")
    logger.info(f"  Std reduction: {percent_reduction:.2f}%")
    
    # Validate optimization results
    if HAS_VALIDATION:
        logger.info("\nValidating optimization results...")
        validation_report = validate_optimization_result(
            w_opt, A, baseline_b0, roi_mask, BOUNDS, opt_history, logger=logger
        )
        generate_validation_summary(validation_report, logger=logger)
    else:
        validation_report = None
        logger.debug("Validation utilities not available")
    
    # Save optimized map (optional)
    optimized_path = os.path.join(outdir_full, "biot_savart_optimized.png")
    plt.figure(figsize=(8, 8))
    plt.contourf(grid_x, grid_y, field_after, levels=20, cmap='RdBu_r')
    plt.contour(grid_x, grid_y, roi_mask.astype(float), levels=[0.5], colors='black', linewidths=2)
    plt.scatter(loop_centers[:, 0], loop_centers[:, 1], c=w_opt, s=50, cmap='RdBu_r', 
                edgecolors='black', vmin=-1, vmax=1)
    plt.colorbar(label='Bz (arb. units)')
    plt.title('Optimized Field')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.axis('equal')
    plt.savefig(optimized_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved optimized map: {optimized_path}")
    
    # Before/after comparison
    logger.info("\nCreating before/after comparison...")
    before_after_path = os.path.join(outdir_full, "biot_savart_before_after.png")
    plot_before_after(field_before, field_after, roi_mask, loops, weights0, w_opt, before_after_path)
    
    # Save weights CSV
    weights_path = os.path.join(outdir_full, "biot_savart_weights.csv")
    save_weights_csv(w_opt, loops, weights_path)
    
    # Optional repo B0 comparison
    if USE_REPO_B0 and DATASET_DIR is not None:
        repo_comparison_path = os.path.join(outdir_full, "biot_savart_repo_comparison.csv")
        maybe_compare_on_repo_b0(
            DATASET_DIR, grid_x, grid_y, roi_mask, w_opt, loops, 
            repo_comparison_path, subject=args.subject, acq=args.acq, logger=logger
        )
    
    # Save stats CSV
    stats_path = os.path.join(outdir_full, "biot_savart_stats.csv")
    with open(stats_path, 'w') as f:
        f.write('metric,value\n')
        f.write(f'baseline_std,{metrics_before["std"]:.6f}\n')
        f.write(f'optimized_std,{metrics_after["std"]:.6f}\n')
        f.write(f'percent_reduction,{percent_reduction:.2f}\n')
        f.write(f'grid_N,{grid_n}\n')
        f.write(f'roi_radius_mm,{ROI_RADIUS_MM}\n')
        f.write(f'alpha,{ALPHA}\n')
        f.write(f'optimizer_success,{success}\n')
    logger.info(f"Saved stats CSV: {stats_path}")
    
    # Save comprehensive JSON report
    report_path = os.path.join(outdir_full, "optimization_report.json")
    
    # Helper function to convert numpy types to Python types for JSON serialization
    def convert_to_json_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj
    
    optimization_report = {
        'configuration': {
            'grid_n': int(grid_n),
            'grid_fov_mm': float(GRID_FOV_MM),
            'roi_radius_mm': float(ROI_RADIUS_MM),
            'n_loops': int(N_LOOPS),
            'coil_radius_mm': float(R_COIL_MM),
            'loop_radius_mm': float(LOOP_RADIUS_MM),
            'alpha': float(ALPHA),
            'bounds': [float(b) for b in BOUNDS],
            'method': str(OPT_METHOD),
            'maxiter': int(MAXITER)
        },
        'data': {
            'subject': str(args.subject),
            'acquisition': str(args.acq) if args.acq else None,
            'dataset_dir': str(DATASET_DIR)
        },
        'optimization': {
            'success': bool(success),
            'n_iterations': int(opt_history['n_iterations']),
            'n_function_evals': int(opt_history['n_function_evals']),
            'initial_objective': float(opt_history['initial_objective']),
            'final_objective': float(obj_value),
            'message': str(opt_history['message']),
            'history': {
                'objective': [float(x) for x in opt_history['objective']],
                'variance': [float(x) for x in opt_history['variance']],
                'regularization': [float(x) for x in opt_history['regularization']],
                'iteration': [int(x) for x in opt_history['iteration']],
                'grad_norm': [float(x) for x in opt_history['grad_norm']]
            }
        },
        'results': {
            'weights': [float(w) for w in w_opt],
            'baseline_metrics': {k: float(v) for k, v in metrics_before.items()},
            'optimized_metrics': {k: float(v) for k, v in metrics_after.items()},
            'improvement_percent': float(percent_reduction)
        }
    }
    
    # Add validation report if available
    if validation_report:
        optimization_report['validation'] = convert_to_json_serializable(validation_report)
    
    with open(report_path, 'w') as f:
        json.dump(optimization_report, f, indent=2)
    logger.info(f"Saved optimization report: {report_path}")
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info("Saved files:")
    logger.info(f"  - {baseline_path}")
    logger.info(f"  - {optimized_path}")
    logger.info(f"  - {before_after_path}")
    logger.info(f"  - {weights_path}")
    logger.info(f"  - {stats_path}")
    if USE_REPO_B0 and DATASET_DIR is not None:
        logger.info(f"  - {repo_comparison_path}")
    logger.info(f"\nImprovement: {percent_reduction:.2f}% reduction in ROI std")
    logger.info("="*60)


if __name__ == "__main__":
    main()

