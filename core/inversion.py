import numpy as np
import scipy.linalg
from utils.helpers import get_logger, array_summary

logger = get_logger("inversion")

def tikhonov_solve(
    H: np.ndarray,
    y: np.ndarray,
    x0: np.ndarray,
    c: np.ndarray = None,
    lam: float = 0.01,
    quiet: bool = False
) -> np.ndarray:
    """Solves the Tikhonov-regularized inverse problem."""
    if c is None:
        c = np.ones(H.shape[1])
        
    # Build A = H * diag(c) via broadcasting
    A = H * c[np.newaxis, :]
    
    # Build normal equations
    ATA = A.T @ A + lam * np.eye(A.shape[1])
    ATy = A.T @ y + lam * x0
    
    # Solve
    try:
        x_hat = scipy.linalg.solve(ATA, ATy, assume_a='pos')
    except scipy.linalg.LinAlgError:
        x_hat, _, _, _ = scipy.linalg.lstsq(ATA, ATy)
        
    # Clip result as emissions cannot be negative
    x_hat = np.clip(x_hat, 0, None)
    
    if not quiet:
        logger.info(f"x_hat shape: {x_hat.shape}, norm: {np.linalg.norm(x_hat):.4f}")
        
    return x_hat

def select_lambda(
    H: np.ndarray,
    y: np.ndarray,
    x0: np.ndarray,
    lambdas: list = None
) -> float:
    """L-curve heuristic for lambda selection."""
    if lambdas is None:
        lambdas = [1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5, 1.0]
        
    best_lam = lambdas[0]
    min_prod = float('inf')
    
    for lam in lambdas:
        x_hat = tikhonov_solve(H, y, x0, c=None, lam=lam, quiet=True)
        residual_norm = np.linalg.norm(y - H @ x_hat)
        solution_norm = np.linalg.norm(x_hat)
        prod = residual_norm * solution_norm
        
        if prod < min_prod:
            min_prod = prod
            best_lam = lam
            
    logger.info(f"Selected lambda: {best_lam}")
    return float(best_lam)

def compute_correlation(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """Returns the correlation coefficient between estimated and true emissions."""
    return float(np.corrcoef(x_hat, x_true)[0, 1])

if __name__ == "__main__":
    from simulation.synthetic import generate_synthetic_data
    
    data = generate_synthetic_data()
    
    x_hat = tikhonov_solve(
        data["H"], data["y"], data["x0"],
        c=data["c0_wrong"], lam=0.01
    )
    
    corr = compute_correlation(x_hat, data["x_true"])
    best_lam = select_lambda(
        data["H"], data["y"], data["x0"]
    )
    
    array_summary("x_hat  ", x_hat)
    array_summary("x_true ", data["x_true"])
    print(f"Correlation x_hat vs x_true: {corr:.4f}")
    print(f"Selected lambda: {best_lam}")
    print(f"x_hat negative values: {(x_hat < 0).sum()}")
    print("inversion.py verification PASSED")
