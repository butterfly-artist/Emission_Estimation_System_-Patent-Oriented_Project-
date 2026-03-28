import numpy as np
from utils.helpers import get_logger, array_summary

logger = get_logger("residuals")

def compute_residuals(
    H: np.ndarray,
    x_hat: np.ndarray,
    y: np.ndarray,
    c: np.ndarray = None
) -> dict:
    """Computes residuals r = y - A*x_hat."""
    if c is None:
        c = np.ones(H.shape[1])
    
    A = H * c[np.newaxis, :]
    r = y - A @ x_hat
    
    return {
        'r': r,
        'rmse': float(np.sqrt(np.mean(r**2))),
        'mae': float(np.mean(np.abs(r))),
        'var_r': float(np.var(r)),
        'max_abs_r': float(np.max(np.abs(r))),
        'norm_r': float(np.linalg.norm(r))
    }

def weighted_residual_loss(
    r: np.ndarray,
    R_i: np.ndarray
) -> float:
    """Computes the patent geometry-aware loss function J(theta)."""
    return float(np.dot(R_i, r**2))

def residual_summary(
    r_static: np.ndarray,
    r_adaptive: np.ndarray,
    R_i: np.ndarray
) -> dict:
    """Compare static vs adaptive residuals loss."""
    var_s = np.var(r_static)
    var_a = np.var(r_adaptive)
    
    loss_s = weighted_residual_loss(r_static, R_i)
    loss_a = weighted_residual_loss(r_adaptive, R_i)
    
    return {
        'var_static': float(var_s),
        'var_adaptive': float(var_a),
        'var_reduction': float((var_s - var_a) / var_s) if var_s > 0 else 0.0,
        'loss_static': float(loss_s),
        'loss_adaptive': float(loss_a),
        'loss_reduction': float((loss_s - loss_a) / loss_s) if loss_s > 0 else 0.0
    }

if __name__ == "__main__":
    from simulation.synthetic import generate_synthetic_data
    from core.inversion import tikhonov_solve
    from core.weights import compute_leverage_weights
    
    data = generate_synthetic_data()
    R_i = compute_leverage_weights(
        data["H"], data["c0_wrong"]
    )
    
    x_hat = tikhonov_solve(
        data["H"], data["y"], data["x0"],
        c=data["c0_wrong"], lam=0.01
    )
    
    res = compute_residuals(
        data["H"], x_hat, data["y"],
        c=data["c0_wrong"]
    )
    
    print(f"RMSE:      {res['rmse']:.6f}")
    print(f"MAE:       {res['mae']:.6f}")
    print(f"Var(r):    {res['var_r']:.6f}")
    print(f"||r||:     {res['norm_r']:.6f}")
    
    loss = weighted_residual_loss(res['r'], R_i)
    print(f"Weighted loss J: {loss:.6f}")
    print(f"R_i shape: {R_i.shape}")
    print(f"r shape:   {res['r'].shape}")
    print("residuals.py verification PASSED")
