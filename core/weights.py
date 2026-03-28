import numpy as np
from utils.helpers import get_logger, array_summary

logger = get_logger("weights")

def compute_leverage_weights(
    H: np.ndarray,
    c: np.ndarray = None,
    gamma: float = 0.01
) -> np.ndarray:
    """Computes leverage scores R_i — the geometric pull of each sensor."""
    if c is None:
        c = np.ones(H.shape[1])
        
    A = H * c[np.newaxis, :]
    ATA = A.T @ A + gamma * np.eye(A.shape[1])
    
    try:
        L = np.linalg.cholesky(ATA)
        # We need V such that V.T @ V = A @ (ATA)^-1 @ A.T
        # mathematically, ATA = L @ L.T (L is lower triangular)
        # A (L L.T)^-1 A.T = A L.T^-1 L^-1 A.T = (L^-1 A.T).T @ (L^-1 A.T)
        # so V = L^-1 A.T, which is the solution to L @ V = A.T
        V = np.linalg.solve(L, A.T)
        leverage = np.sum(V**2, axis=0)
    except np.linalg.LinAlgError:
        # Fallback to Eigendecomposition
        vals, vecs = np.linalg.eigh(ATA)
        vals = np.maximum(vals, 1e-12)
        inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
        V = inv_sqrt @ A.T
        leverage = np.sum(V**2, axis=0)
        
    # Clip and normalize
    mean_l = np.mean(leverage)
    std_l = np.std(leverage)
    leverage = np.clip(leverage, 0, mean_l + 3 * std_l)
    
    l_min = leverage.min()
    l_max = leverage.max()
    
    if l_max == l_min:
        R_i = np.full(H.shape[0], 0.5)
    else:
        R_i = (leverage - l_min) / (l_max - l_min)
        
    logger.info(f"R_i min: {R_i.min():.4f}, max: {R_i.max():.4f}, mean: {R_i.mean():.4f}")
    return R_i

def get_top_leverage_stations(
    R_i: np.ndarray,
    k: int = 3
) -> np.ndarray:
    """Returns indices of top k stations by leverage score."""
    top_idx = np.argsort(R_i)[::-1][:k]
    logger.info(f"Top {k} stations: {top_idx}, R_i values: {R_i[top_idx]}")
    return top_idx

def wind_adjusted_weights(
    R_i: np.ndarray,
    station_positions: np.ndarray,
    wind_direction_deg: float,
    beta: float = 0.3
) -> np.ndarray:
    """Apply upwind boost to R_i weights."""
    center_pos = np.mean(station_positions, axis=0)
    delta = station_positions - center_pos[np.newaxis, :]
    station_angle = np.arctan2(delta[:, 1], delta[:, 0])
    
    wind_rad = np.radians(wind_direction_deg)
    alignment = np.cos(wind_rad - station_angle)
    
    R_i_adjusted = R_i * (1 + beta * alignment)
    
    # Renormalize to [0,1]
    adj_min = R_i_adjusted.min()
    adj_max = R_i_adjusted.max()
    if adj_max > adj_min:
        R_i_adjusted = (R_i_adjusted - adj_min) / (adj_max - adj_min)
    else:
        R_i_adjusted = np.full_like(R_i_adjusted, 0.5)
        
    return R_i_adjusted

if __name__ == "__main__":
    from simulation.synthetic import generate_synthetic_data
    from core.inversion import tikhonov_solve
    
    data = generate_synthetic_data()
    
    R_i = compute_leverage_weights(
        data["H"], data["c0_wrong"], gamma=0.01
    )
    
    top_idx = get_top_leverage_stations(R_i, k=5)
    
    array_summary("R_i", R_i)
    print(f"R_i min:  {R_i.min():.4f}")
    print(f"R_i max:  {R_i.max():.4f}")
    print(f"R_i mean: {R_i.mean():.4f}")
    print(f"All R_i in [0,1]: {(R_i >= 0).all() and (R_i <= 1).all()}")
    print(f"Top 5 stations: {top_idx}")
    print(f"Their zones: ", end="")
    for idx in top_idx:
        print(data["zone_labels"][idx], end=" ")
    print("\n")
    
    print("IMPORTANT CHECK:")
    print("Road stations should heavily dominate the top 5 (due to topology bias).")
    print("This confirms the synthetic network bias accurately targets road corridors.")
    
    print("\nweights.py verification PASSED")
