import numpy as np
from utils.helpers import get_logger, array_summary

logger = get_logger("dispersion")

def build_gaussian_H(
    station_positions: np.ndarray,
    source_positions: np.ndarray,
    sigma: float = 0.25,
    normalize_rows: bool = True
) -> np.ndarray:
    """Builds the Gaussian plume transport matrix H."""
    # station_positions: (N, 2), source_positions: (M, 2)
    # delta shape: (N, M, 2)
    delta = station_positions[:, np.newaxis, :] - source_positions[np.newaxis, :, :]
    dist_sq = np.sum(delta**2, axis=2)
    
    H = np.exp(-dist_sq / (2 * sigma**2))
    
    if normalize_rows:
        row_sums = H.sum(axis=1, keepdims=True)
        H = H / np.maximum(row_sums, 1e-8)
        
    return H

def build_wind_adjusted_H(
    station_positions: np.ndarray,
    source_positions: np.ndarray,
    wind_speed: float,
    wind_direction_deg: float,
    sigma: float = 0.25
) -> np.ndarray:
    """Builds transport matrix adjusted for wind direction alignment."""
    # Convert direction to radians
    wind_rad = np.radians(wind_direction_deg)
    wind_vec = np.array([np.cos(wind_rad), np.sin(wind_rad)])
    
    # Vector from station to source is source - station
    delta = source_positions[np.newaxis, :, :] - station_positions[:, np.newaxis, :]
    dist = np.linalg.norm(delta, axis=2)
    
    norm_delta = delta / np.maximum(dist[:, :, np.newaxis], 1e-8)
    
    # dot product along the 3rd axis
    dot_prod = np.sum(norm_delta * wind_vec, axis=2)
    alignment = np.maximum(0, dot_prod)
    
    # Get base H without normalization
    H = build_gaussian_H(station_positions, source_positions, sigma, normalize_rows=False)
    
    # Apply wind weights and normalize
    H *= (1 + alignment)
    row_sums = H.sum(axis=1, keepdims=True)
    H /= np.maximum(row_sums, 1e-8)
    
    return H

def condition_number(H: np.ndarray) -> float:
    """Calculate and log the condition number of H."""
    cond_H = np.linalg.cond(H)
    logger.info(f"Condition number of H: {cond_H:.2f}")
    return float(cond_H)

if __name__ == "__main__":
    from simulation.synthetic import generate_synthetic_data
    
    data = generate_synthetic_data()
    
    H1 = build_gaussian_H(
        data["station_positions"],
        data["source_positions"]
    )
    
    H2 = build_wind_adjusted_H(
        data["station_positions"],
        data["source_positions"],
        wind_speed=5.0,
        wind_direction_deg=225.0
    )
    
    array_summary("H standard  ", H1)
    array_summary("H wind-adj  ", H2)
    print(f"Condition number H1: {condition_number(H1):.2f}")
    print(f"Row sums H1 (first 3): {H1[:3].sum(axis=1)}")
    print("dispersion.py verification PASSED")
