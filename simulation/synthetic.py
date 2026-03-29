import numpy as np
from utils.helpers import get_logger, array_summary

logger = get_logger("synthetic")

def generate_synthetic_data(
    n_stations: int = 30,
    n_sources: int = 100,
    noise_std: float = 0.05,
    random_seed: int = 42
) -> dict:
    rng = np.random.default_rng(random_seed)
    
    # 1. STATION PLACEMENT (30 total)
    # Road (21)
    road_y = rng.uniform(0.4, 0.6, 21)
    road_x = np.linspace(0.05, 0.95, 21)
    road_pos = np.column_stack((road_x, road_y))
    road_labels = ["road"] * 21
    
    # Residential (6)
    res_x = rng.uniform(0.05, 0.28, 6)
    res_y = rng.uniform(0.05, 0.28, 6)
    res_pos = np.column_stack((res_x, res_y))
    res_labels = ["residential"] * 6
    
    # Industrial (3)
    ind_x = rng.uniform(0.72, 0.95, 3)
    ind_y = rng.uniform(0.72, 0.95, 3)
    ind_pos = np.column_stack((ind_x, ind_y))
    ind_labels = ["industrial"] * 3
    
    station_positions = np.vstack((road_pos, res_pos, ind_pos))
    zone_labels = road_labels + res_labels + ind_labels
    
    # 2. SOURCE GRID (10x10)
    xs = np.linspace(0.05, 0.95, 10)
    ys = np.linspace(0.05, 0.95, 10)
    X, Y = np.meshgrid(xs, ys)
    source_positions = np.column_stack((X.ravel(), Y.ravel()))
    
    sx = source_positions[:, 0]
    sy = source_positions[:, 1]
    
    # Define Source Indices matching instructions
    industrial_idx = (sx > 0.7) & (sy > 0.7)
    road_idx = (sy > 0.4) & (sy < 0.6)
    residential_idx = (sx < 0.3) & (sy < 0.3)
    
    # 3. GROUND TRUTH EMISSIONS
    x_true = np.zeros(100)
    # Residential background is 'remaining sources'
    remaining_idx = ~(industrial_idx | road_idx)
    x_true[remaining_idx] = rng.lognormal(mean=0.2, sigma=0.3, size=np.sum(remaining_idx))
    x_true[road_idx] = rng.lognormal(mean=0.8, sigma=0.5, size=np.sum(road_idx))
    x_true[industrial_idx] = rng.lognormal(mean=1.5, sigma=0.8, size=np.sum(industrial_idx))
    
    # 4. TRANSPORT MATRIX H
    H = np.zeros((30, 100))
    sigma_sq = 0.25**2
    for i in range(30):
        for j in range(100):
            dist_sq = np.sum((station_positions[i] - source_positions[j])**2)
            H[i, j] = np.exp(-dist_sq / (2 * sigma_sq))
        # Normalize each row to sum to 1
        H[i, :] /= np.sum(H[i, :])
        
    # 5. CONVERSION FACTORS
    c0_true = np.ones(100)
    c0_wrong = np.ones(100)
    c0_wrong[industrial_idx] = 0.50
    c0_wrong[road_idx] = 1.60
    c0_wrong[residential_idx] = 0.90
    
    # 6. PRIOR x0
    x0 = np.full(100, np.mean(x_true))
    
    # 7. OBSERVATIONS y
    A_wrong = H * c0_wrong[np.newaxis, :]
    y = A_wrong @ x_true + rng.normal(0, noise_std, 30)
    
    return {
        "H": H,
        "x_true": x_true,
        "x0": x0,
        "y": y,
        "station_positions": station_positions,
        "source_positions": source_positions,
        "zone_labels": zone_labels,
        "c0_true": c0_true,
        "c0_wrong": c0_wrong,
        "industrial_idx": industrial_idx,
        "road_idx": road_idx,
        "residential_idx": residential_idx
    }

if __name__ == "__main__":
    logger.info("Running synthetic.py verification checks...")
    data = generate_synthetic_data()
    
    # 1. Station counts
    zone_labels = data["zone_labels"]
    n_road = sum(1 for label in zone_labels if label == "road")
    n_res = sum(1 for label in zone_labels if label == "residential")
    n_ind = sum(1 for label in zone_labels if label == "industrial")
    
    print("\n--- 1. Station counts per zone ---")
    print(f"road: {n_road}, residential: {n_res}, industrial: {n_ind}")
    
    # 2. Array summaries
    print("\n--- 2. Array summaries ---")
    array_summary("H     ", data["H"])
    array_summary("x_true", data["x_true"])
    array_summary("x0    ", data["x0"])
    array_summary("y     ", data["y"])
    
    # 3. H condition number
    cond_H = np.linalg.cond(data["H"])
    print(f"\n--- 3. H Condition Number ---\nCondition(H): {cond_H:.2f}")
    
    # 4. Conversion factor check
    c0_wrong = data["c0_wrong"]
    mean_ind = np.mean(c0_wrong[data["industrial_idx"]])
    mean_road = np.mean(c0_wrong[data["road_idx"]])
    mean_res = np.mean(c0_wrong[data["residential_idx"]])
    
    print("\n--- 4. Conversion Factor Check ---")
    print(f"mean c0_wrong at industrial sources:  {mean_ind:.2f}")
    print(f"mean c0_wrong at road sources:        {mean_road:.2f}")
    print(f"mean c0_wrong at residential sources: {mean_res:.2f}")
    
    # 5. Final check
    if n_road == 21 and n_res == 6 and n_ind == 3:
        print("\nsynthetic.py verification PASSED")
    else:
        print("\nsynthetic.py verification FAILED: counts mismatch")
