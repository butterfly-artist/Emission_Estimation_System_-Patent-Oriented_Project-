import numpy as np
import csv
import os
from utils.helpers import get_logger
from core.adaptive import adaptive_loop
from core.weights import compute_leverage_weights, get_top_leverage_stations
from core.inversion import tikhonov_solve, compute_correlation

logger = get_logger("dropout")

def get_optimal_theta(
    H: np.ndarray,
    y: np.ndarray,
    x0: np.ndarray,
    c0: np.ndarray,
    lam: float = 0.01,
    gamma: float = 0.005,
    eta: float = 10.0,
    alpha: float = 0.3,
    max_iter: int = 50
) -> np.ndarray:
    
    result = adaptive_loop(
        H, y, x0, c0=c0,
        lam=lam, gamma=gamma,
        eta=eta, alpha=alpha,
        max_iter=max_iter
    )
    return result['theta_final']

def targeted_dropout(
    H: np.ndarray,
    y: np.ndarray,
    x0: np.ndarray,
    x_true: np.ndarray,
    c0: np.ndarray,
    zone_labels: list = None,
    k_drop: int = 5,
    lam: float = 0.01,
    gamma: float = 0.01,
    eta: float = 10.0,
    alpha: float = 0.3,
    max_iter: int = 50
) -> dict:
    """Remove the top k highest-leverage stations using warm start."""
    logger.info("Learning theta on full network for targeted dropout...")
    theta_star = get_optimal_theta(
        H, y, x0, c0, lam=lam, gamma=gamma, eta=eta, alpha=alpha, max_iter=max_iter
    )
    c_star = c0 * np.exp(theta_star)

    R_i_full = compute_leverage_weights(H, c0, gamma)
    drop_idx = get_top_leverage_stations(R_i_full, k=k_drop)
    
    keep_mask = np.ones(H.shape[0], dtype=bool)
    keep_mask[drop_idx] = False
    
    H_reduced = H[keep_mask]
    y_reduced = y[keep_mask]
    
    x_static = tikhonov_solve(H_reduced, y_reduced, x0, c=c0, lam=lam, quiet=True)
    x_adapt = tikhonov_solve(H_reduced, y_reduced, x0, c=c_star, lam=lam, quiet=True)
    
    var_static = float(np.mean((x_static - x_true)**2))
    var_adapt = float(np.mean((x_adapt - x_true)**2))
    S = var_static / var_adapt if var_adapt > 1e-12 else 1.0
    
    corr_s = compute_correlation(x_static, x_true)
    corr_a = compute_correlation(x_adapt, x_true)
    
    logger.info(f"Targeted dropout k={k_drop}: S={S:.4f} | corr_s={corr_s:.4f} | corr_a={corr_a:.4f}")
    
    if zone_labels is not None:
        drop_zone_labels = [zone_labels[i] for i in drop_idx]
    else:
        drop_zone_labels = ["Unknown"] * len(drop_idx)
        
    return {
        'dropped_stations': drop_idx.tolist(),
        'drop_zone_labels': drop_zone_labels,
        'var_static': var_static,
        'var_adapt': var_adapt,
        'S': S,
        'corr_static': corr_s,
        'corr_adapt': corr_a,
        'n_iter': 0
    }

def random_dropout_comparison(
    H, y, x0, x_true, c0,
    dropout_frac=0.40,
    n_trials=50,
    lam=0.01, gamma=0.005,
    eta=10.0, alpha=0.3,
    max_iter=50,
    random_seed=0
):
    rng = np.random.default_rng(random_seed)
    n_stations = H.shape[0]
    k_drop = max(1, int(dropout_frac * n_stations))
    
    logger.info("Learning theta on full network for random dropout...")
    theta_star = get_optimal_theta(
        H, y, x0, c0,
        lam=lam, gamma=gamma,
        eta=eta, alpha=alpha,
        max_iter=max_iter
    )
    c_star = c0 * np.exp(theta_star)
    logger.info(
        f"theta* learned | "
        f"c_star mean={c_star.mean():.4f}"
    )
    
    S_list = []
    var_s_list = []
    var_a_list = []
    corr_s_list = []
    corr_a_list = []
    
    for trial in range(n_trials):
        drop_idx = rng.choice(
            n_stations, k_drop, replace=False
        )
        mask = np.ones(n_stations, dtype=bool)
        mask[drop_idx] = False
        
        H_r = H[mask]
        y_r = y[mask]
        
        x_static = tikhonov_solve(
            H_r, y_r, x0,
            c=c0, lam=lam, quiet=True
        )
        x_adapt = tikhonov_solve(
            H_r, y_r, x0,
            c=c_star, lam=lam, quiet=True
        )
        
        vs = float(np.mean(
            (x_static - x_true)**2
        ))
        va = float(np.mean(
            (x_adapt - x_true)**2
        ))
        S = vs / va if va > 1e-12 else 1.0
        
        S_list.append(S)
        var_s_list.append(vs)
        var_a_list.append(va)
        corr_s_list.append(
            compute_correlation(x_static, x_true)
        )
        corr_a_list.append(
            compute_correlation(x_adapt, x_true)
        )
        
        if (trial + 1) % 10 == 0:
            logger.info(
                f"Trial {trial+1}/{n_trials} "
                f"S={S:.4f} "
                f"mean={np.mean(S_list):.4f}"
            )
    
    S_arr = np.array(S_list)
    return {
        'S_mean': float(np.mean(S_arr)),
        'S_min': float(np.min(S_arr)),
        'S_max': float(np.max(S_arr)),
        'S_std': float(np.std(S_arr)),
        'S_all': S_list,
        'var_s_mean': float(np.mean(var_s_list)),
        'var_a_mean': float(np.mean(var_a_list)),
        'corr_s_mean': float(np.mean(corr_s_list)),
        'corr_a_mean': float(np.mean(corr_a_list)),
        'n_trials': n_trials,
        'dropout_frac': dropout_frac,
        'pass_rate': float(
            np.mean(S_arr > 1.0)
        ),
        'pass_rate_1_2': float(
            np.mean(S_arr > 1.2)
        ),
        'theta_star': theta_star
    }

def high_leverage_targeted_dropout(
    H: np.ndarray,
    y: np.ndarray,
    x0: np.ndarray,
    x_true: np.ndarray,
    c0: np.ndarray,
    zone_labels: list = None,
    k_drop: int = 5,
    lam: float = 0.01,
    gamma: float = 0.01,
    eta: float = 10.0,
    alpha: float = 0.3,
    max_iter: int = 50
) -> dict:
    """Remove the top k stations by R_i leverage score using warm start."""
    logger.info("Learning theta on full network for high leverage...")
    theta_star = get_optimal_theta(
        H, y, x0, c0, lam=lam, gamma=gamma, eta=eta, alpha=alpha, max_iter=max_iter
    )
    c_star = c0 * np.exp(theta_star)

    R_i_full = compute_leverage_weights(H, c0, gamma)
    drop_idx = get_top_leverage_stations(R_i_full, k=k_drop)

    keep_mask = np.ones(H.shape[0], dtype=bool)
    keep_mask[drop_idx] = False

    H_reduced = H[keep_mask]
    y_reduced = y[keep_mask]

    x_static = tikhonov_solve(H_reduced, y_reduced, x0, c=c0, lam=lam, quiet=True)
    x_adapt  = tikhonov_solve(H_reduced, y_reduced, x0, c=c_star, lam=lam, quiet=True)

    var_static = float(np.mean((x_static - x_true)**2))
    var_adapt  = float(np.mean((x_adapt  - x_true)**2))
    S = var_static / var_adapt if var_adapt > 1e-12 else 1.0

    corr_s = compute_correlation(x_static, x_true)
    corr_a = compute_correlation(x_adapt,  x_true)

    if zone_labels is not None:
        drop_zone_labels = [zone_labels[i] for i in drop_idx]
    else:
        drop_zone_labels = ['Unknown'] * len(drop_idx)

    logger.info(
        f"High-leverage dropout k={k_drop}: "
        f"S={S:.4f} | dropped zones: {drop_zone_labels}"
    )

    return {
        'dropped_stations':  drop_idx.tolist(),
        'drop_zone_labels':  drop_zone_labels,
        'var_static':  var_static,
        'var_adapt':   var_adapt,
        'S':           S,
        'corr_static': corr_s,
        'corr_adapt':  corr_a,
        'n_iter':      0
    }


def cluster_dropout(
    H: np.ndarray,
    y: np.ndarray,
    x0: np.ndarray,
    x_true: np.ndarray,
    c0: np.ndarray,
    station_positions: np.ndarray,
    zone_labels: list,
    drop_zone: str = "road",
    lam: float = 0.01,
    gamma: float = 0.01,
    eta: float = 10.0,
    alpha: float = 0.3,
    max_iter: int = 50
) -> dict:
    """Remove ALL stations of one zone type using warm start."""
    logger.info("Learning theta on full network for cluster dropout...")
    theta_star = get_optimal_theta(
        H, y, x0, c0, lam=lam, gamma=gamma, eta=eta, alpha=alpha, max_iter=max_iter
    )
    c_star = c0 * np.exp(theta_star)

    drop_mask = np.array([label == drop_zone for label in zone_labels])
    keep_mask = ~drop_mask
    n_dropped = int(drop_mask.sum())
    
    H_r = H[keep_mask]
    y_r = y[keep_mask]
    
    x_static = tikhonov_solve(H_r, y_r, x0, c=c0, lam=lam, quiet=True)
    x_adapt  = tikhonov_solve(H_r, y_r, x0, c=c_star, lam=lam, quiet=True)
    
    vs = float(np.mean((x_static - x_true)**2))
    va = float(np.mean((x_adapt - x_true)**2))
    S = vs / va if va > 1e-12 else 1.0
    
    logger.info(f"Cluster dropout zone={drop_zone} n_dropped={n_dropped}: S={S:.4f}")
    
    return {
        'drop_zone': drop_zone,
        'n_dropped': n_dropped,
        'var_static': vs,
        'var_adapt': va,
        'S': S,
        'corr_static': compute_correlation(x_static, x_true),
        'corr_adapt': compute_correlation(x_adapt, x_true)
    }

def save_evidence_table(
    results: dict,
    filepath: str = "results/dropout_evidence.csv"
) -> None:
    """Save the results to a structured CSV table."""
    rows = []
    
    if 'targeted' in results:
        t = results['targeted']
        rows.append({
            'scenario': 'Targeted top-3 dropout',
            'var_static': t['var_static'],
            'var_adaptive': t['var_adapt'],
            'S': t['S'],
            'pass_1_0': t['S'] > 1.0,
            'pass_1_2': t['S'] > 1.2
        })
        
    if 'random' in results:
        r = results['random']
        rows.append({
            'scenario': 'Random 30% dropout',
            'var_static': r['var_s_mean'],
            'var_adaptive': r['var_a_mean'],
            'S': r['S_mean'],
            'pass_1_0': r['S_mean'] > 1.0,
            'pass_1_2': r['S_mean'] > 1.2
        })
        
    if 'cluster' in results:
        c = results['cluster']
        rows.append({
            'scenario': 'Road cluster dropout',
            'var_static': c['var_static'],
            'var_adaptive': c['var_adapt'],
            'S': c['S'],
            'pass_1_0': c['S'] > 1.0,
            'pass_1_2': c['S'] > 1.2
        })
        
    if 'high_leverage' in results:
        h = results['high_leverage']
        rows.append({
            'scenario': 'High-leverage top-5 dropout',
            'var_static': h['var_static'],
            'var_adaptive': h['var_adapt'],
            'S': h['S'],
            'pass_1_0': h['S'] > 1.0,
            'pass_1_2': h['S'] > 1.2
        })

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'scenario', 'var_static', 'var_adaptive', 'S', 'pass_1_0', 'pass_1_2'
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            
    logger.info(f"Saved to {filepath}")

if __name__ == "__main__":
    from simulation.synthetic import generate_synthetic_data

    data = generate_synthetic_data()

    print("="*55)
    print("TEST 1: Targeted top-5 dropout")
    print("="*55)
    t_result = targeted_dropout(
        data['H'], data['y'], data['x0'],
        data['x_true'], data['c0_wrong'],
        zone_labels=data['zone_labels'],
        k_drop=5, max_iter=50
    )
    print(f"Dropped zones: {t_result['drop_zone_labels']}")
    print(f"S targeted:    {t_result['S']:.4f}")
    print(f"Corr static:   {t_result['corr_static']:.4f}")
    print(f"Corr adaptive: {t_result['corr_adapt']:.4f}")

    print("\n" + "="*55)
    print("TEST 2: Random 40% dropout (10 trials)")
    print("="*55)
    r_result = random_dropout_comparison(
        data['H'], data['y'], data['x0'],
        data['x_true'], data['c0_wrong'],
        dropout_frac=0.40,
        n_trials=10,
        max_iter=50
    )
    print(f"S mean:        {r_result['S_mean']:.4f}")
    print(f"S min:         {r_result['S_min']:.4f}")
    print(f"S max:         {r_result['S_max']:.4f}")
    print(f"Pass rate S>1: {r_result['pass_rate']*100:.0f}%")

    print("\n" + "="*55)
    print("TEST 3: Road cluster dropout (21 of 30 stations)")
    print("="*55)
    c_result = cluster_dropout(
        data['H'], data['y'], data['x0'],
        data['x_true'], data['c0_wrong'],
        data['station_positions'],
        data['zone_labels'],
        drop_zone='road',
        max_iter=50
    )
    print(f"Stations dropped: {c_result['n_dropped']}")
    print(f"S cluster:     {c_result['S']:.4f}")

    print("\n" + "="*55)
    print("TEST 4: High-leverage top-5 dropout")
    print("="*55)
    h_result = high_leverage_targeted_dropout(
        data['H'], data['y'], data['x0'],
        data['x_true'], data['c0_wrong'],
        zone_labels=data['zone_labels'],
        k_drop=5, max_iter=50
    )
    print(f"Dropped zones: {h_result['drop_zone_labels']}")
    print(f"S high-lev:    {h_result['S']:.4f}")
    print(f"Corr static:   {h_result['corr_static']:.4f}")
    print(f"Corr adaptive: {h_result['corr_adapt']:.4f}")

    all_results = {
        'targeted':     t_result,
        'random':       r_result,
        'cluster':      c_result,
        'high_leverage': h_result
    }
    save_evidence_table(all_results)

    print("\n" + "="*55)
    print("SUMMARY")
    print("="*55)
    print(f"S targeted (k=5):      {t_result['S']:.4f}")
    print(f"S random 40% (mean):   {r_result['S_mean']:.4f}")
    print(f"S road cluster:        {c_result['S']:.4f}")
    print(f"S high-leverage (k=5): {h_result['S']:.4f}")
    all_S = [t_result['S'], r_result['S_mean'], c_result['S'], h_result['S']]
    if all(s > 1.0 for s in all_S):
        print("\ndropout.py verification PASSED — all S > 1.0")
    else:
        print("\ndropout.py verification PARTIAL — some S <= 1.0")

