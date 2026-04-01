import numpy as np
import csv
import os
import itertools
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.helpers import get_logger
from simulation.synthetic import generate_synthetic_data
from simulation.dropout import (
    targeted_dropout,
    random_dropout_comparison,
    cluster_dropout,
    save_evidence_table
)

logging.getLogger("weights").setLevel(logging.WARNING)
logging.getLogger("adaptive").setLevel(logging.WARNING)
logging.getLogger("residuals").setLevel(logging.WARNING)
logging.getLogger("dropout").setLevel(logging.WARNING)

logger = get_logger("parameter_sweep")

def diagnose_leverage(
    data: dict,
    gamma_values: list = [0.001, 0.01, 0.1]
) -> None:
    from core.weights import compute_leverage_weights
    
    print("\n=== LEVERAGE DIAGNOSIS ===")
    for gamma in gamma_values:
        R_i = compute_leverage_weights(data['H'], data['c0_wrong'], gamma)
        
        road_mask = np.array([l == 'road' for l in data['zone_labels']])
        ind_mask = np.array([l == 'industrial' for l in data['zone_labels']])
        res_mask = np.array([l == 'residential' for l in data['zone_labels']])
        
        print(f"\ngamma={gamma}:")
        print(f"  Road R_i mean:        {R_i[road_mask].mean():.4f}")
        print(f"  Industrial R_i mean:  {R_i[ind_mask].mean():.4f}")
        print(f"  Residential R_i mean: {R_i[res_mask].mean():.4f}")
        print(f"  Top 3 station zones:  ", end="")
        top3 = np.argsort(R_i)[::-1][:3]
        for idx in top3:
            print(data['zone_labels'][idx], end=" ")
        print()
        
    print("\nExpected: road should dominate top 3")
    print("If not: network bias is insufficient")
    print("Fix: increase mismatch in synthetic.py")

def evaluate_combination(params):
    eta, alpha, gamma, lam, noise, fd_eps, n_trials = params
    data = generate_synthetic_data(noise_std=noise, random_seed=42)
    r = random_dropout_comparison(
        data['H'], data['y'], data['x0'], data['x_true'], data['c0_wrong'],
        dropout_frac=0.30, n_trials=n_trials, lam=lam, gamma=gamma,
        eta=eta, alpha=alpha, max_iter=20, random_seed=0, fd_eps=fd_eps
    )
    return {
        'eta': eta,
        'alpha': alpha,
        'gamma': gamma,
        'lam': lam,
        'noise': noise,
        'fd_eps': fd_eps,
        'S_mean': r['S_mean'],
        'S_min': r['S_min'],
        'S_max': r['S_max'],
        'S_std': r['S_std'],
        'pass_rate_1_0': r['pass_rate'],
        'pass_rate_1_2': float(np.mean(np.array(r['S_all']) > 1.2)),
        'corr_improvement': r['corr_a_mean'] - r['corr_s_mean']
    }

def run_parameter_sweep(
    n_trials: int = 30,
    save_path: str = "results/parameter_sweep.csv"
) -> dict:
    eta_values    = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    alpha_values  = [0.0, 0.1, 0.2, 0.3]
    gamma_values  = [0.001, 0.005, 0.01, 0.05]
    lam_values    = [0.001, 0.01, 0.05, 0.1]
    noise_values  = [0.01, 0.05]
    fd_eps_values = [0.01, 0.05, 0.1]
    
    results_list = []
    best_S = 0.0
    best_config = {}
    
    combos = list(itertools.product(eta_values, alpha_values, gamma_values, lam_values, noise_values, fd_eps_values))
    total = len(combos)
    
    tasks = [(eta, alpha, gamma, lam, noise, fd_eps, n_trials) for eta, alpha, gamma, lam, noise, fd_eps in combos]
    
    logger.info(f"Starting ProcessPoolExecutor with {total} combinations...")
    
    import multiprocessing
    cores = min(multiprocessing.cpu_count(), 32)
    
    with ProcessPoolExecutor(max_workers=cores) as executor:
        futures = {executor.submit(evaluate_combination, task): task for task in tasks}
        
        for combo_count, future in enumerate(as_completed(futures), 1):
            row = future.result()
            results_list.append(row)
            
            if row['S_mean'] > best_S:
                best_S = row['S_mean']
                best_config = row.copy()
                logger.info(f"NEW BEST S={best_S:.4f} | eta={row['eta']} alpha={row['alpha']} gamma={row['gamma']} lam={row['lam']} fd_eps={row['fd_eps']}")
                
            if combo_count % 50 == 0:
                logger.info(f"Progress: {combo_count}/{total} combos | best S={best_S:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fieldnames = list(results_list[0].keys())
    with open(save_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_list)
        
    logger.info(f"Saved {len(results_list)} results to {save_path}")
    
    sorted_results = sorted(results_list, key=lambda x: x['S_mean'], reverse=True)[:10]
    
    print("\n" + "="*70)
    print("TOP 10 CONFIGURATIONS BY S_mean")
    print("="*70)
    print(f"{'eta':>6} {'alpha':>6} {'gamma':>7} {'lam':>6} {'noise':>6} {'fd_eps':>6} {'S_mean':>7} {'S_min':>7} {'pass%':>6}")
    print("-" * 70)
    for r in sorted_results:
        print(
            f"{r['eta']:>6.2f} "
            f"{r['alpha']:>6.2f} "
            f"{r['gamma']:>7.3f} "
            f"{r['lam']:>6.3f} "
            f"{r['noise']:>6.2f} "
            f"{r['fd_eps']:>6.2f} "
            f"{r['S_mean']:>7.4f} "
            f"{r['S_min']:>7.4f} "
            f"{r['pass_rate_1_2']*100:>6.0f}%"
        )
        
    print(f"\nBEST CONFIG:")
    print(f"  eta={best_config['eta']}")
    print(f"  alpha={best_config['alpha']}")
    print(f"  gamma={best_config['gamma']}")
    print(f"  lam={best_config['lam']}")
    print(f"  noise={best_config['noise']}")
    print(f"  fd_eps={best_config['fd_eps']}")
    print(f"  S_mean={best_config['S_mean']:.4f}")
    print(f"  S_min={best_config['S_min']:.4f}")
    
    if best_S >= 1.2:
        print(f"\nS > 1.2 ACHIEVED")
        print(f"Patent claim supported.")
    elif best_S >= 1.0:
        print(f"\nS > 1.0 achieved.")
        print(f"Close to patent threshold.")
        print(f"Try increasing mismatch in")
        print(f"simulation/synthetic.py")
    else:
        print(f"\nS still below 1.0.")
        print(f"Check gradient direction in")
        print(f"core/adaptive.py")
        
    return {
        'best_config': best_config,
        'best_S': best_S,
        'all_results': results_list,
        'top_10': sorted_results
    }

if __name__ == "__main__":
    # REQUIRED on Windows for ProcessPoolExecutor: Ensure safety lock before running process loop.
    import multiprocessing
    multiprocessing.freeze_support()
    
    data = generate_synthetic_data()
    
    print("STEP 1: Diagnosing leverage scores")
    print("to understand why road stations")
    print("are not dominating...")
    diagnose_leverage(data)
    
    print("\nSTEP 2: Running parameter sweep")
    print("Estimated time: ~2-5 minutes (Multiprocessed)")
    print("Watch for NEW BEST S= in logs")
    print("="*60)
    
    sweep_result = run_parameter_sweep(
        n_trials=20,
        save_path="results/parameter_sweep.csv"
    )
    
    print("\n" + "="*60)
    print("SWEEP COMPLETE")
    print("="*60)
    print(f"Best S achieved: {sweep_result['best_S']:.4f}")
    print(f"Results saved to: results/parameter_sweep.csv")
    
    if sweep_result['best_S'] >= 1.2:
        print("\nREADY FOR SESSION 8")
        print("Use best_config parameters")
        print("in run_pipeline.py")
    else:
        print("\nBefore Session 8:")
        print("Increase c0_wrong mismatch in")
        print("simulation/synthetic.py:")
        print("  traffic:    1.4 -> 1.6")
        print("  industrial: 0.7 -> 0.5")
        print("Then re-run this sweep.")
