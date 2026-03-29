import numpy as np
import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.helpers import get_logger, Timer
from data.loaders import load_hyderabad_data
from core.adaptive import adaptive_loop
from core.weights import (
    compute_leverage_weights,
    get_top_leverage_stations
)
from core.inversion import (
    tikhonov_solve,
    compute_correlation
)
from core.residuals import (
    compute_residuals,
    weighted_residual_loss,
    residual_summary
)

logger = get_logger("run_realdata")

CANONICAL_PARAMS = {
    'eta':   0.2,
    'alpha': 0.3,
    'gamma': 0.005,
    'lam':   0.01
}

def real_data_dropout_test(
    H: np.ndarray,
    y: np.ndarray,
    x0: np.ndarray,
    params: dict,
    n_trials: int = 30,
    dropout_frac: float = 0.25
) -> dict:

    # STEP 1: Full data reference estimate
    x_ref_static = tikhonov_solve(
        H, y, x0,
        c=np.ones(H.shape[1]),
        lam=params['lam'], quiet=True
    )
    result_ref = adaptive_loop(
        H, y, x0,
        c0=np.ones(H.shape[1]),
        lam=params['lam'],
        gamma=params['gamma'],
        eta=params['eta'],
        alpha=params['alpha'],
        max_iter=25
    )
    x_ref_adapt = result_ref['x_adapt']

    # STEP 2: Monte Carlo dropout
    rng = np.random.default_rng(42)
    n = H.shape[0]
    k = max(1, int(dropout_frac * n))
    
    S_list = []
    var_s_list = []
    var_a_list = []
    
    for trial in range(n_trials):
        drop_idx = rng.choice(n, k, replace=False)
        mask = np.ones(n, dtype=bool)
        mask[drop_idx] = False
        
        H_r = H[mask]
        y_r = y[mask]
        
        x_s = tikhonov_solve(
            H_r, y_r, x0,
            c=np.ones(H.shape[1]),
            lam=params['lam'], quiet=True
        )
        res = adaptive_loop(
            H_r, y_r, x0,
            c0=np.ones(H.shape[1]),
            lam=params['lam'],
            gamma=params['gamma'],
            eta=params['eta'],
            alpha=params['alpha'],
            max_iter=20
        )
        x_a = res['x_adapt']
        
        vs = float(np.mean((x_s - x_ref_static)**2))
        va = float(np.mean((x_a - x_ref_adapt)**2))
        S = vs / va if va > 1e-12 else 1.0
        
        S_list.append(S)
        var_s_list.append(vs)
        var_a_list.append(va)
        
        if (trial+1) % 10 == 0:
            logger.info(
                f"Trial {trial+1}/{n_trials} "
                f"S={S:.4f} "
                f"mean={np.mean(S_list):.4f}"
            )
            
    S_arr = np.array(S_list)
    return {
        'S_mean':     float(np.mean(S_arr)),
        'S_min':      float(np.min(S_arr)),
        'S_max':      float(np.max(S_arr)),
        'S_std':      float(np.std(S_arr)),
        'S_all':      S_list,
        'var_s_mean': float(np.mean(var_s_list)),
        'var_a_mean': float(np.mean(var_a_list)),
        'pass_rate':  float(np.mean(S_arr > 1.0)),
        'x_ref_static': x_ref_static,
        'x_ref_adapt':  x_ref_adapt,
        'adapt_result': result_ref
    }


def run_realdata_pipeline(
    api_key: str,
    resource_id: str,
    params: dict = None,
    n_trials: int = 30,
    save_dir: str = "results"
) -> dict:

    if params is None:
        params = CANONICAL_PARAMS
        
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/figures", exist_ok=True)
    
    # STEP 1
    logger.info("Step 1: Loading real CPCB data")
    with Timer("CPCB data load"):
        data = load_hyderabad_data(
            api_key=api_key,
            resource_id=resource_id,
            pollutant="NO2"
        )
        
    if not data:
        logger.error("Failed to load real data")
        return {}
        
    logger.info(f"Loaded {data['n_stations']} real stations")
    H   = data['H']
    y   = data['y']
    x0  = data['x0']
    c0  = data['c0_initial']
    
    # STEP 2
    logger.info("Step 2: Static inversion")
    x_static = tikhonov_solve(
        H, y, x0, c=c0,
        lam=params['lam'], quiet=True
    )
    r_static = compute_residuals(H, x_static, y, c=c0)
    
    # STEP 3
    logger.info("Step 3: Adaptive loop")
    with Timer("adaptive loop real data"):
        adapt_result = adaptive_loop(
            H, y, x0, c0=c0,
            lam=params['lam'],
            gamma=params['gamma'],
            eta=params['eta'],
            alpha=params['alpha'],
            max_iter=25
        )
    x_adapt = adapt_result['x_adapt']
    r_adapt = compute_residuals(
        H, x_adapt, y,
        c=adapt_result['c_final']
    )
    
    logger.info(
        f"Converged: {adapt_result['converged']} "
        f"in {adapt_result['n_iter']} iters"
    )
    
    # STEP 4
    R_i = compute_leverage_weights(
        H, adapt_result['c_final'],
        gamma=params['gamma']
    )
    top_idx = get_top_leverage_stations(R_i, k=3)
    top_stations = [data['station_names'][i] for i in top_idx]
    logger.info(f"Top 3 leverage stations: {top_stations}")
    
    # STEP 5
    logger.info(f"Step 5: Real data dropout test ({n_trials} trials)")
    with Timer("real dropout test"):
        dropout = real_data_dropout_test(
            H, y, x0, params,
            n_trials=n_trials,
            dropout_frac=0.25
        )
        
    # STEP 6
    res_comp = residual_summary(r_static['r'], r_adapt['r'], R_i)
    
    # STEP 7
    evidence = [{
        'scenario': 'Real CPCB Hyderabad NO2',
        'n_stations': data['n_stations'],
        'S_mean': dropout['S_mean'],
        'S_min':  dropout['S_min'],
        'S_max':  dropout['S_max'],
        'pass_rate': dropout['pass_rate'],
        'var_reduction': res_comp['var_reduction'],
        'loss_reduction': res_comp['loss_reduction'],
        'n_iter': adapt_result['n_iter'],
        'converged': adapt_result['converged'],
        'pass_1_0': dropout['S_mean'] > 1.0,
        'pass_1_2': dropout['S_mean'] > 1.2
    }]
    path = f"{save_dir}/real_evidence.csv"
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=evidence[0].keys())
        writer.writeheader()
        writer.writerows(evidence)
    logger.info(f"Real evidence saved: {path}")

    # STEP 8
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        'Real CPCB Data — Hyderabad NO2\n'
        'Adaptive Emission Inversion Results',
        fontsize=13
    )
    
    ax = axes[0, 0]
    sc = ax.scatter(data['source_positions'][:, 0], data['source_positions'][:, 1], c=x_adapt, cmap='viridis', label='Sources')
    ax.scatter(data['station_positions'][:, 0], data['station_positions'][:, 1], s=R_i * 200, c='white', edgecolors='black', label='Stations')
    ax.set_title("Adaptive Emission Estimate\n(Real CPCB Data)")
    plt.colorbar(sc, ax=ax, label="Emission intensity")
    ax.legend(loc="upper left")
    
    ax = axes[0, 1]
    ax.plot(adapt_result['losses'], marker='o', c='purple')
    ax.set_title("Adaptive Loop Convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("J(theta) weighted loss")
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    indices = np.arange(len(r_static['r']))
    width = 0.35
    ax.bar(indices - width/2, r_static['r'], width, label='Static', color='red', alpha=0.7)
    ax.bar(indices + width/2, r_adapt['r'], width, label='Adaptive', color='blue', alpha=0.7)
    ax.set_title("Residuals per Station")
    ax.set_xlabel("Station index")
    ax.set_ylabel("Residual value")
    ax.legend()
    
    ax = axes[1, 1]
    S_arr = np.array(dropout['S_all'])
    ax.hist(S_arr, bins=10, color='cadetblue', edgecolor='black')
    ax.axvline(1.0, color='red', linestyle='--', label='S=1.0')
    ax.axvline(1.2, color='green', linestyle='--', label='S=1.2')
    ax.set_title(f"Stability S Distribution\nReal Data: mean S={dropout['S_mean']:.4f}")
    ax.legend()
    
    plt.tight_layout()
    fig_path = f"{save_dir}/figures/realdata_results.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Figure saved: {fig_path}")

    # STEP 9
    print("\n" + "="*60)
    print("REAL DATA PIPELINE COMPLETE")
    print("="*60)
    print(f"{'City':<30} {'Hyderabad':>15}")
    print(f"{'Stations':<30} {data['n_stations']:>15}")
    print(f"{'Pollutant':<30} {'NO2':>15}")
    print(f"{'Data source':<30} {'CPCB data.gov.in':>15}")
    print("-" * 50)
    print(f"{'Static RMSE':<30} {r_static['rmse']:>15.6f}")
    print(f"{'Adaptive RMSE':<30} {r_adapt['rmse']:>15.6f}")
    print(f"{'Var reduction':<30} {res_comp['var_reduction']*100:>14.1f}%")
    print(f"{'S mean (real dropout)':<30} {dropout['S_mean']:>15.4f}")
    print(f"{'S min':<30} {dropout['S_min']:>15.4f}")
    print(f"{'Pass rate S>1.0':<30} {dropout['pass_rate']*100:>14.0f}%")
    print(f"{'Iterations':<30} {adapt_result['n_iter']:>15}")
    print("-" * 50)
    
    if dropout['S_mean'] > 1.2:
        print("REAL DATA RESULT: STRONG")
        print("S > 1.2 on real Hyderabad data")
        print("Patent claim holds on real data")
    elif dropout['S_mean'] > 1.0:
        print("REAL DATA RESULT: VALID")
        print("S > 1.0 on real Hyderabad data")
        print("Patent claim supported")
    else:
        print("REAL DATA RESULT: MARGINAL")
        print("Tune parameters for real data")
        
    print(f"\nEvidence: {save_dir}/real_evidence.csv")
    print(f"Figure:   {fig_path}")
    
    return {
        'n_stations':     data['n_stations'],
        'S_mean':         dropout['S_mean'],
        'S_min':          dropout['S_min'],
        'pass_rate':      dropout['pass_rate'],
        'var_reduction':  res_comp['var_reduction'],
        'converged':      adapt_result['converged'],
        'station_names':  data['station_names'],
        'R_i':            R_i,
        'top_stations':   [data['station_names'][i] for i in top_idx]
    }


if __name__ == "__main__":
    API_KEY     = "579b464db66ec23bdd0000019f264217b1eb456c76a616628cc180b9"
    RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
    
    print("REAL DATA PIPELINE")
    print("Hyderabad NO2 — CPCB data.gov.in")
    print("="*60)
    
    if API_KEY == "579b464db66ec23bdd0000019f264217b1eb456c76a616628cc180b9":
        print("Replace API_KEY with your key")
        print("in pipeline/run_realdata.py")
    else:
        result = run_realdata_pipeline(
            api_key=API_KEY,
            resource_id=RESOURCE_ID,
            params=CANONICAL_PARAMS,
            n_trials=30
        )
        
        if result:
            if result['S_mean'] > 1.0:
                print("\nREADY FOR DASHBOARD")
                print("Session 11: dashboard/app.py")
