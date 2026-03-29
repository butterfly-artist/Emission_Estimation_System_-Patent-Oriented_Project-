import numpy as np
import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.helpers import get_logger, Timer
from simulation.synthetic import generate_synthetic_data
from simulation.dropout import (
    targeted_dropout,
    random_dropout_comparison,
    cluster_dropout,
    save_evidence_table
)
from core.adaptive import adaptive_loop
from core.weights import compute_leverage_weights
from core.inversion import tikhonov_solve, compute_correlation
from core.residuals import compute_residuals, residual_summary

logger = get_logger("pipeline")

CANONICAL_PARAMS = {
    'eta':    0.2,
    'alpha':  0.3,
    'gamma':  0.005,
    'lam':    0.01,
    'noise':  0.05
}

def run_full_pipeline(
    params: dict = None,
    n_trials: int = 50,
    save_dir: str = "results"
) -> dict:

    if params is None:
        params = CANONICAL_PARAMS
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/figures", exist_ok=True)
    
    logger.info("="*50)
    logger.info("ADAPTIVE GHG INVERSION PIPELINE")
    logger.info("="*50)

    # STEP 1
    logger.info("Step 1: Generating synthetic data")
    with Timer("data generation"):
        data = generate_synthetic_data(
            noise_std=params['noise'],
            random_seed=42
        )
    logger.info(
        f"Stations: {data['H'].shape[0]} | "
        f"Sources: {data['H'].shape[1]}"
    )

    # STEP 2
    logger.info("Step 2: Static inversion baseline")
    with Timer("static inversion"):
        x_static = tikhonov_solve(
            data['H'], data['y'], data['x0'],
            c=data['c0_wrong'],
            lam=params['lam'], quiet=True
        )
    corr_static = compute_correlation(x_static, data['x_true'])
    logger.info(f"Static corr: {corr_static:.4f}")

    # STEP 3
    logger.info("Step 3: Running adaptive loop")
    with Timer("adaptive loop"):
        adapt_result = adaptive_loop(
            data['H'], data['y'], data['x0'],
            c0=data['c0_wrong'],
            lam=params['lam'],
            gamma=params['gamma'],
            eta=params['eta'],
            alpha=params['alpha'],
            max_iter=25
        )
    x_adapt = adapt_result['x_adapt']
    corr_adapt = compute_correlation(x_adapt, data['x_true'])
    logger.info(
        f"Adaptive corr: {corr_adapt:.4f} "
        f"(+{corr_adapt-corr_static:+.4f})"
    )
    logger.info(
        f"Converged: {adapt_result['converged']} "
        f"in {adapt_result['n_iter']} iterations"
    )

    # STEP 4
    logger.info("Step 4: Computing final R_i weights")
    R_i = compute_leverage_weights(
        data['H'], adapt_result['c_final'],
        gamma=params['gamma']
    )
    from core.weights import get_top_leverage_stations
    top_idx = get_top_leverage_stations(R_i, k=5)
    top_zones = [data['zone_labels'][i] for i in top_idx]
    logger.info(f"Top 5 zones: {top_zones}")

    # STEP 5
    logger.info("Step 5: Running dropout stress tests")
    
    logger.info("  5a: Targeted top-3 dropout")
    with Timer("targeted dropout"):
        t_result = targeted_dropout(
            data['H'], data['y'],
            data['x0'], data['x_true'],
            data['c0_wrong'],
            zone_labels=data['zone_labels'],
            k_drop=3,
            lam=params['lam'],
            gamma=params['gamma'],
            eta=params['eta'],
            alpha=params['alpha'],
            max_iter=20
        )
    
    logger.info("  5b: Random 30% dropout (50 trials)")
    with Timer("random dropout 50 trials"):
        r_result = random_dropout_comparison(
            data['H'], data['y'],
            data['x0'], data['x_true'],
            data['c0_wrong'],
            dropout_frac=0.30,
            n_trials=n_trials,
            lam=params['lam'],
            gamma=params['gamma'],
            eta=params['eta'],
            alpha=params['alpha'],
            max_iter=20,
            random_seed=42
        )
    
    logger.info("  5c: Road cluster dropout")
    with Timer("cluster dropout"):
        c_result = cluster_dropout(
            data['H'], data['y'],
            data['x0'], data['x_true'],
            data['c0_wrong'],
            data['station_positions'],
            data['zone_labels'],
            drop_zone='road',
            lam=params['lam'],
            gamma=params['gamma'],
            eta=params['eta'],
            alpha=params['alpha'],
            max_iter=20
        )

    # STEP 6
    logger.info("Step 6: Saving patent evidence table")
    all_dropout = {
        'targeted': t_result,
        'random':   r_result,
        'cluster':  c_result
    }
    evidence_path = f"{save_dir}/patent_evidence_table.csv"
    save_evidence_table(all_dropout, filepath=evidence_path)

    # STEP 7
    logger.info("Step 7: Generating results figure")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        'Adaptive GHG Emission Inversion — Patent Evidence\n'
        f'S_mean={r_result["S_mean"]:.4f} | '
        f'Canonical params: eta={params["eta"]} '
        f'alpha={params["alpha"]} '
        f'gamma={params["gamma"]}',
        fontsize=12
    )
    
    # Panel 1: Emission comparison
    ax = axes[0, 0]
    ax.scatter(data['x_true'], x_static, c='red', alpha=0.6, label=f'Static (r={corr_static:.2f})')
    ax.scatter(data['x_true'], x_adapt, c='blue', alpha=0.6, label=f'Adapt (r={corr_adapt:.2f})')
    # 1:1 line
    max_val = max(data['x_true'].max(), x_static.max(), x_adapt.max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    ax.set_title("Emission Estimates vs Ground Truth")
    ax.set_xlabel("True Emissions")
    ax.set_ylabel("Estimated Emissions")
    ax.legend()
    
    # Panel 2: Loss convergence
    ax = axes[0, 1]
    ax.plot(adapt_result['losses'], marker='o', color='purple')
    ax.set_title("Adaptive Loop Loss Convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("J(theta)")
    ax.grid(True, alpha=0.3)
    
    # Panel 3: R_i leverage scores
    ax = axes[0, 2]
    colors = []
    for lbl in data['zone_labels']:
        if lbl == 'road': colors.append('blue')
        elif lbl == 'industrial': colors.append('red')
        else: colors.append('green')
    ax.bar(range(len(R_i)), R_i, color=colors)
    ax.set_title("Station Leverage Scores R_i")
    ax.set_xlabel("Station Index")
    ax.set_ylabel("Leverage Score R_i")
    
    # Panel 4: Residuals comparison
    ax = axes[1, 0]
    res_s = compute_residuals(data['H'], x_static, data['y'], c=data['c0_wrong'])
    res_a = compute_residuals(data['H'], x_adapt, data['y'], c=adapt_result['c_final'])
    rmse_s = res_s['rmse']
    rmse_a = res_a['rmse']
    ax.plot(res_s['r'], label=f'Static (RMSE={rmse_s:.3f})', color='red', alpha=0.7)
    ax.plot(res_a['r'], label=f'Adaptive (RMSE={rmse_a:.3f})', color='blue', alpha=0.7)
    ax.set_title("Residuals: Static vs Adaptive")
    ax.set_xlabel("Station Index")
    ax.set_ylabel("Residual Error")
    ax.legend()
    
    # Panel 5: S distribution
    ax = axes[1, 1]
    S_arr = np.array(r_result['S_all'])
    ax.hist(S_arr, bins=15, color='cadetblue', edgecolor='black')
    ax.axvline(1.0, color='red', linestyle='--', label='S=1.0 (Baseline)')
    ax.axvline(1.2, color='green', linestyle='--', label='S=1.2 (Patent bounds)')
    ax.set_title(f"S Distribution (50 trials)\nmean={r_result['S_mean']:.4f}")
    ax.set_xlabel("Stability Metric S")
    ax.set_ylabel("Frequency")
    ax.legend()
    
    # Panel 6: Evidence summary table
    ax = axes[1, 2]
    ax.axis('tight')
    ax.axis('off')
    cell_text = [
        ["Scenario", "S Metric", "Pass (S>1.2)?"],
        ["Targeted dropout", f"{t_result['S']:.4f}", "Yes" if t_result['S']>1.2 else "No"],
        ["Random 30%", f"{r_result['S_mean']:.4f}", "Yes" if r_result['S_mean']>1.2 else "No"],
        ["Road cluster", f"{c_result['S']:.4f}", "Yes" if c_result['S']>1.2 else "No"]
    ]
    ax.table(cellText=cell_text, loc='center', cellLoc='center', edges='open')
    ax.set_title("Patent Evidence Table")
    
    plt.tight_layout()
    fig_path = f"{save_dir}/figures/pipeline_results.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Figure saved: {fig_path}")

    # STEP 8
    print("\n" + "="*60)
    print("PIPELINE COMPLETE — PATENT EVIDENCE SUMMARY")
    print("="*60)
    print(f"{'Metric':<35} {'Value':>10}")
    print("-" * 50)
    print(f"{'Static correlation':<35} {corr_static:>10.4f}")
    print(f"{'Adaptive correlation':<35} {corr_adapt:>10.4f}")
    print(f"{'Corr improvement':<35} {corr_adapt-corr_static:>+10.4f}")
    print(f"{'Adaptive iterations':<35} {adapt_result['n_iter']:>10}")
    print(f"{'S targeted dropout':<35} {t_result['S']:>10.4f}")
    print(f"{'S random 30% (mean)':<35} {r_result['S_mean']:>10.4f}")
    print(f"{'S random 30% (min)':<35} {r_result['S_min']:>10.4f}")
    print(f"{'S road cluster':<35} {c_result['S']:>10.4f}")
    
    pass_rate_1_2 = float(np.mean(np.array(r_result['S_all']) > 1.2))
    print(f"{'Pass rate S>1.2':<35} {pass_rate_1_2*100:>9.0f}%")
    print("-" * 50)
    
    overall_pass = (
        r_result['S_mean'] > 1.2 and
        r_result['S_min'] > 1.0 and
        t_result['S'] > 1.0
    )
    
    if overall_pass:
        print("PATENT CLAIM: PROVEN")
        print("S > 1.2 consistently across")
        print("multiple dropout scenarios.")
    else:
        print("PATENT CLAIM: MARGINAL")
        print("Some scenarios below threshold.")
        
    print(f"\nEvidence table: {evidence_path}")
    print(f"Results figure: {fig_path}")

    return {
        'corr_static': corr_static,
        'corr_adapt': corr_adapt,
        'S_targeted': t_result['S'],
        'S_random_mean': r_result['S_mean'],
        'S_random_min': r_result['S_min'],
        'S_cluster': c_result['S'],
        'pass_rate': pass_rate_1_2,
        'n_iter': adapt_result['n_iter'],
        'converged': adapt_result['converged'],
        'overall_pass': overall_pass,
        'R_i': R_i,
        'adapt_result': adapt_result
    }

if __name__ == "__main__":
    print("ADAPTIVE GHG EMISSION INVERSION")
    print("Patent: Variance-Stabilized Atmospheric Emission Estimation")
    print("="*60)
    print(f"Canonical params: {CANONICAL_PARAMS}")
    print("="*60)
    
    result = run_full_pipeline(
        params=CANONICAL_PARAMS,
        n_trials=50
    )
    
    if result['overall_pass']:
        print("\nREADY FOR REAL DATA (Session 9)")
        print("READY FOR DASHBOARD (Session 11)")
    else:
        print("\nReview S values before Session 9")
