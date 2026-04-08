import numpy as np
from utils.helpers import get_logger, array_summary
from core.inversion import tikhonov_solve
from core.weights import compute_leverage_weights
from core.residuals import compute_residuals, weighted_residual_loss

logger = get_logger("adaptive")

def adaptive_loop(
    H: np.ndarray,
    y: np.ndarray,
    x0: np.ndarray,
    c0: np.ndarray = None,
    lam: float = 0.01,
    gamma: float = 0.01,
    eta: float = 10.0,
    alpha: float = 0.3,
    max_iter: int = 50,
    eps: float = 1e-8
) -> dict:

    # STEP 0 â€” INITIALIZATION
    if c0 is None:
        c0 = np.ones(H.shape[1])

    theta = np.zeros(H.shape[1])
    theta_prev = np.zeros(H.shape[1])

    def theta_to_c(theta):
        # Log-parameterization keeps c strictly positive
        c = c0 * np.exp(theta)
        return c

    losses = []
    all_residuals = []
    theta_norms = []

    # STEP 1 â€” STATIC BASELINE (before loop)
    x_static = tikhonov_solve(H, y, x0, c=c0, lam=lam, quiet=True)
    r_static = compute_residuals(H, x_static, y, c=c0)['r']

    # STEP 2 â€” ADAPTIVE LOOP
    R_i = np.ones(H.shape[0])  # initialise so return dict is always valid
    for k in range(max_iter):

        # A. Current c from theta
        c_k = theta_to_c(theta)

        # B. Solve inversion with current c
        x_hat = tikhonov_solve(H, y, x0, c=c_k, lam=lam, quiet=True)

        # C. Residuals: r = y - H @ (c_k * x_hat)
        res = compute_residuals(H, x_hat, y, c=c_k)
        r = res['r']

        # D. Leverage weights R_i
        R_i = compute_leverage_weights(H, c_k, gamma=gamma)

        # E. Loss = Var(r) directly
        loss = float(np.var(r))
        losses.append(loss)
        all_residuals.append(r.copy())

        # Adaptive learning rate: halve eta if loss is diverging
        if k > 2 and losses[-1] > losses[-3]:
            logger.info(f"Loss increasing at iter {k}, reducing eta")
            eta = eta * 0.5

        # F. ANALYTICAL GRADIENT OF VAR(r) WEIGHTED BY R_i
        n = len(r)
        grad = np.zeros(len(theta))
        for j in range(len(theta)):
            dr_j = -H[:, j] * c_k[j] * x_hat[j]
            grad[j] = float(
                2/n * np.dot(R_i * r, dr_j)
                - 2 * np.mean(R_i * r) * np.mean(dr_j)
            )

        # G. DAMPED GRADIENT UPDATE WITH MOMENTUM
        delta_theta = -eta * grad + alpha * (theta - theta_prev)
        theta_prev = theta.copy()
        theta = theta + delta_theta
        theta_norms.append(float(np.linalg.norm(delta_theta)))

        # H. CONVERGENCE CHECK
        if k > 0 and abs(losses[-1] - losses[-2]) < eps:
            logger.info(
                f"Converged at iter {k+1}: "
                f"dVar={abs(losses[-1]-losses[-2]):.2e}"
            )
            break

        # I. LOG PROGRESS every 10 iterations
        if k % 10 == 0:
            logger.info(
                f"Iter {k+1}: loss={loss:.6f} "
                f"||delta_theta||={theta_norms[-1]:.6f}"
            )

    # STEP 3 â€” FINAL ADAPTIVE RESULT
    c_final = theta_to_c(theta)
    x_adapt = tikhonov_solve(H, y, x0, c=c_final, lam=lam, quiet=True)
    r_adapt = compute_residuals(H, x_adapt, y, c=c_final)['r']

    # STEP 4 â€” RETURN DICT
    return {
        'x_static'  : x_static,
        'x_adapt'   : x_adapt,
        'r_static'  : r_static,
        'r_adapt'   : r_adapt,
        'losses'    : losses,
        'theta_norms': theta_norms,
        'theta_final': theta,
        'c_final'   : c_final,
        'R_i_final' : R_i,
        'n_iter'    : k + 1,
        'converged' : theta_norms[-1] < eps if theta_norms else False
    }


if __name__ == "__main__":
    import time
    from simulation.synthetic import generate_synthetic_data
    from core.inversion import compute_correlation

    data = generate_synthetic_data()

    print("Running adaptive loop (analytical gradient)...")

    t0 = time.time()
    result = adaptive_loop(
        data['H'],
        data['y'],
        data['x0'],
        c0=data['c0_wrong'],
        lam=0.01,
        gamma=0.005,
        eta=10.0,
        alpha=0.3,
        max_iter=50
    )
    elapsed = time.time() - t0

    corr_static = compute_correlation(result['x_static'], data['x_true'])
    corr_adapt  = compute_correlation(result['x_adapt'],  data['x_true'])

    print(f"\n=== ADAPTIVE LOOP RESULTS ===")
    print(f"Runtime:            {elapsed:.2f}s")
    print(f"Iterations run:     {result['n_iter']}")
    print(f"Converged:          {result['converged']}")
    print(f"Initial loss:       {result['losses'][0]:.6f}")
    print(f"Final loss:         {result['losses'][-1]:.6f}")
    print(f"Loss reduction:     {(1-result['losses'][-1]/result['losses'][0])*100:.1f}%")
    print(f"\nCorr static:        {corr_static:.4f}")
    print(f"Corr adaptive:      {corr_adapt:.4f}")
    print(f"Corr improvement:   {corr_adapt-corr_static:+.4f}")
    print(f"\nVar(r_static):      {np.var(result['r_static']):.6f}")
    print(f"Var(r_adaptive):    {np.var(result['r_adapt']):.6f}")

    S_single = np.var(result['r_static']) / np.var(result['r_adapt'])
    print(f"S (single run):     {S_single:.4f}")
    print(f"\nTheta final mean:   {result['theta_final'].mean():.4f}")
    print(f"c_final mean:       {result['c_final'].mean():.4f}")

    if S_single > 1.0 and elapsed < 5.0:
        print("\nadaptive.py verification PASSED")
        print("S > 1.0 and runtime < 5s")
    elif S_single > 1.0:
        print("\nadaptive.py verification PARTIAL")
        print(f"S > 1.0 but runtime={elapsed:.1f}s (target <5s)")
    else:
        print("\nWARNING: S <= 1.0")
        print("Analytical gradient not yet converging. Tune eta.")
