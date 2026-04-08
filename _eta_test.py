import logging
logging.getLogger("weights").setLevel(logging.WARNING)
logging.getLogger("adaptive").setLevel(logging.WARNING)
logging.getLogger("residuals").setLevel(logging.WARNING)
logging.getLogger("dropout").setLevel(logging.WARNING)

from simulation.synthetic import generate_synthetic_data
from simulation.dropout import random_dropout_comparison

print("\n" + "="*50)
print("ETA SENSITIVITY TEST")
print("="*50)

data = generate_synthetic_data()

for eta_test in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]:
    r = random_dropout_comparison(
        data['H'], data['y'],
        data['x0'], data['x_true'],
        data['c0_wrong'],
        dropout_frac=0.40,
        n_trials=10,
        lam=0.01,
        gamma=0.005,
        eta=eta_test,
        alpha=0.3,
        max_iter=50,
        random_seed=42
    )
    print(
        f"eta={eta_test:.1f} | "
        f"S_mean={r['S_mean']:.4f} | "
        f"S_min={r['S_min']:.4f} | "
        f"pass%={r['pass_rate']*100:.0f}%"
    )
