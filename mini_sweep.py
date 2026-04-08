import numpy as np
from simulation.synthetic import generate_synthetic_data
from simulation.dropout import random_dropout_comparison

data = generate_synthetic_data(n_stations=30, n_sources=100)
H = data['H']
y = data['y']
x0 = data['x0']
x_true = data['x_true']
c0 = data['c0_wrong']

print("eta    gamma  | S_mean  S_min   pass%")
print("-" * 45)

best_S = 0
best_cfg = {}

for eta in [0.05, 0.1, 0.2, 0.3, 0.5]:
    for gamma in [0.001, 0.005, 0.01, 0.05]:
        try:
            res = random_dropout_comparison(
                H, y, x0, x_true, c0,
                dropout_frac=0.3,
                n_trials=20,
                eta=eta,
                gamma=gamma
            )
            S_mean = res['S_mean']
            S_min = res['S_min']
            pass_pct = res['pass_rate_1_2'] * 100
            print(f"eta={eta:<5} gamma={gamma:<6} | {S_mean:.4f}  {S_min:.4f}  {pass_pct:.0f}%")
            if S_mean > best_S:
                best_S = S_mean
                best_cfg = {'eta': eta, 'gamma': gamma}
        except Exception as e:
            print(f"eta={eta} gamma={gamma} | ERROR: {e}")

print()
print(f"BEST: S_mean={best_S:.4f} at {best_cfg}")
