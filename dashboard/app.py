import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

# Insert root to import from core, simulation, pipeline, utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.helpers import get_logger

logger = get_logger("dashboard")

# PAGE CONFIG
st.set_page_config(
    page_title="Adaptive Emission Inversion Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SIDEBAR
st.sidebar.title("Emission Inversion")
st.sidebar.caption("Adaptive Representativeness-Weighted Inversion")

st.sidebar.markdown("### DATA SETTINGS")
run_mode = st.sidebar.radio("Data source", ["Real CPCB Data", "Synthetic"])

api_key = ""
resource_id = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
if run_mode == "Real CPCB Data":
    api_key = st.sidebar.text_input("data.gov.in API Key", type="password", placeholder="Enter your API key")
    resource_id = st.sidebar.text_input("Resource ID", value="3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69")

st.sidebar.markdown("### INVERSION HYPERPARAMETERS")
lam = st.sidebar.slider("Regularisation λ", 0.001, 0.1, 0.01, step=0.001, format="%.3f")
eta = st.sidebar.slider("Learning Rate η", 0.05, 0.5, 0.2, step=0.05)
gamma = st.sidebar.slider("Leverage Weight γ", 0.001, 0.05, 0.005, step=0.001, format="%.3f")
alpha = st.sidebar.slider("Momentum α", 0.0, 0.9, 0.3, step=0.1)
max_iter = st.sidebar.slider("Max Iterations", 10, 50, 25)

st.sidebar.markdown("### DROPOUT TEST")
dropout_frac = st.sidebar.slider("Random Dropout Fraction", 0.1, 0.5, 0.25, step=0.05)
n_trials = st.sidebar.slider("Dropout Trials", 10, 50, 30)

run_btn = st.sidebar.button("▶ Run Pipeline", type="primary", use_container_width=True)

# HEADER
st.title("🌍 Adaptive Emission Inversion Dashboard")
st.caption("Patent: Adaptive Representativeness-Weighted Atmospheric Emission Inversion")

# PIPELINE CACHED FUNCTIONS
@st.cache_data
def run_synthetic_cached(lam, eta, gamma, alpha, max_iter, n_trials, dropout_frac):
    from simulation.synthetic import generate_synthetic_data
    from simulation.dropout import random_dropout_comparison
    from core.adaptive import adaptive_loop
    from core.weights import compute_leverage_weights
    from core.inversion import tikhonov_solve
    
    data = generate_synthetic_data(noise_std=0.05)
    
    x_static = tikhonov_solve(
        data['H'], data['y'], data['x0'],
        c=data['c0_wrong'], lam=lam, quiet=True
    )
    
    result = adaptive_loop(
        data['H'], data['y'], data['x0'],
        c0=data['c0_wrong'], lam=lam, gamma=gamma,
        eta=eta, alpha=alpha, max_iter=max_iter
    )
    
    R_i = compute_leverage_weights(data['H'], result['c_final'], gamma)
    
    dropout = random_dropout_comparison(
        data['H'], data['y'], data['x0'], data['x_true'], data['c0_wrong'],
        dropout_frac=dropout_frac, n_trials=n_trials,
        lam=lam, gamma=gamma, eta=eta, alpha=alpha, max_iter=15
    )
    return data, x_static, result, R_i, dropout

@st.cache_data
def run_realdata_cached(api_key, resource_id, lam, eta, gamma, alpha, max_iter, n_trials, dropout_frac):
    from data.loaders import load_hyderabad_data
    from core.adaptive import adaptive_loop
    from core.weights import compute_leverage_weights
    from core.inversion import tikhonov_solve
    from pipeline.run_realdata import real_data_dropout_test
    
    params = {'lam': lam, 'eta': eta, 'gamma': gamma, 'alpha': alpha}
    
    data = load_hyderabad_data(api_key=api_key, resource_id=resource_id)
    if not data:
        return None
        
    x_static = tikhonov_solve(
        data['H'], data['y'], data['x0'],
        c=data['c0_initial'], lam=lam, quiet=True
    )
    
    result = adaptive_loop(
        data['H'], data['y'], data['x0'],
        c0=data['c0_initial'], lam=lam, gamma=gamma,
        eta=eta, alpha=alpha, max_iter=max_iter
    )
    
    R_i = compute_leverage_weights(data['H'], result['c_final'], gamma)
    
    dropout = real_data_dropout_test(
        data['H'], data['y'], data['x0'], params,
        n_trials=n_trials, dropout_frac=dropout_frac
    )
    return data, x_static, result, R_i, dropout

# EXECUTE CONFIG BUTTON
if run_btn:
    with st.spinner("Running pipeline..."):
        if run_mode == "Synthetic":
            output = run_synthetic_cached(lam, eta, gamma, alpha, max_iter, n_trials, dropout_frac)
            data, x_static, result, R_i, dropout = output
            
            station_names = [f"S{i:02d}" for i in range(data['H'].shape[0])]
            source_pos = data['source_positions']
            station_pos = data['station_positions']
            x_adapt = result['x_adapt']
        else:
            if not api_key or len(api_key) < 10:
                st.error("Enter your data.gov.in API key in the sidebar")
                st.stop()
            output = run_realdata_cached(api_key, resource_id, lam, eta, gamma, alpha, max_iter, n_trials, dropout_frac)
            if output is None:
                st.error("Failed to load CPCB data")
                st.stop()
            data, x_static, result, R_i, dropout = output
            
            station_names = data['station_names']
            source_pos = data['source_positions']
            station_pos = data['station_positions']
            x_adapt = result['x_adapt']
            
    st.success("Pipeline complete")
else:
    st.info("Set parameters in the sidebar and click ▶ Run Pipeline")
    st.stop()

# METRICS ROW
metric_cols = st.columns(5)
with metric_cols[0]:
    if run_mode == "Synthetic":
        from core.inversion import compute_correlation
        c_stat = compute_correlation(x_static, data['x_true'])
        st.metric("Static Correlation", f"{c_stat:.4f}")
    else:
        st.metric("Static Correlation", "N/A (real data)")

with metric_cols[1]:
    if run_mode == "Synthetic":
        c_adapt = compute_correlation(x_adapt, data['x_true'])
        diff = c_adapt - c_stat
        st.metric("Adaptive Correlation", f"{c_adapt:.4f}", f"{diff:+.4f}")
    else:
        st.metric("Adaptive Correlation", "N/A (real data)")

with metric_cols[2]:
    s_mean = dropout['S_mean']
    delta = s_mean - 1.2
    st.metric("Stability S (Random)", f"{s_mean:.3f}", f"{delta:+.3f} vs 1.2")

with metric_cols[3]:
    st.metric("Stability S (Min)", f"{dropout['S_min']:.3f}")

with metric_cols[4]:
    st.metric("Adaptive Iterations", f"{result['n_iter']}", "converged" if result['converged'] else "max reached")

# 4 TABS
tab1, tab2, tab3, tab4 = st.tabs(["🗺 Emission Map", "⊛ Station Weights (Ri)", "📉 Convergence", "📊 Stability Proof"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=source_pos[:,0], y=source_pos[:,1], mode='markers', marker=dict(size=12, color=x_static, colorscale='Reds', showscale=True), name='Sources'))
        fig1.add_trace(go.Scatter(x=station_pos[:,0], y=station_pos[:,1], mode='markers', marker=dict(size=8, color='white', line=dict(color='black', width=1)), name='Stations'))
        fig1.update_layout(title="Static Inversion x̂")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=source_pos[:,0], y=source_pos[:,1], mode='markers', marker=dict(size=12, color=x_adapt, colorscale='Viridis', showscale=True), name='Sources'))
        sz = R_i * 20 + 4
        fig2.add_trace(go.Scatter(x=station_pos[:,0], y=station_pos[:,1], mode='markers', marker=dict(size=sz, color='white', line=dict(color='black', width=1)), name='Stations'))
        fig2.update_layout(title="Adaptive x̂* (Patent)")
        st.plotly_chart(fig2, use_container_width=True)
        
    df_ri = pd.DataFrame({
        'Station': [str(s) for s in station_names],
        'R_i': [float(r) for r in R_i],
        'Zone': ['unknown']*len(R_i)
    })
    st.caption("Station Leverage Values:")
    st.dataframe(
        df_ri.astype({
            'Station': str,
            'R_i': float
        })
    )

with tab2:
    colors = ['green' if r > 0.66 else 'orange' if r > 0.33 else 'blue' for r in R_i]
    fig_bar = go.Figure(data=[go.Bar(x=station_names, y=R_i, marker_color=colors)])
    fig_bar.add_hline(y=0.33, line_dash="dash", line_color="gray")
    fig_bar.add_hline(y=0.66, line_dash="dash", line_color="red")
    fig_bar.update_layout(title="Leverage Scores Ri per Station", yaxis_title="R_i value")
    st.plotly_chart(fig_bar, use_container_width=True)
    
    mc1, mc2, mc3 = st.columns(3)
    high_lev = sum(R_i > 0.66)
    max_ri_idx = np.argmax(R_i)
    max_ri_val = R_i[max_ri_idx]
    max_ri_name = station_names[max_ri_idx]
    mean_ri = np.mean(R_i)
    
    mc1.metric("High leverage count (Ri > 0.66)", str(high_lev))
    mc2.metric("Max Ri", f"{max_ri_val:.3f}", max_ri_name)
    mc3.metric("Mean Ri", f"{mean_ri:.3f}")

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        fig_loss = px.line(y=result['losses'], markers=True, title="J(theta) Loss per Iteration")
        fig_loss.update_layout(xaxis_title="Iteration", yaxis_title="J(theta)")
        st.plotly_chart(fig_loss, use_container_width=True)
        
    with c2:
        from core.residuals import compute_residuals
        c0_for_res = data['c0_wrong'] if run_mode == "Synthetic" else data['c0_initial']
        r_s = compute_residuals(data['H'], x_static, data['y'], c=c0_for_res)['r']
        r_a = compute_residuals(data['H'], x_adapt, data['y'], c=result['c_final'])['r']
        
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(y=r_s, name="Static", mode='lines+markers', line=dict(color='red')))
        fig_r.add_trace(go.Scatter(y=r_a, name="Adaptive", mode='lines+markers', line=dict(color='blue')))
        fig_r.update_layout(title="Residuals: Static vs Adaptive")
        st.plotly_chart(fig_r, use_container_width=True)
        
    theta_vals = result['theta_final']
    t_colors = ['orange' if t > 0 else 'blue' for t in theta_vals]
    fig_t = go.Figure(data=[go.Bar(y=theta_vals, marker_color=t_colors)])
    fig_t.update_layout(title="Final theta values (conversion factor updates)")
    st.plotly_chart(fig_t, use_container_width=True)

with tab4:
    s_mean_str = f"{s_mean:.4f}"
    if s_mean > 1.2:
        st.markdown(f"<h1 style='text-align: center; color: green;'>S_mean = {s_mean_str}</h1>", unsafe_allow_html=True)
    elif s_mean > 1.0:
        st.markdown(f"<h1 style='text-align: center; color: orange;'>S_mean = {s_mean_str}</h1>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h1 style='text-align: center; color: red;'>S_mean = {s_mean_str}</h1>", unsafe_allow_html=True)
        
    fig_hist = px.histogram(dropout['S_all'], nbins=15)
    fig_hist.add_vline(x=1.0, line_dash="dash", line_color="red")
    fig_hist.add_vline(x=1.2, line_dash="dash", line_color="green")
    st.plotly_chart(fig_hist, use_container_width=True)
    
    table_data = [
        {"Scenario": "Random dropout mean", "S": f"{dropout['S_mean']:.4f}", "Pass": "✓" if dropout['S_mean'] > 1.2 else "✗"},
        {"Scenario": "Random dropout min", "S": f"{dropout['S_min']:.4f}", "Pass": "✓" if dropout['S_min'] > 1.0 else "✗"},
        {"Scenario": "Pass rate > 1.0", "S": f"{dropout['pass_rate']*100:.1f}%", "Pass": "✓" if dropout['pass_rate'] == 1.0 else "✗"},
        {"Scenario": "Pass rate > 1.2", "S": f"{np.mean(np.array(dropout['S_all'])>1.2)*100:.1f}%", "Pass": "✓" if np.mean(np.array(dropout['S_all'])>1.2) > 0.5 else "✗"},
    ]
    st.dataframe(pd.DataFrame(table_data), use_container_width=True)
    
    if s_mean > 1.2:
        st.success("Patent Claim: S > 1.2 proven on real CPCB Hyderabad data")
    elif s_mean > 1.0:
        st.warning("Patent Claim: S > 1.0 supported, but strictly < 1.2")
    else:
        st.error("Patent Claim Failed")
