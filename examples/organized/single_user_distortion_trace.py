#!/usr/bin/env python3
"""
Single-user semantic distortion trace comparison.

A single AR(1) process is pre-generated and shared across all three protocols
(DR8, DR9, Semantic), so they are evaluated on exactly the same underlying
physical signal.  This produces three distinct, informative curves per protocol:

  - x_k(t)      : true process state (continuous, high-resolution)
  - x̂_k(t)     : gateway estimate (step function that jumps on successful TX)
  - D_k(t)      : |x_k(t) - x̂_k(t)|  real instantaneous distortion
  - ε_th(t)     : adaptive transmission threshold (Semantic only)

One figure per protocol is generated (3 total), each with two stacked panels:
  Panel 1  – x_k(t) and x̂_k(t) on the same axes
  Panel 2  – D_k(t) and ε_th(t), with TX/no-TX markers

Outputs (simulation_results/single_user_trace/<timestamp>/):
  trace_<protocol>.csv                   epoch-level data
  continuous_process.csv                 pre-generated x_k(t) at 1 s resolution
  trace_DR8.{png,eps,jpg}
  trace_DR9.{png,eps,jpg}
  trace_Semantic.{png,eps,jpg}
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lrfhss.ar1_process import AR1Process
from lrfhss.traffic import DistortionAwareExponentialTraffic, SemanticTraffic
from examples.organized.sim_params import get_base_traffic_params, get_semantic_params
from examples.organized.plot_style import apply_matlab_style, save_fig, COLORS


# ── Simulation parameters ────────────────────────────────────────────────────
SIM_TIME = 3600
SEED = 7
DT_FINE = 1.0   # AR(1) pre-generation step (s)


def _build_output_dir() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(project_root, 'simulation_results', 'single_user_trace', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _build_decision_intervals(sim_time: float, average_interval: float,
                               seed: int, mode: str) -> list[float]:
    normalized = mode.strip().lower()
    if normalized == 'periodic':
        dt = float(average_interval)
        intervals, now = [], 0.0
        while now + dt <= sim_time:
            intervals.append(dt)
            now += dt
        return intervals
    if normalized == 'poisson':
        rng = np.random.default_rng(seed)
        intervals, now = [], 0.0
        while True:
            dt = float(rng.exponential(scale=average_interval))
            if now + dt > sim_time:
                break
            now += dt
            intervals.append(dt)
        return intervals
    raise ValueError("INTERVAL_MODE must be 'periodic' or 'poisson'")


def _build_x_hat_continuous(t_cont: np.ndarray, epoch_df: pd.DataFrame) -> np.ndarray:
    """
    Build x̂_k(t) as a step function on the continuous time grid.

    At each epoch where tx_decision=True, x̂ jumps to x_current.
    Between transmissions, x̂ stays constant (held value).
    """
    x_hat = np.full_like(t_cont, np.nan)
    tx_rows = epoch_df[epoch_df['tx_decision']].copy()

    if tx_rows.empty:
        # Never transmitted: fill with first x_hat from epoch data
        if 'x_hat' in epoch_df.columns and not epoch_df.empty:
            x_hat[:] = epoch_df['x_hat'].iloc[0]
        return x_hat

    # Fill step-wise
    current_val = epoch_df['x_hat'].iloc[0]  # value before first TX
    prev_idx = 0
    for _, row in tx_rows.iterrows():
        t_tx = row['time']
        mask = (t_cont >= prev_idx * DT_FINE) & (t_cont < t_tx)
        x_hat[mask] = current_val
        current_val = row['x_current']
        prev_idx = int(np.searchsorted(t_cont, t_tx))

    # Fill remainder after last TX
    x_hat[prev_idx:] = current_val
    return x_hat


def _simulate_trace(traffic_gen, decision_intervals: list[float]) -> pd.DataFrame:
    for dt in decision_intervals:
        traffic_gen.on_decision_epoch(dt)
    trace = traffic_gen.get_trace()
    if not trace:
        return pd.DataFrame(columns=['time', 'x_current', 'x_hat', 'distortion',
                                     'threshold', 'tx_decision'])
    return pd.DataFrame(trace)


def _plot_protocol_figure(
    protocol: str,
    t_cont: np.ndarray,
    x_cont: np.ndarray,
    x_hat_cont: np.ndarray,
    epoch_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Two-panel figure for one protocol."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    clr = COLORS.get(protocol, '#0072BD')

    # ── Panel 1: x_k(t) and x̂_k(t) ──────────────────────────────────
    ax1.plot(t_cont, x_cont, color=clr, linewidth=1.2, label=r'$x_k(t)$')
    ax1.step(t_cont, x_hat_cont, where='post', color='#D95319', linewidth=1.2,
             linestyle='--', label=r'$\hat{x}_k(t)$')

    # Mark TX instants on x_k(t)
    tx_df = epoch_df[epoch_df['tx_decision']]
    if not tx_df.empty:
        ax1.scatter(tx_df['time'], tx_df['x_current'], color='k',
                    marker='v', s=30, zorder=4, label='TX epoch')

    ax1.set_ylabel(r'Process value')
    ax1.legend(loc='upper right', ncol=3)
    ax1.tick_params(direction='in')

    # ── Panel 2: D_k(t) and ε_th ──────────────────────────────────────
    # Real continuous distortion from pre-generated process
    d_cont = np.abs(x_cont - x_hat_cont)
    ax2.plot(t_cont, d_cont, color=clr, linewidth=1.2, label=r'$D_k(t)$')

    # Threshold curve (Semantic only; NaN for DR8/DR9)
    if 'threshold' in epoch_df.columns and np.isfinite(epoch_df['threshold']).any():
        ax2.step(epoch_df['time'], epoch_df['threshold'], where='post',
                 color='#7E2F8E', linestyle=':', linewidth=1.4,
                 label=r'$\varepsilon_{\mathrm{th}}(\Delta)$')

    # TX/No-TX markers: placed just BEFORE the update (D_k before it drops to zero).
    # epoch 'time' is the end of the interval; the decision fires at t_tx,
    # so we look up D_k at the last fine-grid sample strictly before t_tx.
    def _d_before(times_arr):
        """D_k value one dt_fine step before each epoch time."""
        vals = []
        for t_tx in times_arr:
            idx = int(np.searchsorted(t_cont, t_tx, side='left'))
            idx = max(idx - 1, 0)          # one sample before the update
            vals.append(d_cont[idx])
        return np.array(vals)

    if not tx_df.empty:
        ax2.scatter(tx_df['time'], _d_before(tx_df['time'].values),
                    color='k', marker='v', s=30, zorder=4, label='TX')

    # No-TX markers (Semantic only)
    if protocol == 'Semantic':
        no_tx_df = epoch_df[~epoch_df['tx_decision']]
        if not no_tx_df.empty:
            ax2.scatter(no_tx_df['time'], _d_before(no_tx_df['time'].values),
                        color='#555555', marker='x', s=32, alpha=0.9,
                        zorder=4, label='No TX')

    ax2.set_ylabel(r'$D_k(t) = |x_k - \hat{x}_k|$')
    ax2.set_xlabel(r'Time, $t$ (s)')
    ax2.legend(loc='upper right', ncol=3)
    ax2.tick_params(direction='in')

    fig.tight_layout()
    save_fig(fig, output_dir, f'trace_{protocol}')


def main():
    apply_matlab_style()
    np.random.seed(SEED)

    base_params = get_base_traffic_params()
    semantic_params = get_semantic_params()
    # Use a beta that makes the threshold visible within the trace window:
    # ε_th(λ) = max(ε_min, ε_0 - β·λ) = max(0.2, 1.25 - 0.003·300) = 0.35
    semantic_params['beta'] = 0.003

    lam = base_params['average_interval']   # 300 s

    print('=' * 80)
    print('Single-User Distortion Trace: DR8 vs DR9 vs Semantic')
    print('=' * 80)

    output_dir = _build_output_dir()

    # ── Pre-generate the shared AR(1) process ────────────────────────────
    # alpha=0.95 is defined per epoch (λ=300 s); lambda_ref scales it to per-second.
    proc = AR1Process(
        alpha=base_params.get('alpha', 0.95),
        sigma_w=base_params.get('sigma_w', 1.0),
        sim_time=SIM_TIME,
        dt_fine=DT_FINE,
        lambda_ref=lam,
        seed=SEED,
    )
    t_cont, x_cont = proc.full_trace()
    pd.DataFrame({'time': t_cont, 'x': x_cont}).to_csv(
        os.path.join(output_dir, 'continuous_process.csv'), index=False
    )
    print(f'AR(1) process: {len(t_cont)} points at {DT_FINE}s resolution')

    # ── Build decision epochs (shared across all protocols) ────────────────────
    decision_intervals = _build_decision_intervals(
        sim_time=SIM_TIME,
        average_interval=lam,
        seed=SEED,
        mode='periodic',
    )
    print(f'Decision epochs: {len(decision_intervals)} (periodic, λ={lam}s)')

    # ── Run each protocol on the same process ─────────────────────────────
    common = dict(base_params)
    common['track_trace'] = True
    common['ar1_process'] = proc

    sem = dict(semantic_params)
    sem['track_trace'] = True
    sem['ar1_process'] = proc

    protocols = {
        'DR8':      DistortionAwareExponentialTraffic(dict(common)),
        'DR9':      DistortionAwareExponentialTraffic(dict(common)),
        'Semantic': SemanticTraffic(dict(sem)),
    }

    traces: dict[str, pd.DataFrame] = {}
    for protocol, gen in protocols.items():
        print(f'Simulating {protocol}...')
        traces[protocol] = _simulate_trace(gen, decision_intervals)
        csv_path = os.path.join(output_dir, f'trace_{protocol.lower()}.csv')
        traces[protocol].to_csv(csv_path, index=False)
        tx_rate = float(traces[protocol]['tx_decision'].mean())
        print(f'  TX rate: {tx_rate:.3f}  |  epochs: {len(traces[protocol])}')

    if 'threshold' in traces['Semantic'].columns:
        thr = traces['Semantic']['threshold']
        print(f'Semantic threshold range: [{thr.min():.3f}, {thr.max():.3f}]')

    # ── Generate one figure per protocol ─────────────────────────────────
    for protocol, epoch_df in traces.items():
        x_hat_cont = _build_x_hat_continuous(t_cont, epoch_df)
        _plot_protocol_figure(protocol, t_cont, x_cont, x_hat_cont, epoch_df, output_dir)
        print(f'[OK] Figure saved: trace_{protocol}.{{png,eps,jpg}}')

    print('=' * 80)
    print('[DONE] Trace simulation finished.')
    print(f'Results folder: {output_dir}')
    print('=' * 80)


if __name__ == '__main__':
    main()
