#!/usr/bin/env python3
"""
Single-user semantic distortion trace comparison.

Compares DR8, DR9 and Semantic policy for one monitored process, plotting:
- semantic distortion over time
- adaptive threshold (semantic only)
- transmission instants

Outputs:
- simulation_results/single_user_trace/<timestamp>/trace_<protocol>.csv
- simulation_results/single_user_trace/<timestamp>/single_user_distortion_trace.png
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

plt.style.use('classic')
matplotlib.rcParams.update({
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': ':',
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.8,
})

# Add project root to import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lrfhss.traffic import DistortionAwareExponentialTraffic, SemanticTraffic
from examples.organized.sim_params import get_base_traffic_params, get_semantic_params


SIM_TIME = 3600
SEED = 7
INTERVAL_MODE = 'periodic'  # 'periodic' or 'poisson'

COMMON_PARAMS = get_base_traffic_params()
COMMON_PARAMS['track_trace'] = True

SEMANTIC_PARAMS = get_semantic_params()
SEMANTIC_PARAMS['track_trace'] = True
SEMANTIC_PARAMS['beta'] = 0.0005


def _build_output_dir() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(project_root, 'simulation_results', 'single_user_trace', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _generate_shared_poisson_intervals(sim_time: float, average_interval: float, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    now = 0.0
    intervals = []
    scale = float(average_interval)

    while True:
        dt = float(rng.exponential(scale=scale))
        if now + dt > sim_time:
            break
        now += dt
        intervals.append(dt)

    return intervals


def _generate_shared_periodic_intervals(sim_time: float, average_interval: float) -> list[float]:
    dt = float(average_interval)
    if dt <= 0:
        return []

    intervals = []
    now = 0.0
    while now + dt <= sim_time:
        intervals.append(dt)
        now += dt
    return intervals


def _build_decision_intervals(sim_time: float, average_interval: float, seed: int, mode: str) -> list[float]:
    normalized = mode.strip().lower()
    if normalized == 'periodic':
        return _generate_shared_periodic_intervals(sim_time=sim_time, average_interval=average_interval)
    if normalized == 'poisson':
        return _generate_shared_poisson_intervals(
            sim_time=sim_time,
            average_interval=average_interval,
            seed=seed,
        )
    raise ValueError("INTERVAL_MODE must be 'periodic' or 'poisson'")


def _simulate_trace(traffic_gen, decision_intervals: list[float]) -> pd.DataFrame:
    for dt in decision_intervals:
        traffic_gen.on_decision_epoch(dt)

    trace = traffic_gen.get_trace()
    if not trace:
        return pd.DataFrame(columns=['time', 'distortion', 'threshold', 'tx_decision'])
    return pd.DataFrame(trace)


def _plot_protocol(ax, df: pd.DataFrame, protocol: str):
    if df.empty:
        ax.text(0.5, 0.5, 'No events in trace', ha='center', va='center', transform=ax.transAxes)
        ax.grid(True, alpha=0.3)
        return

    ax.plot(df['time'], df['distortion'], label=r'$D_k(t)$', color='#1f77b4')

    if np.isfinite(df['threshold']).any():
        ax.plot(df['time'], df['threshold'], label=r'$\epsilon_{th}(t)$', color='#ff7f0e', linestyle='--')

    tx_df = df[df['tx_decision'] == True]
    if not tx_df.empty:
        ax.scatter(
            tx_df['time'],
            tx_df['distortion'],
            color='#2ca02c',
            marker='o',
            s=24,
            label='TX',
            zorder=3,
        )

    if protocol == 'Semantic':
        no_tx_df = df[df['tx_decision'] == False]
        if not no_tx_df.empty:
            ax.scatter(
                no_tx_df['time'],
                no_tx_df['distortion'],
                color='#d62728',
                marker='x',
                s=26,
                alpha=0.8,
                label='No TX',
                zorder=3,
            )

    ax.text(0.01, 0.92, protocol, transform=ax.transAxes, fontsize=10)
    ax.set_ylabel(r'$D_k(t)$')
    ax.grid(True, alpha=0.3)


def main():
    np.random.seed(SEED)

    print('=' * 80)
    print('Single-User Distortion Trace: DR8 vs DR9 vs Semantic')
    print('=' * 80)

    output_dir = _build_output_dir()
    decision_intervals = _build_decision_intervals(
        sim_time=SIM_TIME,
        average_interval=COMMON_PARAMS['average_interval'],
        seed=SEED,
        mode=INTERVAL_MODE,
    )
    print(f'Decision epoch mode: {INTERVAL_MODE} | Events: {len(decision_intervals)}')

    protocol_to_gen = {
        'DR8': DistortionAwareExponentialTraffic(dict(COMMON_PARAMS)),
        'DR9': DistortionAwareExponentialTraffic(dict(COMMON_PARAMS)),
        'Semantic': SemanticTraffic(dict(SEMANTIC_PARAMS)),
    }

    traces = {}
    for protocol, gen in protocol_to_gen.items():
        print(f'Running single-user trace for {protocol}...')
        traces[protocol] = _simulate_trace(gen, decision_intervals)
        csv_path = os.path.join(output_dir, f'trace_{protocol.lower()}.csv')
        traces[protocol].to_csv(csv_path, index=False)
        print(f'[OK] Saved: {csv_path}')

    semantic_df = traces['Semantic']
    if not semantic_df.empty:
        thr_min = float(np.nanmin(semantic_df['threshold']))
        thr_max = float(np.nanmax(semantic_df['threshold']))
        tx_rate = float(semantic_df['tx_decision'].mean())
        print(f'Semantic threshold range: [{thr_min:.3f}, {thr_max:.3f}]')
        print(f'Semantic transmission ratio: {tx_rate:.3f}')

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    for ax, protocol in zip(axes, ['DR8', 'DR9', 'Semantic']):
        _plot_protocol(ax, traces[protocol], protocol)

    axes[-1].set_xlabel(r'Time, $t$ (s)')
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    if handles:
        unique = {}
        for h, l in zip(handles, labels):
            if l not in unique:
                unique[l] = h
        fig.legend(
            list(unique.values()),
            list(unique.keys()),
            loc='upper center',
            ncol=4,
            frameon=False,
            scatterpoints=1,
            numpoints=1,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    plot_path = os.path.join(output_dir, 'single_user_distortion_trace.png')
    fig.savefig(plot_path, dpi=250, bbox_inches='tight')
    plt.close(fig)

    print(f'[OK] Plot saved: {plot_path}')
    print('=' * 80)
    print('[DONE] Trace simulation finished.')
    print(f'Results folder: {output_dir}')
    print('=' * 80)


if __name__ == '__main__':
    main()
