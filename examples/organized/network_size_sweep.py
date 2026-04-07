#!/usr/bin/env python3
"""
Sweep by number of devices comparing DR8, DR9 and Semantic communication.

Generated metrics:
- Success Probability
- Mean AoI
- Mean Semantic Distortion
- Energy Efficiency

Outputs:
- simulation_results/network_sweep/<timestamp>/raw_runs.csv
- simulation_results/network_sweep/<timestamp>/summary_by_protocol.csv
- simulation_results/network_sweep/<timestamp>/metrics_by_num_devices.png
- simulation_results/network_sweep/<timestamp>/tradeoff_aoi_vs_energy.png
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Add project root to import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lrfhss.run import run_sim_detailed
from lrfhss.settings import Settings
from lrfhss.traffic import DistortionAwareExponentialTraffic, PrecomputedSemanticTraffic
from examples.organized.sim_params import get_base_traffic_params, get_semantic_params
from examples.organized.plot_style import apply_matlab_style, save_fig, protocol_plot_kwargs


FULL_SIM_TIME = 3600
FULL_SEEDS = [0, 1, 2]
FULL_NODE_COUNTS = [10000, 30000, 50000, 70000, 90000, 110000, 130000, 150000]

# Faster default profile for exploratory runs.
QUICK_SIM_TIME = 1200
QUICK_SEEDS = [0, 1]
QUICK_NODE_COUNTS = list(np.linspace(10000, 150000, 9, dtype=int))

FULL_NODE_COUNTS = list(np.linspace(10000, 150000, 15, dtype=int))

TX_POWER_W = 0.1


def _build_protocols() -> dict:
    base_traffic_params = get_base_traffic_params()
    semantic_params = get_semantic_params()
    return {
        'DR8': {
            'code': '1/3',
            'headers': 3,
            'traffic_class': DistortionAwareExponentialTraffic,
            'traffic_param': base_traffic_params,
        },
        'DR9': {
            'code': '2/3',
            'headers': 2,
            'traffic_class': DistortionAwareExponentialTraffic,
            'traffic_param': base_traffic_params,
        },
        'Semantic': {
            'traffic_class': PrecomputedSemanticTraffic,
            'traffic_param': semantic_params,
        },
    }


def _build_output_dir() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(project_root, 'simulation_results', 'network_sweep', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _energy_efficiency(goodput_bits: float, total_tx_airtime: float) -> float:
    energy_j = TX_POWER_W * total_tx_airtime
    if energy_j <= 0:
        return 0.0
    return goodput_bits / energy_j


def _run_single_case(num_devices: int, protocol: str, cfg: dict, seed: int, sim_time: int) -> dict:
    num_nodes = num_devices // 8

    settings_kwargs = {
        'number_nodes': num_nodes,
        'simulation_time': sim_time,
        'traffic_class': cfg['traffic_class'],
        'traffic_param': dict(cfg['traffic_param']),
    }

    if 'code' in cfg:
        settings_kwargs['code'] = cfg['code']
    if 'headers' in cfg:
        settings_kwargs['headers'] = cfg['headers']

    details = run_sim_detailed(Settings(**settings_kwargs), seed=seed)
    return {
        'devices': num_devices,
        'nodes_simulated': num_nodes,
        'protocol': protocol,
        'seed': seed,
        'success_probability': details['success_ratio'],
        'mean_aoi_s': details['mean_aoi'],
        'mean_semantic_distortion': details['mean_semantic_distortion'],
        'energy_efficiency_bits_per_j': _energy_efficiency(
            details['goodput_bits'], details['total_tx_airtime']
        ),
        'goodput_bits': details['goodput_bits'],
        'total_tx_airtime_s': details['total_tx_airtime'],
        'transmitted_packets': details['transmitted'],
    }


def run_experiment(sim_time: int, seeds: list[int], node_counts: list[int], n_jobs: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    output_dir = _build_output_dir()
    protocols = _build_protocols()

    rows = []
    total = len(node_counts) * len(protocols) * len(seeds)
    idx = 0

    for num_devices in node_counts:
        for protocol, cfg in protocols.items():
            tasks = [(num_devices, protocol, cfg, seed, sim_time) for seed in seeds]

            print(
                f"[{idx + 1}-{idx + len(seeds)}/{total}] "
                f"devices={num_devices:,} protocol={protocol} running {len(seeds)} seeds"
            )

            batch_rows = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(_run_single_case)(*task) for task in tasks
            )
            rows.extend(batch_rows)
            idx += len(seeds)

    raw_df = pd.DataFrame(rows)
    summary_df = (
        raw_df.groupby(['devices', 'protocol'], as_index=False)
        .agg(
            success_probability=('success_probability', 'mean'),
            success_probability_std=('success_probability', 'std'),
            mean_aoi_s=('mean_aoi_s', 'mean'),
            mean_aoi_s_std=('mean_aoi_s', 'std'),
            mean_semantic_distortion=('mean_semantic_distortion', 'mean'),
            mean_semantic_distortion_std=('mean_semantic_distortion', 'std'),
            energy_efficiency_bits_per_j=('energy_efficiency_bits_per_j', 'mean'),
            energy_efficiency_bits_per_j_std=('energy_efficiency_bits_per_j', 'std'),
            goodput_bits=('goodput_bits', 'mean'),
        )
        .sort_values(['devices', 'protocol'])
        .fillna(0.0)
    )

    raw_path = os.path.join(output_dir, 'raw_runs.csv')
    summary_path = os.path.join(output_dir, 'summary_by_protocol.csv')
    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f'[OK] Raw runs saved to: {raw_path}')
    print(f'[OK] Summary saved to: {summary_path}')

    return raw_df, summary_df, output_dir


def _plot_single_metric(summary_df: pd.DataFrame, output_dir: str,
                        metric: str, ylabel: str, filename: str) -> None:
    """Plot one metric vs number of devices as a standalone figure."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    ticks = sorted(summary_df['devices'].unique())

    for protocol in ['DR8', 'DR9', 'Semantic']:
        p_df = summary_df[summary_df['protocol'] == protocol].sort_values('devices')
        ax.plot(p_df['devices'], p_df[metric], **protocol_plot_kwargs(protocol))

        std_col = f'{metric}_std'
        if std_col in p_df.columns:
            y = p_df[metric].to_numpy(dtype=float)
            y_std = p_df[std_col].to_numpy(dtype=float)
            ax.fill_between(
                p_df['devices'], y - y_std, y + y_std,
                color=protocol_plot_kwargs(protocol)['color'], alpha=0.10,
            )

    ax.set_xlabel('Number of Devices')
    ax.set_ylabel(ylabel)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{int(t / 1000)}K' for t in ticks], rotation=30)
    ax.legend(loc='best')
    fig.tight_layout()
    save_fig(fig, output_dir, filename)


def plot_results(summary_df: pd.DataFrame, output_dir: str) -> None:
    """Generate one standalone figure per metric."""
    metrics = [
        ('mean_aoi_s',                    'Average Age of Information (s)',  'aoi_vs_devices'),
        ('mean_semantic_distortion',      'Mean Semantic Distortion',       'distortion_vs_devices'),
        ('energy_efficiency_bits_per_j',  'Energy Efficiency (bits/J)',     'ee_vs_devices'),
        ('success_probability',           'Success Probability',            'success_prob_vs_devices'),
    ]
    for metric, ylabel, filename in metrics:
        _plot_single_metric(summary_df, output_dir, metric, ylabel, filename)


def plot_tradeoff_aoi_vs_energy(summary_df: pd.DataFrame, output_dir: str) -> None:
    """Trade-off scatter: AoI vs Energy Efficiency."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for protocol in ['DR8', 'DR9', 'Semantic']:
        p_df = summary_df[summary_df['protocol'] == protocol].sort_values('devices')
        ax.plot(
            p_df['mean_aoi_s'],
            p_df['energy_efficiency_bits_per_j'],
            **protocol_plot_kwargs(protocol),
        )

        first = p_df.iloc[0]
        last = p_df.iloc[-1]
        clr = protocol_plot_kwargs(protocol)['color']
        ax.annotate(
            f"{int(first['devices'] / 1000)}K",
            (first['mean_aoi_s'], first['energy_efficiency_bits_per_j']),
            textcoords='offset points', xytext=(5, 5), fontsize=8, color=clr,
        )
        ax.annotate(
            f"{int(last['devices'] / 1000)}K",
            (last['mean_aoi_s'], last['energy_efficiency_bits_per_j']),
            textcoords='offset points', xytext=(5, -10), fontsize=8, color=clr,
        )

    ax.set_xlabel('Mean AoI (s)')
    ax.set_ylabel('Energy Efficiency (bits/J)')
    ax.legend(loc='best')
    fig.tight_layout()
    save_fig(fig, output_dir, 'tradeoff_aoi_vs_energy')


def main():
    apply_matlab_style()

    parser = argparse.ArgumentParser(description='Network-size sweep for DR8/DR9/Semantic')
    parser.add_argument('--full', action='store_true', help='Run full profile (slower, higher fidelity)')
    parser.add_argument('--jobs', type=int, default=-1, help='Parallel workers for per-seed simulations')
    args = parser.parse_args()

    if args.full:
        sim_time = FULL_SIM_TIME
        seeds = FULL_SEEDS
        node_counts = FULL_NODE_COUNTS
        profile = 'FULL'
    else:
        sim_time = QUICK_SIM_TIME
        seeds = QUICK_SEEDS
        node_counts = QUICK_NODE_COUNTS
        profile = 'QUICK'

    print('=' * 80)
    print('Network Size Sweep: DR8 vs DR9 vs Semantic')
    print(f'Profile: {profile} | sim_time={sim_time}s | seeds={seeds} | jobs={args.jobs}')
    print('=' * 80)

    _, summary_df, output_dir = run_experiment(sim_time=sim_time, seeds=seeds, node_counts=node_counts, n_jobs=args.jobs)
    plot_results(summary_df, output_dir)
    plot_tradeoff_aoi_vs_energy(summary_df, output_dir)

    print('=' * 80)
    print('[DONE] Simulation finished.')
    print(f'Results folder: {output_dir}')
    print('=' * 80)


if __name__ == '__main__':
    main()
