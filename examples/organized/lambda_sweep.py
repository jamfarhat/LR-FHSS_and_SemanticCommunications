#!/usr/bin/env python3
"""
Lambda (average_interval) Sweep with Fixed Network Size.

Fixes the number of devices (default 60K) and varies the average inter-packet
interval λ for DR8, DR9 and Semantic, plotting:
  - Goodput (bits)
  - Energy Efficiency (bits/J)
  - Mean AoI (s)
  - Mean Semantic Distortion
  - Success Probability

Outputs (in simulation_results/lambda_sweep/<timestamp>/):
  - raw_runs.csv, summary.csv
  - lambda_sweep_metrics.{png,eps,jpg}
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lrfhss.run import run_sim_detailed
from lrfhss.settings import Settings
from lrfhss.traffic import DistortionAwareExponentialTraffic, SemanticTraffic
from examples.organized.sim_params import get_base_traffic_params, get_semantic_params
from examples.organized.plot_style import apply_matlab_style, save_fig, protocol_plot_kwargs

DEFAULT_DEVICES = 60000
SIM_TIME = 3600
SEEDS = [0, 1, 2]
TX_POWER_W = 0.1

LAMBDA_VALUES = [60, 120, 180, 300, 450, 600, 900, 1200, 1800]


def _build_output_dir() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(project_root, 'simulation_results', 'lambda_sweep', timestamp)
    os.makedirs(out, exist_ok=True)
    return out


def _energy_efficiency(goodput_bits: float, total_tx_airtime: float) -> float:
    energy_j = TX_POWER_W * total_tx_airtime
    return goodput_bits / energy_j if energy_j > 0 else 0.0


def _run_single(lam: float, protocol: str, cfg: dict,
                num_devices: int, seed: int, sim_time: int) -> dict:
    num_nodes = num_devices // 8

    tp = dict(cfg['traffic_param'])
    tp['average_interval'] = lam

    kw = {
        'number_nodes': num_nodes,
        'simulation_time': sim_time,
        'traffic_class': cfg['traffic_class'],
        'traffic_param': tp,
    }
    if 'code' in cfg:
        kw['code'] = cfg['code']
    if 'headers' in cfg:
        kw['headers'] = cfg['headers']

    det = run_sim_detailed(Settings(**kw), seed=seed)
    return {
        'lambda_s': lam,
        'protocol': protocol,
        'seed': seed,
        'success_probability': det['success_ratio'],
        'mean_aoi_s': det['mean_aoi'],
        'mean_semantic_distortion': det['mean_semantic_distortion'],
        'energy_efficiency': _energy_efficiency(det['goodput_bits'], det['total_tx_airtime']),
        'goodput_bits': det['goodput_bits'],
        'transmitted': det['transmitted'],
    }



def main():
    apply_matlab_style()

    parser = argparse.ArgumentParser(description='Lambda sweep for DR8/DR9/Semantic')
    parser.add_argument('--devices', type=int, default=DEFAULT_DEVICES)
    parser.add_argument('--jobs', type=int, default=-1)
    parser.add_argument('--quick', action='store_true', help='Use fewer lambda values and seeds')
    args = parser.parse_args()

    num_devices = args.devices
    lambdas = [120, 300, 600, 1200] if args.quick else LAMBDA_VALUES
    seeds = [0, 1] if args.quick else SEEDS
    sim_time = 1200 if args.quick else SIM_TIME

    output_dir = _build_output_dir()

    base_tp = get_base_traffic_params()
    sem_tp = get_semantic_params()
    protocols = {
        'DR8': {'code': '1/3', 'headers': 3,
                'traffic_class': DistortionAwareExponentialTraffic, 'traffic_param': base_tp},
        'DR9': {'code': '2/3', 'headers': 2,
                'traffic_class': DistortionAwareExponentialTraffic, 'traffic_param': base_tp},
        'Semantic': {'traffic_class': SemanticTraffic, 'traffic_param': sem_tp},
    }

    print('=' * 80)
    print(f'Lambda Sweep | devices={num_devices:,} | lambdas={lambdas} | seeds={seeds}')
    print('=' * 80)

    rows = []
    total = len(lambdas) * len(protocols) * len(seeds)
    idx = 0
    for lam in lambdas:
        for pname, cfg in protocols.items():
            print(
                f'[{idx + 1}-{idx + len(seeds)}/{total}] '
                f'lambda={lam}s protocol={pname}'
            )
            batch = Parallel(n_jobs=args.jobs, backend='loky')(
                delayed(_run_single)(lam, pname, cfg, num_devices, s, sim_time)
                for s in seeds
            )
            rows.extend(batch)
            idx += len(seeds)

    raw_df = pd.DataFrame(rows)
    summary_df = (
        raw_df.groupby(['lambda_s', 'protocol'], as_index=False)
        .agg(
            success_probability=('success_probability', 'mean'),
            success_probability_std=('success_probability', 'std'),
            mean_aoi_s=('mean_aoi_s', 'mean'),
            mean_aoi_s_std=('mean_aoi_s', 'std'),
            mean_semantic_distortion=('mean_semantic_distortion', 'mean'),
            mean_semantic_distortion_std=('mean_semantic_distortion', 'std'),
            energy_efficiency=('energy_efficiency', 'mean'),
            energy_efficiency_std=('energy_efficiency', 'std'),
            goodput_bits=('goodput_bits', 'mean'),
        )
        .sort_values(['lambda_s', 'protocol'])
        .fillna(0.0)
    )

    raw_df.to_csv(os.path.join(output_dir, 'raw_runs.csv'), index=False)
    summary_df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)
    print(f'\n[OK] CSVs saved to {output_dir}')

    # ── Plot (one figure per metric) ───────────────────────────────────
    metric_cfg = [
        ('mean_aoi_s',              'Mean AoI (s)',               'lambda_aoi'),
        ('mean_semantic_distortion','Mean Semantic Distortion',   'lambda_distortion'),
        ('energy_efficiency',       'Energy Efficiency (bits/J)', 'lambda_ee'),
        ('success_probability',     'Success Probability',        'lambda_success_prob'),
    ]

    for col, ylabel, fname in metric_cfg:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for pname in ['DR8', 'DR9', 'Semantic']:
            df_p = summary_df[summary_df['protocol'] == pname].sort_values('lambda_s')
            ax.plot(df_p['lambda_s'], df_p[col], **protocol_plot_kwargs(pname))
            std_col = f'{col}_std'
            if std_col in df_p.columns:
                y = df_p[col].to_numpy(dtype=float)
                ys = df_p[std_col].to_numpy(dtype=float)
                ax.fill_between(df_p['lambda_s'], y - ys, y + ys,
                                color=protocol_plot_kwargs(pname)['color'], alpha=0.10)

        ax.set_xlabel(r'Average interval $\lambda$ (s)')
        ax.set_ylabel(ylabel)
        ax.legend(loc='best')
        fig.tight_layout()
        save_fig(fig, output_dir, fname)

    print('\n' + '=' * 80)
    print('[DONE] Lambda sweep finished.')
    print(f'Results folder: {output_dir}')
    print('=' * 80)


if __name__ == '__main__':
    main()
