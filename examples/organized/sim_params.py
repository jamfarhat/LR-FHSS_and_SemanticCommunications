"""Shared simulation parameters used by organized example scripts."""

from __future__ import annotations


def get_base_traffic_params() -> dict:
    return {
        'average_interval': 300,
        'alpha': 0.95,
        'sigma_w': 1.0,
        'lambda_ref': 300,   # fixed reference epoch for AR(1) alpha scaling
    }


def get_semantic_params() -> dict:
    params = get_base_traffic_params()
    params.update(
        {
            'epsilon_0': 1.25,
            'epsilon_min': 0.20,
            'beta': 0.001,
            'semantic_configs': [
                {'max_distortion': 0.85, 'headers': 1, 'code': '5/6'},   # low-distortion crossing (AoI-driven, threshold decayed)
                {'max_distortion': 1.10, 'headers': 2, 'code': '2/3'},   # normal threshold crossing
                {'max_distortion': float('inf'), 'headers': 3, 'code': '1/2'},  # high-distortion crossing, need reliability
            ],
        }
    )
    return params
