"""
utils.py — shared helpers used by both run.py and run_semantic.py.
"""

import numpy as np


def compute_aoi(nodes, simulation_time: float) -> float:
    """
    Time-average AoI via the geometric-decomposition formula (Eq. 3-5).

        Δ̄_k = (1/T) ∫₀ᵀ Δ_k(t) dt

    Each inter-reception interval contributes a trapezoidal area Q_n:

        Q_n = Y_n · (t'_{n-1} − t*_{n-1}) + Y_n² / 2          (Eq. 5)

    where
        Y_n  = t'_n − t'_{n-1}   (inter-reception gap)
        t'_n = successful reception time
        t*_n = generation (initial_timestamp) of packet n

    Parameters
    ----------
    nodes : list of Node / SemanticNode
        Each node must expose .initial_timestamp and .final_timestamp lists
        of equal length.  final_timestamp[i] == 0 means packet i was lost.
    simulation_time : float

    Returns
    -------
    float — network mean AoI (averaged over all nodes)
    """
    aoi_per_node = []

    for n in nodes:
        total_area        = 0.0
        last_success_time = 0.0
        last_gen_time     = 0.0
        had_success       = False

        for gen_t, recv_t in zip(n.initial_timestamp, n.final_timestamp):
            if recv_t == 0:
                continue                            # packet lost — skip

            had_success = True
            Y_n       = recv_t - last_success_time
            age_start = last_success_time - last_gen_time  # Δ at start of interval

            # Trapezoidal area: Q_n = Y_n·age_start + Y_n²/2
            total_area += Y_n * age_start + (Y_n ** 2) / 2.0

            last_success_time = recv_t
            last_gen_time     = gen_t

        # Tail: from last reception to end of simulation
        if had_success:
            Y_tail    = simulation_time - last_success_time
            age_start = last_success_time - last_gen_time
            total_area += Y_tail * age_start + (Y_tail ** 2) / 2.0
        else:
            # No successful reception: AoI grew from 0 throughout → triangle
            total_area = (simulation_time ** 2) / 2.0

        aoi_per_node.append(total_area / simulation_time)

    return float(np.mean(aoi_per_node)) if aoi_per_node else simulation_time / 2.0
