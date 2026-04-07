from lrfhss.lrfhss_core import *
from lrfhss.settings import Settings
from lrfhss.utils import compute_aoi
import simpy

def run_sim(settings: Settings, seed=0):
    """Backward-compatible summary output used by existing scripts."""
    details = _run_sim_internal(settings, seed=seed)
    return details['summary']


def run_sim_detailed(settings: Settings, seed=0):
    """Extended simulation metrics for post-processing and plotting."""
    return _run_sim_internal(settings, seed=seed)


def _run_sim_internal(settings: Settings, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    env = simpy.Environment()
    bs = Base(settings.obw, settings.threshold)
    
    nodes = []
    
    for i in range(settings.number_nodes):
        # CRITICAL: Each node needs its OWN traffic_generator instance
        # (not shared across nodes).
        # Inject sim_time so threshold-based classes (e.g. PrecomputedSemanticTraffic)
        # can size their AR(1) pre-generation correctly.
        _tp = dict(settings.traffic_generator.traffic_param)
        _tp.setdefault('sim_time', float(settings.simulation_time))
        traffic_gen = settings.traffic_generator.__class__(_tp)
        node = Node(
            settings.obw,
            settings.headers,
            settings.payloads,
            settings.threshold,
            settings.payload_size,
            settings.header_duration,
            settings.payload_duration,
            settings.transceiver_wait,
            traffic_gen,
        )
        bs.add_node(node.id)
        nodes.append(node)
        env.process(node.transmit(env, bs))
    # start simulation
    env.run(until=settings.simulation_time)

    # after simulation

    mean_aoi  = compute_aoi(nodes, settings.simulation_time)
    success    = sum(bs.packets_received.values())
    transmitted = sum(n.transmitted for n in nodes)
    total_tx_airtime = sum(n.tx_airtime for n in nodes)

    distortions = []
    for node in nodes:
        tg = node.traffic_generator
        # Use time-averaged distortion: D̄ = (1/T) ∫ |x(t) - x̂(t)| dt
        # This is the fair comparison metric across all protocols.
        if hasattr(tg, 'get_time_averaged_distortion'):
            d = tg.get_time_averaged_distortion()
            if not np.isnan(d):
                distortions.append(float(d))
        elif hasattr(tg, 'get_average_distortion'):
            d = tg.get_average_distortion()
            if not np.isnan(d):
                distortions.append(float(d))
    mean_semantic_distortion = float(np.mean(distortions)) if distortions else float('nan')

    # Collect per-setup usage statistics (semantic config distribution)
    from collections import Counter
    setup_counter = Counter()
    for node in nodes:
        for h, code in node.tx_config_log:
            setup_counter[(h, code)] += 1
    total_configs = sum(setup_counter.values())
    setup_usage = {
        f'h={h}_cr={code}': {'count': cnt, 'pct': 100.0 * cnt / total_configs if total_configs > 0 else 0.0}
        for (h, code), cnt in sorted(setup_counter.items(), key=lambda x: -x[1])
    }

    if transmitted == 0:
        summary = [[1.0], [0], [0], [mean_aoi], [0.0]]
        return {
            'summary': summary,
            'success_ratio': 1.0,
            'goodput_bits': 0.0,
            'transmitted': 0,
            'mean_aoi': float(mean_aoi),
            'total_tx_airtime': 0.0,
            'mean_semantic_distortion': mean_semantic_distortion,
            'success_count': int(success),
            'setup_usage': setup_usage,
        }

    success_ratio = success / transmitted
    goodput_bits = success * settings.payload_size * 8
    summary = [
        [success / transmitted],          # PDR
        [success * settings.payload_size * 8], # goodput (bits)
        [transmitted],
        [mean_aoi],
        [total_tx_airtime],              # total packet airtime (s)
    ]
    return {
        'summary': summary,
        'success_ratio': float(success_ratio),
        'goodput_bits': float(goodput_bits),
        'transmitted': int(transmitted),
        'mean_aoi': float(mean_aoi),
        'total_tx_airtime': float(total_tx_airtime),
        'mean_semantic_distortion': mean_semantic_distortion,
        'success_count': int(success),
        'setup_usage': setup_usage,
    }

    #Get the average success per device, used to plot the CDF 
    #success_per_device = [1 if n.transmitted == 0 else bs.packets_received[n.id]/n.transmitted for n in nodes]
    #return success_per_device
if __name__ == "__main__":
   s = Settings()
   print(run_sim(s))