from lrfhss.lrfhss_core import *
from lrfhss.settings import Settings
from lrfhss.utils import compute_aoi
import simpy

def run_sim(settings: Settings, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    env = simpy.Environment()
    bs = Base(settings.obw, settings.threshold)
    
    nodes = []
    
    for i in range(settings.number_nodes):
        # CRITICAL: Each node needs its OWN traffic_generator instance
        # (not shared across nodes)
        traffic_gen = settings.traffic_generator.__class__(settings.traffic_generator.traffic_param)
        node = Node(settings.obw, settings.headers, settings.payloads, settings.header_duration, settings.payload_duration, settings.transceiver_wait, traffic_gen)
        bs.add_node(node.id)
        nodes.append(node)
        env.process(node.transmit(env, bs))
    # start simulation
    env.run(until=settings.simulation_time)

    # after simulation

    mean_aoi  = compute_aoi(nodes, settings.simulation_time)
    success    = sum(bs.packets_received.values())
    transmitted = sum(n.transmitted for n in nodes)

    if transmitted == 0:
        return [[1.0], [0], [0], [mean_aoi]]

    return [
        [success / transmitted],          # PDR
        [success * settings.payload_size], # goodput (bytes)
        [transmitted],
        [mean_aoi],
    ]

    #Get the average success per device, used to plot the CDF 
    #success_per_device = [1 if n.transmitted == 0 else bs.packets_received[n.id]/n.transmitted for n in nodes]
    #return success_per_device
if __name__ == "__main__":
   s = Settings()
   print(run_sim(s))