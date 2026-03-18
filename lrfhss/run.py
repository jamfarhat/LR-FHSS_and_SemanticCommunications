from lrfhss.lrfhss_core import *
from lrfhss.acrda import BaseACRDA
from lrfhss.settings import Settings
import simpy

def run_sim(settings: Settings, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    env = simpy.Environment()
    bs = Base(settings.obw, settings.threshold)
    
    nodes = []
    
    for i in range(settings.number_nodes):
        node = Node(settings.obw, settings.headers, settings.payloads, settings.header_duration, settings.payload_duration, settings.transceiver_wait, settings.traffic_generator)
        bs.add_node(node.id)
        nodes.append(node)
        env.process(node.transmit(env, bs))
    # start simulation
    env.run(until=settings.simulation_time)

    # after simulation

    ######## Cálculo da AoI ########  
    aoi = [] 
    
    for n in nodes: 
        last_reception = 0.0
        last_generation = 0.0
        total_area = 0.0
        loop=0
        has_success = False  # Check if any packet was received

        for p in n.final_timestamp:
            #H_i_num = 0
            if p != 0:  # Successful reception
                has_success = True
                current_generation_time = n.initial_timestamp[loop]

                if last_reception == 0.0:  # First reception
                    # AoI from 0 to p is a triangle (no prior packets)
                    time_interval = p - 0
                    total_area += (0.5 * (time_interval ** 2))/time_interval
                else:
                    # AoI from last_reception_time to p
                    time_interval = p - last_reception
                    triangle_area = 0.5 * (time_interval ** 2)
                    rectangle_height = last_reception - last_generation if last_reception > 0 else 0.0
                    rectangle_area = rectangle_height * time_interval
                    total_area += (triangle_area + rectangle_area)/time_interval
                
                last_reception = p
                last_generation = current_generation_time
            loop=loop+1

        if has_success:
            if p==0: # Cálculo do AoI até o final da simulação se o último pacote não foi decodificado.
                time_interval = settings.simulation_time - last_reception
                triangle_area = 0.5 * (time_interval ** 2)
                rectangle_height = last_reception - last_generation
                rectangle_area = rectangle_height * time_interval
                total_area += (triangle_area + rectangle_area)/time_interval
        else:
            # Nenhum pacote recebido durante toda simulação : AoI is a triangle from 0 to settings.simulation_time
            total_area = 0.5 * (settings.simulation_time ** 2)/settings.simulation_time

        aoi.append(total_area/settings.simulation_time)

    AoI_media=np.mean(aoi)
    
    ######## Cálculo da AoI ######## 

    success = sum(bs.packets_received.values())
    transmitted = sum(n.transmitted for n in nodes)

    if transmitted == 0: #If no transmissions are made, we consider 100% success as there were no outages
        return 1
    else:
        
        #sucess rate, goodput , transmitidos, aoi media
        return [[success/transmitted], [success*settings.payload_size], [transmitted], [AoI_media]]

    #Get the average success per device, used to plot the CDF 
    #success_per_device = [1 if n.transmitted == 0 else bs.packets_received[n.id]/n.transmitted for n in nodes]
    #return success_per_device
if __name__ == "__main__":
   s = Settings()
   print(run_sim(s))