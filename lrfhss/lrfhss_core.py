import random
import numpy as np
from abc import ABC, abstractmethod


class Fragment():
    def __init__(self, type, duration, channel, packet):
        self.packet = packet
        self.duration = duration
        self.success = 0
        self.transmitted = 0
        self.type = type
        self.channel = channel
        self.timestamp = 0
        self.id = id(self)
        self.collided = []

class Packet():
    def __init__(self, node_id, obw, headers, payloads, header_duration, payload_duration, threshold=None):
        self.id = id(self)
        self.node_id = node_id
        self.index_transmission = 0
        self.success = 0
        self.threshold = threshold
        self.channels = random.choices(range(obw), k=headers+payloads)
        self.fragments = []

        for h in range(headers):
            self.fragments.append(Fragment('header',header_duration, self.channels[h], self.id))
        for p in range(payloads):
            self.fragments.append(Fragment('payload',payload_duration, self.channels[p+h+1], self.id))

    def next(self):
        self.index_transmission += 1
        try:
            return self.fragments[self.index_transmission - 1]
        except IndexError:
            return False


class Traffic(ABC):
    @abstractmethod
    def __init__(self, traffic_param):
        self.traffic_param = traffic_param

    @abstractmethod
    def traffic_function(self):
        pass

class Node():
    def __init__(self, obw, headers, payloads, threshold, payload_size, header_duration, payload_duration, transceiver_wait, traffic_generator):
        self.id = id(self)
        self.transmitted = 0
        self.traffic_generator = traffic_generator
        self.transceiver_wait = transceiver_wait
        # Packet info that Node has to store
        self.obw = obw
        self.default_headers = headers
        self.default_payloads = payloads
        self.default_threshold = threshold
        self.payload_size = payload_size
        self.header_duration = header_duration
        self.payload_duration = payload_duration
        self.initial_timestamp = []
        self.final_timestamp = []
        self.tx_airtime = 0.0
        self.headers = headers
        self.payloads = payloads
        self.threshold = threshold
        self.tx_config_log = []  # list of (headers, code) per transmitted packet
        
        self.packet = Packet(self.id, self.obw, self.headers, self.payloads, self.header_duration, self.payload_duration, self.threshold)

    def _code_to_payload_threshold(self, code):
        if code == '1/3':
            payloads = np.ceil((self.payload_size + 3) / 2).astype('int')
            threshold = np.ceil(payloads / 3).astype('int')
        elif code == '2/3':
            payloads = np.ceil((self.payload_size + 3) / 4).astype('int')
            threshold = np.ceil((2 * payloads) / 3).astype('int')
        elif code == '5/6':
            payloads = np.ceil((self.payload_size + 3) / 5).astype('int')
            threshold = np.ceil((5 * payloads) / 6).astype('int')
        elif code == '1/2':
            payloads = np.ceil((self.payload_size + 3) / 3).astype('int')
            threshold = np.ceil(payloads / 2).astype('int')
        else:
            payloads = self.default_payloads
            threshold = self.default_threshold
        return int(payloads), int(threshold)

    def _apply_semantic_tx_config(self):
        self.headers = self.default_headers
        self.payloads = self.default_payloads
        self.threshold = self.default_threshold

        if not hasattr(self.traffic_generator, 'get_tx_params'):
            return

        cfg = self.traffic_generator.get_tx_params()
        if not cfg:
            return

        if 'headers' in cfg:
            self.headers = int(cfg['headers'])

        if 'code' in cfg:
            self.payloads, self.threshold = self._code_to_payload_threshold(cfg['code'])

    def next_transmission(self):
        return self.traffic_generator.traffic_function()

    def end_of_transmission(self):
        self.packet = Packet(self.id, self.obw, self.headers, self.payloads, self.header_duration, self.payload_duration, self.threshold)

    def transmit(self, env, bs):
        while True:
            wait_time = self.next_transmission()
            yield env.timeout(wait_time)

            if hasattr(self.traffic_generator, 'on_decision_epoch'):
                self.traffic_generator.on_decision_epoch(wait_time)

            if hasattr(self.traffic_generator, 'should_send_now'):
                if not self.traffic_generator.should_send_now():
                    continue

            self.transmitted += 1
            self._apply_semantic_tx_config()
            self.tx_airtime += self.headers * self.header_duration + self.payloads * self.payload_duration

            # Log the configuration used for this packet
            cfg = self.traffic_generator.get_tx_params() if hasattr(self.traffic_generator, 'get_tx_params') else None
            code_used = cfg.get('code', None) if cfg else None
            self.tx_config_log.append((self.headers, code_used))

            self.packet = Packet(
                self.id, self.obw, self.headers, self.payloads,
                self.header_duration, self.payload_duration, self.threshold
            )
            bs.add_packet(self.packet)
            next_fragment = self.packet.next()
            first_payload = 0
            self.initial_timestamp.append(env.now)

            while next_fragment:
                if first_payload == 0 and next_fragment.type == 'payload':
                    first_payload = 1
                    yield env.timeout(self.transceiver_wait)
                next_fragment.timestamp = env.now
                bs.check_collision(next_fragment)
                bs.receive_packet(next_fragment)
                yield env.timeout(next_fragment.duration)
                bs.finish_fragment(next_fragment)
                if self.packet.success == 0:
                    bs.try_decode(self, self.packet, env.now)
                next_fragment = self.packet.next()

            if self.packet.success == 0:
                self.final_timestamp.append(0)

            self.end_of_transmission()

class Base():
    def __init__(self, obw, threshold):
        self.id = id(self)
        self.transmitting = {}
        for channel in range(obw):
            self.transmitting[channel] = []
        self.packets_received = {}
        self.threshold = threshold

    def add_packet(self, packet):
        pass

    def add_node(self, id):
        self.packets_received[id] = 0

    def receive_packet(self, fragment):
        self.transmitting[fragment.channel].append(fragment)

    def finish_fragment(self, fragment):
        self.transmitting[fragment.channel].remove(fragment)
        if len(fragment.collided) == 0:
            fragment.success = 1
        fragment.transmitted = 1

    def check_collision(self, fragment):
        for f in self.transmitting[fragment.channel]:
            f.collided.append(fragment)
            fragment.collided.append(f)

    def try_decode(self, node, packet, now):
        h_success = sum(
            1 if ((len(f.collided) == 0) and f.transmitted == 1 and f.type == 'header') else 0
            for f in packet.fragments
        )
        p_success = sum(
            1 if ((len(f.collided) == 0) and f.transmitted == 1 and f.type == 'payload') else 0
            for f in packet.fragments
        )
        required_payloads = packet.threshold if packet.threshold is not None else self.threshold
        success = 1 if ((h_success > 0) and (p_success >= required_payloads)) else 0
        if success == 1:
            self.packets_received[packet.node_id] += 1
            packet.success = 1
            node.final_timestamp.append(now)
            return True
        return False