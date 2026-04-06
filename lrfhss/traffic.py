import random
import numpy as np
from lrfhss.lrfhss_core import Traffic
import warnings

# Optional — imported lazily to avoid circular imports in simple use cases.
_AR1Process = None

def _get_ar1_class():
    global _AR1Process
    if _AR1Process is None:
        from lrfhss.ar1_process import AR1Process
        _AR1Process = AR1Process
    return _AR1Process


class Exponential_Traffic(Traffic):
    """Exponential inter-arrival traffic."""
    def __init__(self, traffic_param):
        super().__init__(traffic_param)
        if 'average_interval' not in self.traffic_param:
            warnings.warn('Exponential_Traffic: average_interval missing. Using default 900s')
            self.traffic_param['average_interval'] = 900

    def traffic_function(self):
        return random.expovariate(1 / self.traffic_param['average_interval'])


class Uniform_Traffic(Traffic):
    """Uniform inter-arrival traffic."""
    def __init__(self, traffic_param):
        super().__init__(traffic_param)
        if 'max_interval' not in self.traffic_param:
            warnings.warn('Uniform_Traffic: max_interval missing. Using default 1800s')
            self.traffic_param['max_interval'] = 1800

    def traffic_function(self):
        return random.uniform(0, self.traffic_param['max_interval'])


class Constant_Traffic(Traffic):
    """Constant inter-arrival traffic with Gaussian deviation."""
    def __init__(self, traffic_param):
        super().__init__(traffic_param)
        if 'constant_interval' not in self.traffic_param:
            warnings.warn('Constant_Traffic: constant_interval missing. Using default 900s')
            self.traffic_param['constant_interval'] = 900
        if 'standard_deviation' not in self.traffic_param:
            warnings.warn('Constant_Traffic: standard_deviation missing. Using default 10')
            self.traffic_param['standard_deviation'] = 10

    def traffic_function(self):
        if self.transmitted == 0:
            return random.uniform(0, 2 * self.traffic_param['constant_interval'])
        return max(
            0,
            self.traffic_param['constant_interval']
            + random.gauss(0, self.traffic_param['standard_deviation'])
        )


class DistortionAwareExponentialTraffic(Traffic):
    """
    Traditional exponential traffic with AR(1)-based distortion monitoring.

    This class keeps the same transmission behavior as Exponential_Traffic
    (always transmit on each epoch) and adds semantic distortion tracking so
    DR8/DR9 can be fairly compared against semantic scheduling.
    """

    def __init__(self, traffic_param):
        super().__init__(traffic_param)
        if 'average_interval' not in self.traffic_param:
            warnings.warn('DistortionAwareExponentialTraffic: average_interval missing. Using default 900s')
            self.traffic_param['average_interval'] = 900.0

        self.lambda_interval = self.traffic_param['average_interval']
        self.alpha = self.traffic_param.get('alpha', 0.95)
        self.sigma_w = self.traffic_param.get('sigma_w', 1.0)
        self.track_trace = bool(self.traffic_param.get('track_trace', False))

        # Optional pre-generated AR(1) process (shared across protocols).
        # If provided, x_current is obtained via process.query(t) instead of
        # the internal incremental update.
        self._ar1_process = traffic_param.get('ar1_process', None)

        self.x_current = np.random.normal(0, self.sigma_w)
        self.x_last_tx = np.random.normal(0, self.sigma_w)
        self.aoi_local = np.random.uniform(0, self.lambda_interval)
        self._time = 0.0

        self._distortion_accum = 0.0
        self._decision_count = 0
        self._trace = []

    def traffic_function(self):
        return random.expovariate(1 / self.lambda_interval)

    def update_ar1(self, dt):
        if self._ar1_process is not None:
            # Use pre-generated process — just advance the clock and query.
            self.x_current = self._ar1_process.query(self._time)
            return
        if self.lambda_interval <= 0:
            dt_ratio = 1.0
        else:
            dt_ratio = max(1e-9, dt / self.lambda_interval)
        alpha_eff = self.alpha ** dt_ratio
        sigma_eff = self.sigma_w * np.sqrt(dt_ratio)
        self.x_current = alpha_eff * self.x_current + np.random.normal(0, sigma_eff)

    def get_distortion(self):
        return abs(self.x_current - self.x_last_tx)

    def on_decision_epoch(self, dt):
        self._time += dt
        self.update_ar1(dt)
        self.aoi_local += dt

        distortion = self.get_distortion()
        self._distortion_accum += distortion
        self._decision_count += 1

        # Traditional policy always transmits at each scheduled epoch.
        x_last_before_tx = self.x_last_tx
        self.x_last_tx = self.x_current
        self.aoi_local = 0.0

        if self.track_trace:
            self._trace.append({
                'time':        self._time,
                'x_current':   float(self.x_current),
                'x_hat':       float(x_last_before_tx),
                'distortion':  distortion,
                'threshold':   np.nan,
                'tx_decision': True,
            })

    def should_send_now(self):
        return True

    def get_average_distortion(self):
        if self._decision_count == 0:
            return np.nan
        return self._distortion_accum / self._decision_count

    def get_trace(self):
        return list(self._trace)


## Semantic-Driven Traffic with AR(1) Process Model
class SemanticTraffic(Traffic):
    r"""
    Feedback-free semantic transmission scheduling based on AR(1) process monitoring.
    
    FAIR COMPARISON WITH TRADITIONAL TRAFFIC:
    =============================================
    Both use the SAME inter-packet generation interval λ (average_interval parameter).
    
    Traditional: Always transmits packets at λ-intervals
    Semantic:    Generates packets at λ-intervals BUT decides WHETHER to transmit
                 based on semantic relevance check.
    
    PROCESS MODEL (Equation 1):
    ===========================
    Each device monitors a physical process via AR(1):
        x_k(t) = α*x_k(t-1) + w_k(t)
    
    where:
    - α ∈ (0,1) is autocorrelation (0.95 = slow variation like temperature)
    - w_k(t) ~ N(0, σ_w²) is process noise
    
    SEMANTIC DISTORTION (Equation 7):
    ==================================
    Device estimates what gateway believes about its state:
        D_k(t) = |x_k(t) - x̂_k(t)|
    
    where x̂_k(t) = x_k(t - Δ_local) is the last transmitted value
    (optimistic assumption it was successfully decoded)
    
    TRANSMISSION DECISION (Equations 9-10):
    =======================================
    At each λ-interval:
        a_k(t) = 1 if D_k(t) ≥ ε_th(Δ_local), else 0
    
    where threshold adapts with Age of Information:
        ε_th(Δ) = max(ε_min, ε_0 - β*Δ)
    
    - Small Δ: high threshold (recent tx, need big change to send again)
    - Large Δ: low threshold (old tx, force update to prevent staleness)
    - β=0: fixed threshold
    - β>0: decreases threshold over time (force periodic updates)
    
    When a_k(t)=1: transmit packet and reset Δ_local=0
    When a_k(t)=0: skip transmission, continue accumulating Δ_local
    """
    
    def __init__(self, traffic_param):
        super().__init__(traffic_param)
        
        # AR(1) process parameters
        self.alpha = traffic_param.get('alpha', 0.95)
        self.sigma_w = traffic_param.get('sigma_w', 1.0)
        
        # Semantic threshold parameters
        # Backward-compatible alias: some scripts still use threshold_0.
        self.epsilon_0 = traffic_param.get('epsilon_0', traffic_param.get('threshold_0', 1.0))
        self.epsilon_min = traffic_param.get('epsilon_min', 0.1)
        self.beta = traffic_param.get('beta', 0.01)
        
        # CRITICAL for fair comparison: use same interval as traditional traffic
        if 'average_interval' not in traffic_param:
            warnings.warn('SemanticTraffic: average_interval missing. Using default 900s')
            traffic_param['average_interval'] = 900.0
        self.lambda_interval = traffic_param['average_interval']
        self.track_trace = bool(traffic_param.get('track_trace', False))

        # Optional pre-generated AR(1) process (shared across protocols).
        self._ar1_process = traffic_param.get('ar1_process', None)

        # Optional semantic-to-robustness mapping.
        # Each entry is ordered by increasing max_distortion and may specify
        # headers and code. Example:
        # [
        #   {'max_distortion': 0.5, 'headers': 1, 'code': '5/6'},
        #   {'max_distortion': 1.5, 'headers': 2, 'code': '2/3'},
        #   {'max_distortion': float('inf'), 'headers': 3, 'code': '1/3'},
        # ]
        default_configs = [
            {'max_distortion': self.epsilon_0, 'headers': 1, 'code': '5/6'},
            {'max_distortion': 2 * self.epsilon_0, 'headers': 2, 'code': '2/3'},
            {'max_distortion': float('inf'), 'headers': 3, 'code': '1/3'},
        ]
        self.semantic_configs = sorted(
            traffic_param.get('semantic_configs', default_configs),
            key=lambda c: c.get('max_distortion', float('inf')),
        )
        
        # AR(1) process state
        # Initialize with random values to break synchronization between nodes
        self.x_current = np.random.normal(0, self.sigma_w)
        self.x_last_tx = np.random.normal(0, self.sigma_w)  # Random initial estimate
        self.aoi_local = np.random.uniform(0, self.lambda_interval)  # Random initial AoI
        
        # Transmission approval flag (set by traffic_function, checked by Node)
        self._approve_transmission = True
        self._last_distortion = 0.0
        self._last_threshold = self.epsilon_0
        self._last_aoi_before = self.aoi_local
        self._last_aoi_after = self.aoi_local
        self._last_tx_decision = False
        self._tx_config = None
        self._time = 0.0
        self._distortion_accum = 0.0
        self._decision_count = 0
        self._trace = []
        
        # Tracking for avoiding multiple updates in same λ-period
        # Set to True when traffic_function() is called, False when Node transmits/skips
        self._called_this_period = False
    
    def update_ar1(self, dt):
        """Evolve AR(1) process over elapsed time dt."""
        if self._ar1_process is not None:
            self.x_current = self._ar1_process.query(self._time)
            return
        # Internal incremental update (backward-compatible).
        if self.lambda_interval <= 0:
            dt_ratio = 1.0
        else:
            dt_ratio = max(1e-9, dt / self.lambda_interval)
        alpha_eff = self.alpha ** dt_ratio
        sigma_eff = self.sigma_w * np.sqrt(dt_ratio)
        w = np.random.normal(0, sigma_eff)
        self.x_current = alpha_eff * self.x_current + w
    
    def get_distortion(self):
        """Equation (7): D_k(t) = |x_k(t) - x̂_k(t)|"""
        return abs(self.x_current - self.x_last_tx)
    
    def get_threshold(self, aoi_local):
        """Equation (10): ε_th(Δ) = max(ε_min, ε_0 - β*Δ)"""
        return max(self.epsilon_min, self.epsilon_0 - self.beta * aoi_local)
    
    def should_transmit(self):
        """Equation (9): Decision rule a_k(t)"""
        distortion = self.get_distortion()
        threshold = self.get_threshold(self.aoi_local)
        return distortion >= threshold
    
    def should_send_now(self):
        """
        Called by Node.transmit() to decide if current timeout should result in transmission.
        Returns the flag set by the last traffic_function() call.
        """
        return self._approve_transmission

    def _select_semantic_config(self, distortion):
        """Select semantic robustness configuration based on distortion."""
        if not self.semantic_configs:
            return None
        for cfg in self.semantic_configs:
            if distortion < cfg.get('max_distortion', float('inf')):
                return cfg
        return self.semantic_configs[-1]

    def get_tx_params(self):
        """Return the selected transmission config for current epoch."""
        return self._tx_config

    def on_decision_epoch(self, dt):
        """Update process/AoI and evaluate semantic decision at the current epoch."""
        self._time += dt
        self.update_ar1(dt=dt)
        self.aoi_local += dt
        self._last_aoi_before = self.aoi_local

        distortion = self.get_distortion()
        self._distortion_accum += distortion
        self._decision_count += 1
        self._last_distortion = distortion
        threshold = self.get_threshold(self.aoi_local)
        self._last_threshold = threshold

        if distortion >= threshold:
            self._approve_transmission = True
            self._last_tx_decision = True
            self._tx_config = self._select_semantic_config(distortion)
            self.x_last_tx = self.x_current
            self.aoi_local = 0.0
        else:
            self._approve_transmission = False
            self._last_tx_decision = False
            self._tx_config = None
        self._last_aoi_after = self.aoi_local

        if self.track_trace:
            self._trace.append({
                'time':        self._time,
                'x_current':   float(self.x_current),
                'x_hat':       float(self.x_last_tx),
                'distortion':  distortion,
                'threshold':   threshold,
                'tx_decision': self._last_tx_decision,
            })

    def get_average_distortion(self):
        if self._decision_count == 0:
            return np.nan
        return self._distortion_accum / self._decision_count

    def get_trace(self):
        return list(self._trace)
    
    def traffic_function(self):
        # Return the next decision epoch. Semantic state update happens when that
        # epoch is reached inside Node.transmit().
        return random.expovariate(1 / self.lambda_interval)