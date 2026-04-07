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
        # lambda_ref: fixed reference epoch for AR(1) alpha scaling.
        # Must NOT change with average_interval so the process stays the same.
        self.lambda_ref = self.traffic_param.get('lambda_ref', self.lambda_interval)
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
        
        # Time-integrated distortion for fair comparison
        self._distortion_time_integral = 0.0
        self._last_distortion_value = 0.0

    def traffic_function(self):
        return random.expovariate(1 / self.lambda_interval)

    def update_ar1(self, dt):
        if self._ar1_process is not None:
            # Use pre-generated process — just advance the clock and query.
            self.x_current = self._ar1_process.query(self._time)
            return
        if self.lambda_ref <= 0:
            dt_ratio = 1.0
        else:
            dt_ratio = max(1e-9, dt / self.lambda_ref)
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
        # Time-integrated distortion (trapezoidal rule)
        self._distortion_time_integral += 0.5 * (self._last_distortion_value + distortion) * dt
        self._last_distortion_value = distortion
        
        self._distortion_accum += distortion
        self._decision_count += 1

        # Traditional policy always transmits at each scheduled epoch.
        x_last_before_tx = self.x_last_tx
        self.x_last_tx = self.x_current
        self.aoi_local = 0.0
        # After TX, distortion drops to zero
        self._last_distortion_value = 0.0

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
    
    def get_time_averaged_distortion(self):
        """Return time-averaged distortion (integral / time)."""
        if self._time <= 0:
            return np.nan
        return self._distortion_time_integral / self._time

    def get_trace(self):
        return list(self._trace)


## Semantic-Driven Traffic with AR(1) Process Model
class SemanticTraffic(Traffic):
    r"""
    Semantic transmission scheduling based on AR(1) process monitoring.

    Uses the same inter-packet generation interval λ as traditional traffic,
    but decides WHETHER to transmit based on distortion vs adaptive threshold:
        a_k(t) = 1  if  |x_k(t) - x̂_k(t)| ≥ ε_th(Δ_local)
    where ε_th(Δ) = max(ε_min, ε_0 - β·Δ).
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
        # lambda_ref: fixed reference epoch for AR(1) alpha scaling.
        # Must NOT change with average_interval so the process stays the same.
        self.lambda_ref = float(traffic_param.get('lambda_ref', self.lambda_interval))
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
        
        # Transmission approval flag (set by on_decision_epoch, checked by Node)
        self._approve_transmission = True
        self._tx_config = None
        self._time = 0.0
        self._distortion_accum = 0.0
        self._decision_count = 0
        self._trace = []

        # Time-integrated distortion for fair comparison
        self._distortion_time_integral = 0.0
        self._last_distortion_value = 0.0
    
    def update_ar1(self, dt):
        """Evolve AR(1) process over elapsed time dt."""
        if self._ar1_process is not None:
            self.x_current = self._ar1_process.query(self._time)
            return
        # Internal incremental update (backward-compatible).
        if self.lambda_ref <= 0:
            dt_ratio = 1.0
        else:
            dt_ratio = max(1e-9, dt / self.lambda_ref)
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
        distortion = self.get_distortion()
        # Time-integrated distortion (trapezoidal rule)
        self._distortion_time_integral += 0.5 * (self._last_distortion_value + distortion) * dt

        self._distortion_accum += distortion
        self._decision_count += 1

        threshold = self.get_threshold(self.aoi_local)
        tx_decision = distortion >= threshold

        if tx_decision:
            self._approve_transmission = True
            self._tx_config = self._select_semantic_config(distortion)
            self.x_last_tx = self.x_current
            self.aoi_local = 0.0
            self._last_distortion_value = 0.0
        else:
            self._approve_transmission = False
            self._tx_config = None
            self._last_distortion_value = distortion

        if self.track_trace:
            self._trace.append({
                'time':        self._time,
                'x_current':   float(self.x_current),
                'x_hat':       float(self.x_last_tx),
                'distortion':  distortion,
                'threshold':   threshold,
                'tx_decision': tx_decision,
            })

    def get_average_distortion(self):
        if self._decision_count == 0:
            return np.nan
        return self._distortion_accum / self._decision_count
    
    def get_time_averaged_distortion(self):
        """Return time-averaged distortion (integral / time)."""
        if self._time <= 0:
            return np.nan
        return self._distortion_time_integral / self._time

    def get_trace(self):
        return list(self._trace)
    
    def traffic_function(self):
        # Return the next decision epoch. Semantic state update happens when that
        # epoch is reached inside Node.transmit().
        return random.expovariate(1 / self.lambda_interval)


class PrecomputedSemanticTraffic(Traffic):
    r"""
    Semantic traffic with pre-computed threshold crossings from the AR(1) process.

    Generates the full AR(1) signal at init and determines every crossing instant
    where |x(t) − x̂(t)| ≥ ε_th(AoI). Those instants become the only simulation
    events — no "skip" epochs are scheduled.

    Parameters in traffic_param
    ----------------------------
    alpha, sigma_w           : AR(1) model coefficients
    epsilon_0, epsilon_min,
    beta                     : threshold parameters  ε_th(Δ) = max(ε_min, ε_0 − β·Δ)
    average_interval         : used as lambda_ref for AR(1) alpha scaling [s]
    sim_time                 : total simulation duration [s]  ← injected by run.py
    dt_fine_semantic         : time resolution for AR(1) pre-generation [s] (default 1.0)
    semantic_configs         : list of {'max_distortion', 'headers', 'code'}
    track_trace              : bool – record per-crossing trace
    """

    def __init__(self, traffic_param: dict) -> None:
        super().__init__(traffic_param)

        self.alpha          = float(traffic_param.get('alpha', 0.95))
        self.sigma_w        = float(traffic_param.get('sigma_w', 1.0))
        # Backward-compatible alias
        self.epsilon_0      = float(traffic_param.get('epsilon_0',
                                    traffic_param.get('threshold_0', 1.0)))
        self.epsilon_min    = float(traffic_param.get('epsilon_min', 0.1))
        self.beta           = float(traffic_param.get('beta', 0.01))
        self.lambda_interval = float(traffic_param.get('average_interval', 900.0))
        # lambda_ref: fixed reference epoch for AR(1) alpha scaling.
        self.lambda_ref = float(traffic_param.get('lambda_ref', self.lambda_interval))
        self.track_trace    = bool(traffic_param.get('track_trace', False))

        sim_time  = float(traffic_param.get('sim_time', 3600.0))
        dt_fine   = float(traffic_param.get('dt_fine_semantic', 1.0))

        # Each device draws a unique seed from numpy's global (seeded) RNG so
        # that runs are deterministic given np.random.seed(...) in run.py.
        seed_val = int(np.random.randint(0, 2**31))
        AR1Cls = _get_ar1_class()
        self._ar1_process = AR1Cls(
            alpha=self.alpha,
            sigma_w=self.sigma_w,
            sim_time=sim_time,
            dt_fine=dt_fine,
            lambda_ref=self.lambda_ref,
            seed=seed_val,
        )

        default_configs = [
            {'max_distortion': 0.68 * self.epsilon_0,  'headers': 1, 'code': '5/6'},   # AoI-driven low-distortion crossing
            {'max_distortion': 0.88 * self.epsilon_0,  'headers': 2, 'code': '2/3'},   # normal crossing
            {'max_distortion': float('inf'),           'headers': 3, 'code': '1/3'},   # high-distortion crossing
        ]
        self.semantic_configs = sorted(
            traffic_param.get('semantic_configs', default_configs),
            key=lambda c: c.get('max_distortion', float('inf')),
        )

        # Pre-computed crossing times and associated data
        self._tx_times:       list[float] = []
        self._tx_distortions: list[float] = []
        self._tx_thresholds:  list[float] = []
        self._tx_configs:     list        = []
        self._tx_index: int = 0

        self._precompute_crossings()

        # Run-time metrics
        self._time              = 0.0
        self._distortion_accum  = 0.0
        self._decision_count    = 0
        self._tx_config         = None
        self._approve_transmission = True
        self._trace: list = []

    # ── Threshold helper ──────────────────────────────────────────────────

    def _get_threshold(self, aoi: float) -> float:
        return max(self.epsilon_min, self.epsilon_0 - self.beta * aoi)

    def _select_semantic_config(self, distortion: float):
        if not self.semantic_configs:
            return None
        for cfg in self.semantic_configs:
            if distortion < cfg.get('max_distortion', float('inf')):
                return cfg
        return self.semantic_configs[-1]

    # ── Core pre-computation ──────────────────────────────────────────────

    def _precompute_crossings(self) -> None:
        """Scan the pre-generated AR(1) signal and record every crossing instant.
        
        Also computes the exact time-integrated distortion:
            D̄ = (1/T) ∫₀ᵀ |x(t) - x̂(t)| dt
        using rectangular rule at dt_fine resolution.
        """
        t_arr = self._ar1_process.t_array
        x_arr = self._ar1_process.x_array

        x_last_tx   = float(x_arr[0])
        last_tx_time = float(t_arr[0])

        tx_times:       list[float] = []
        tx_distortions: list[float] = []
        tx_thresholds:  list[float] = []
        tx_configs:     list        = []

        # Time-integrated distortion (exact from full signal)
        distortion_integral = 0.0

        for i in range(1, len(t_arr)):
            t         = float(t_arr[i])
            x         = float(x_arr[i])
            dt_step   = t - float(t_arr[i - 1])
            aoi       = t - last_tx_time
            distortion = abs(x - x_last_tx)

            # Accumulate time-integrated distortion (rectangular rule at fine resolution)
            distortion_integral += distortion * dt_step

            threshold  = self._get_threshold(aoi)

            if distortion >= threshold:
                tx_times.append(t)
                tx_distortions.append(distortion)
                tx_thresholds.append(threshold)
                tx_configs.append(self._select_semantic_config(distortion))
                x_last_tx    = x
                last_tx_time = t

        self._tx_times       = tx_times
        self._tx_distortions = tx_distortions
        self._tx_thresholds  = tx_thresholds
        self._tx_configs     = tx_configs

        # Store time-integrated distortion
        total_time = float(t_arr[-1] - t_arr[0]) if len(t_arr) > 1 else 1.0
        self._distortion_time_integral = distortion_integral
        self._total_precomputed_time   = total_time

    # ── Traffic interface ─────────────────────────────────────────────────

    def traffic_function(self) -> float:
        """
        Return the inter-event time to the next pre-computed threshold crossing.

        When all crossings have been consumed, returns a time far beyond the
        simulation end so the node stays idle for the remainder of the run.
        """
        if self._tx_index >= len(self._tx_times):
            # Idle for the rest of the simulation
            return self._ar1_process.sim_time + 1e6

        t_next = self._tx_times[self._tx_index]
        t_prev = self._tx_times[self._tx_index - 1] if self._tx_index > 0 else 0.0
        return t_next - t_prev

    def on_decision_epoch(self, dt: float) -> None:
        """
        Called by Node.transmit() after the timer fires.

        Advances the internal clock and commits metrics for the current crossing.
        Index is incremented here so that the *next* call to traffic_function()
        returns the interval to the following crossing.
        """
        self._time += dt
        i = self._tx_index  # index of the crossing that just fired

        if i < len(self._tx_times):
            distortion = self._tx_distortions[i]
            threshold  = self._tx_thresholds[i]
            cfg        = self._tx_configs[i]

            self._distortion_accum += distortion
            self._decision_count   += 1
            self._approve_transmission = True
            self._tx_config = cfg

            if self.track_trace:
                t_c = self._tx_times[i]
                x_c = float(np.interp(t_c,
                                      self._ar1_process.t_array,
                                      self._ar1_process.x_array))
                self._trace.append({
                    'time':        t_c,
                    'x_current':   x_c,
                    'x_hat':       x_c - distortion,
                    'distortion':  distortion,
                    'threshold':   threshold,
                    'tx_decision': True,
                })

            self._tx_index += 1
        else:
            self._approve_transmission = False
            self._tx_config = None

    def should_send_now(self) -> bool:
        """Every scheduled event is a genuine transmission."""
        return self._approve_transmission

    def get_tx_params(self):
        return self._tx_config

    def get_average_distortion(self) -> float:
        if self._decision_count == 0:
            return float('nan')
        return self._distortion_accum / self._decision_count

    def get_time_averaged_distortion(self) -> float:
        """Return time-averaged distortion from pre-computed full signal."""
        if self._total_precomputed_time <= 0:
            return float('nan')
        return self._distortion_time_integral / self._total_precomputed_time

    def get_trace(self) -> list:
        return list(self._trace)