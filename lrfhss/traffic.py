import random
import numpy as np
from lrfhss.lrfhss_core import Traffic
import warnings


## Exponential traffic
class Exponential_Traffic(Traffic):
    def __init__(self, traffic_param):
        super().__init__(traffic_param)
        if not 'average_interval' in self.traffic_param:
            warnings.warn(f'traffic_param average_interval key missing for Exponential_Traffic. Using with average_interval=900 as default')
            self.traffic_param['average_interval'] = 900

    def traffic_function(self):
        return random.expovariate(1/self.traffic_param['average_interval'])

## Uniform traffic
class Uniform_Traffic(Traffic):
    def __init__(self, traffic_param):
        super().__init__(traffic_param)
        if not 'max_interval' in self.traffic_param:
            warnings.warn(f'traffic_param max_interval key missing for Uniform_Traffic. Using with max_interval=1800 as default')
            self.traffic_param['max_interval'] = 1800

    def traffic_function(self):
        return random.uniform(0,self.traffic_param['max_interval'])

## Constant traffic with small gaussian deviation
class Constant_Traffic(Traffic):
    def __init__(self, traffic_param):
        super().__init__(traffic_param)
        if not 'constant_interval' in self.traffic_param:
            warnings.warn(f'traffic_param constant_interval key missing for Constant_Traffic. Using with constant_interval=900 as default')
            self.traffic_param['constant_interval'] = 900

        if not 'standard_deviation' in self.traffic_param:
            warnings.warn(f'traffic_param standard_deviation key missing for Constant_Traffic. Using with standard_deviation=900 as default')
            self.traffic_param['standard_deviation'] = 10

    def traffic_function(self):
        # First transmissions is random, devices are not initiated at the same time
        if self.transmitted == 0:
            return random.uniform(0,2*self.traffic_param['constant_interval'])
        else:
            return max(0, self.traffic_param['constant_interval'] + random.gauss(0,self.traffic_param['standard_deviation']))
    

## 2-state Markovian Traffic
class Two_State_Markovian_Traffic(Traffic):
    def __init__(self, traffic_param):
        super().__init__(traffic_param)
        if not 'transition_matrix' in self.traffic_param:
            warnings.warn(f'traffic_param transition_matrix key missing for Two_State_Markovian_Traffic. Using transition_matrix=[0.5, 0.5; 0.5, 0.5] as default')
            self.traffic_param['transition_matrix'] = [[0.5, 0.5],[0.5, 0.5]]
        
        if not 'markov_time' in self.traffic_param:
            warnings.warn(f'traffic_param markov_time key missing for Two_State_Markovian_Traffic. Using markov_time=0.5 as default')
            self.traffic_param['markov_time'] = 0.5

    def traffic_function(self):
        discrete_time = 1
        try:
            state = self.state
        except AttributeError:
            state = 0

        if random.random() >= self.traffic_param['transition_matrix'][state][0]:
            return max(0,discrete_time * self.traffic_param['markov_time'] + random.gauss(0,1))

        discrete_time+=1
        transition_probability = self.traffic_param['transition_matrix'][0][0]
        while random.random() < transition_probability:
            discrete_time+=1
        
        self.state=1
        return max(0,discrete_time * self.traffic_param['markov_time'] + random.gauss(0,1))


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
        self.epsilon_0 = traffic_param.get('epsilon_0', 1.0)
        self.epsilon_min = traffic_param.get('epsilon_min', 0.1)
        self.beta = traffic_param.get('beta', 0.01)
        
        # CRITICAL for fair comparison: use same interval as traditional traffic
        if 'average_interval' not in traffic_param:
            warnings.warn('SemanticTraffic: average_interval missing. Using default 900s')
            traffic_param['average_interval'] = 900.0
        self.lambda_interval = traffic_param['average_interval']
        
        # AR(1) process state
        # Initialize with random values to break synchronization between nodes
        self.x_current = np.random.normal(0, self.sigma_w)
        self.x_last_tx = np.random.normal(0, self.sigma_w)  # Random initial estimate
        self.aoi_local = np.random.uniform(0, self.lambda_interval)  # Random initial AoI
        
        # Transmission approval flag (set by traffic_function, checked by Node)
        self._approve_transmission = True
        
        # Tracking for avoiding multiple updates in same λ-period
        # Set to True when traffic_function() is called, False when Node transmits/skips
        self._called_this_period = False
    
    def update_ar1(self, dt):
        """Evolve AR(1) process over time dt."""
        w = np.random.normal(0, self.sigma_w)
        self.x_current = self.alpha * self.x_current + w
    
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
    
    def traffic_function(self):
        """
        SEMANTIC TRAFFIC DECISION AT λ-INTERVALS:
        
        DESIGN: Called by Node at λ-intervals. Decides whether next transmission is
        semantically relevant and sets _approve_transmission flag.
        
        CRITICAL: Add jitter/randomness to break synchronization (like Exponential_Traffic).
        Use exponential distribution with mean λ to maintain fairness while adding asynchrony.
        
        Only process evolution happens on FIRST call per λ-period (tracked via _called_this_period).
        Subsequent calls in same period return immediately without processing.
        
        Flow:
        1. If already called this period: return exponential without processing
        2. Else: advance AR(1), update AoI, evaluate decision
        3. Set _approve_transmission flag
        4. Mark as "called this period"
        5. Return exponential random time (like Traditional traffic)
        """
        # If already called this period, return exponential without processing
        # (avoid multiple AR(1) evolutions in same λ-interval)
        if self._called_this_period:
            return random.expovariate(1 / self.lambda_interval)
        
        # Mark as called this period
        self._called_this_period = True
        
        # Step 1: Advance physical process (λ time has passed since last decision)
        self.update_ar1(dt=self.lambda_interval)
        
        # Step 2: Increment local AoI
        self.aoi_local += self.lambda_interval
        
        # Step 3: Evaluate transmission decision using CURRENT state
        if self.should_transmit():
            # APPROVE: Semantic relevance condition met
            self._approve_transmission = True
            self.x_last_tx = self.x_current
            self.aoi_local = 0.0
        else:
            # DENY: Not semantically relevant
            self._approve_transmission = False
        
        # Step 4: Return exponential random time (for asynchrony like Traditional traffic)
        # This breaks synchronization while maintaining fair average interval λ
        return random.expovariate(1 / self.lambda_interval)