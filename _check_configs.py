import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
from lrfhss.traffic import PrecomputedSemanticTraffic
from examples.organized.sim_params import get_semantic_params
from collections import Counter

np.random.seed(42)
p = get_semantic_params()
p["sim_time"] = 3600.0
tg = PrecomputedSemanticTraffic(p)

configs = [cfg.get("headers") for cfg in tg._tx_configs]
c = Counter(configs)
total = len(configs)
print("Total crossings:", total)
for k, v in sorted(c.items()):
    print("  h=%d: %d (%.1f%%)" % (k, v, 100.0*v/total))

distortions = tg._tx_distortions
print("\nDistortion at crossings:")
print("  min=%.3f  max=%.3f  p25=%.3f  p50=%.3f  p75=%.3f" % (
    min(distortions), max(distortions),
    sorted(distortions)[int(0.25*len(distortions))],
    sorted(distortions)[int(0.50*len(distortions))],
    sorted(distortions)[int(0.75*len(distortions))],
))
print("  epsilon_0=%.2f  boundaries: h1<2.00  h2<3.50  h3>=3.50" % tg.epsilon_0)
