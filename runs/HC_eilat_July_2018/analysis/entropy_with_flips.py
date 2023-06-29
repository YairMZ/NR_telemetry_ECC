import numpy as np
p0 = 0.9  # real probability of 0
p1 = 1 - p0
p_flip = 0.05
ent_real = -p0 * np.log2(p0) - p1 * np.log2(p1)

p0_measured = p0 + p_flip
p1_measured = 1 - p0_measured
ent_measured = - p0_measured * np.log2(p0_measured) - p1_measured * np.log2(p1_measured)
