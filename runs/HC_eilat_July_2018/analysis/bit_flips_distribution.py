"""
presume buffer sof 1944 bits each, with bit flip probability of 0.06.
This implies 116 bit flips per buffer.
Presume 1432 buffers for simulation.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="darkgrid")
n = 1432
n_bits = 1944
flip_p = 0.06

# draw indices of bit flips per buffer
rng = np.random.default_rng()
error_idx = np.vstack(
    tuple(rng.choice(n_bits, size=int(n_bits * flip_p), replace=False)
          for _ in range(n))
)
df = pd.DataFrame({"error_idx":error_idx.flatten()})

sns.displot(df, x="error_idx", bins=100)
#%%
