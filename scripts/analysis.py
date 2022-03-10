import os
import pickle
from utils.information_theory import prob, entropy
import numpy as np
import matplotlib.pyplot as plt

results_path = "/Users/yairmazal/google_drive/My Drive/University/PhD/NR_telemetry_ECC/runs/HC_eilat_July_2018/results"
summary_results = {}

# list dirs
dirs = os.listdir(results_path)
for dir in dirs:
    if os.path.isdir(os.path.join(results_path, dir)):
        files = os.listdir(os.path.join(results_path, dir))
        for file in files:
            if "summary" in file:
                with open(os.path.join(results_path, dir, file), "rb") as f:
                    summary_results[file] = pickle.load(f)

# real entropy of data
# load encoded data
with open(os.path.join(results_path, "2022-03-03_19_17_18/2022-03-03_19_17_18_simulation_entropy_vs_pure_LDPC.pickle"), "rb") as f:
    encoded = pickle.load(f)[0]['encoded']
encoded = np.array([np.array(buffer, dtype=np.int_) for buffer in encoded])
p = prob(encoded)
buffer_ent = entropy(p)

figure = plt.figure()
plt.plot(buffer_ent, '.')
plt.xlabel("buffer index")
plt.ylabel("binary entropy of whole buffer")
plt.show()

bit_dist = prob(encoded.T)
bit_ent = entropy(bit_dist)

figure = plt.figure()
plt.plot(bit_ent, '.', np.array([0.2] * len(bit_ent)))
plt.xlabel("bit index")
plt.ylabel("binary entropy of each bit")
plt.show()
print("percent of structure bits: ", np.sum(bit_ent < 0.2)*100/len(bit_ent), "%")

