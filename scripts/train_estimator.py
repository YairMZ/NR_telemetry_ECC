import numpy as np
import pickle
from bitstring import Bits
import argparse
from ldpc.wifi_spec_codes import WiFiSpecCode
from ldpc.encoder import EncoderWiFi
import json
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='Run decoding on simulated data using multiprocessing.')
parser.add_argument("--n_clusters", default=1, help="number of clusters", type=int)
args = parser.parse_args()

rng = np.random.default_rng()

with open('../runs/HC_eilat_July_2018/data/hc_to_ship.pickle', 'rb') as f:
    hc_tx = pickle.load(f)
    hc_bin_data = [Bits(auto=tx.get("bin")) for tx in hc_tx.get("20000")]

if args.n_clusters == 1:
    spec = WiFiSpecCode.N1944_R23
elif args.n_clusters == 2:
    spec = WiFiSpecCode.N1296_R12
elif args.n_clusters == 3:
    spec = WiFiSpecCode.N648_R34
else:
    raise ValueError("Invalid number of clusters")

n = len(hc_bin_data)
encoder = EncoderWiFi(spec=spec)
encoded = []
for binary_data in hc_bin_data[:n]:
    if args.n_clusters == 1:
        pad_len = encoder.k - len(binary_data)
        padded = binary_data + Bits(auto=rng.integers(low=0, high=2, size=pad_len))
        encoded.append(encoder.encode(padded))
    elif args.n_clusters == 2:
        padded = binary_data[:576] + Bits(auto=rng.integers(low=0, high=2, size=encoder.k-576))
        encoded.extend((encoder.encode(padded), encoder.encode(binary_data[576:])))
    elif args.n_clusters == 3:
        padded = binary_data[:416] + Bits(auto=rng.integers(low=0, high=2, size=encoder.k-416))
        encoded.append(encoder.encode(padded))
        padded = binary_data[416:864] + Bits(auto=rng.integers(low=0, high=2, size=encoder.k - (864-416)))
        encoded.append(encoder.encode(padded))
        padded = binary_data[864:] + Bits(auto=rng.integers(low=0, high=2, size=encoder.k - (1224-864)))
        encoded.append(encoder.encode(padded))

data = np.zeros((len(encoded), encoder.n), dtype=np.uint16)
for i, encoded_bits in enumerate(encoded):
    data[i] = np.array(encoded_bits, dtype=np.uint16)
p = sum(data)/len(encoded)
dist = np.zeros((encoder.n, 2), dtype=np.float_)
dist[:, 0] = 1-p
dist[:, 1] = p

plt.hist(p, bins='auto')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()

with open(f'../runs/HC_eilat_July_2018/data/encoded_data_data_model_{args.n_clusters}_clusters.json', 'w') as f:
    json.dump(dist.tolist(), f, indent=4, sort_keys=True)
