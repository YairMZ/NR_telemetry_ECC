import pickle
from bitstring import Bits, BitArray
import numpy as np
from inference import BufferClassifier
from scipy.io import savemat
import os


with open('../runs/HC_eilat_July_2018/data/hc_to_ship.pickle', 'rb') as f:
    hc_tx = pickle.load(f)
rng = np.random.default_rng()
two_sec_bin = [Bits(auto=tx.get("bin")) for tx in hc_tx.get("20000")]
weight_scheme = "sqrt"  # see implemented schemes in merge_clusters method of BufferClassifier
print(weight_scheme)


# option 1: use rate 1/2 with N=1296, k=648, and two classes of buffers
buffers = []
n = 1296
r = 1/2
p = 0.07
k = int(n*r)
pad_len = k - len(two_sec_bin[0]) // 2
no_errors = int(k*p)
n_classes = 2
n_training = 100
for b in two_sec_bin:
    errors = rng.choice(k, size=no_errors, replace=False)
    corrupted = BitArray(b[:576] + Bits(auto=rng.integers(low=0, high=2, size=k-576)))
    for idx in errors:
        corrupted[idx] = not corrupted[idx]
    buffers.append((Bits(corrupted), 0))

    errors = rng.choice(k, size=no_errors, replace=False)
    corrupted = BitArray(b[576:])
    for idx in errors:
        corrupted[idx] = not corrupted[idx]
    buffers.append((Bits(corrupted), 1))

    # buffers.append((b[:612] + Bits(auto=rng.integers(low=0, high=2, size=pad_len)), 0))
    # buffers.append((b[612:] + Bits(auto=rng.integers(low=0, high=2, size=pad_len)), 1))

    # buffers.append((b[:612], 0))
    # buffers.append((b[612:], 1))
# classification metrics can be "hamming" or "LL".
# merging metrics can be "KlDiv" (symmetric) and "Hellinger". KlDiv underperformed due to unbound nature. linear
classifier = BufferClassifier(n_training, n_classes, classify_dist="LL", merge_dist="Hellinger", weight_scheme=weight_scheme)
actual_classes = np.zeros(len(buffers), dtype=np.int_)
labels = np.zeros(len(buffers), dtype=np.int_)
savemat(f"n_classes_{n_classes}_clustering_data.mat",
        {f"clustering_data_{n_classes}": buffers}, do_compression=True)
for idx, b in enumerate(buffers):
    actual_classes[idx] = b[1]
    labels[idx] = classifier.classify(b[0])
result_1 = [np.unique(labels[n_training - 1 + i :: n_classes], return_counts=True) for i in range(n_classes)]

# option 2: use rate 3/4 with N=648, k=486, and three classes of buffers
buffers = []
n = 648
r = 3/4
p = 0.07
k = int(n*r)
pad_len = k - len(two_sec_bin[0]) // 2
no_errors = int(k*p)
n_classes = 3
n_training = 100
for b in two_sec_bin:
    errors = rng.choice(k, size=no_errors, replace=False)
    corrupted = BitArray(b[:416] + Bits(auto=rng.integers(low=0, high=2, size=k-416)))
    for idx in errors:
        corrupted[idx] = not corrupted[idx]
    buffers.append((Bits(corrupted), 0))

    errors = rng.choice(k, size=no_errors, replace=False)
    corrupted = BitArray(b[416:864] + Bits(auto=rng.integers(low=0, high=2, size=k-(864-416))))
    for idx in errors:
        corrupted[idx] = not corrupted[idx]
    buffers.append((Bits(corrupted), 1))

    errors = rng.choice(k, size=no_errors, replace=False)
    corrupted = BitArray(b[864:] + Bits(auto=rng.integers(low=0, high=2, size=k - (1224 - 864))))
    for idx in errors:
        corrupted[idx] = not corrupted[idx]
    buffers.append((Bits(corrupted), 2))
    # buffers.append((b[:612] + Bits(auto=rng.integers(low=0, high=2, size=pad_len)), 0))
    # buffers.append((b[612:] + Bits(auto=rng.integers(low=0, high=2, size=pad_len)), 1))

# classification metrics can be "hamming" or "LL".
# merging metrics can be "KlDiv" (symmetric) and "Hellinger". KlDiv underperformed due to unbound nature. linear
classifier = BufferClassifier(n_training, n_classes, classify_dist="LL", merge_dist="Hellinger", weight_scheme=weight_scheme)
actual_classes = np.zeros(len(buffers), dtype=np.int_)
labels = np.zeros(len(buffers), dtype=np.int_)
savemat(f"n_classes_{n_classes}_clustering_data.mat",
        {f"clustering_data_{n_classes}": buffers}, do_compression=True)
for idx, b in enumerate(buffers):
    actual_classes[idx] = b[1]
    labels[idx] = classifier.classify(b[0])
result_2 = [np.unique(labels[n_training - 1 + i::n_classes], return_counts=True) for i in range(n_classes)]
# values1, counts1 = np.unique(labels[99::3], return_counts=True)
# values2, counts2 = np.unique(labels[100::2], return_counts=True)