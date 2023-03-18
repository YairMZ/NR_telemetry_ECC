from itertools import combinations
from bitstring import Bits
from numpy.typing import NDArray
import numpy as np
from utils.information_theory import prob, hellinger
from utils.bit_operations import hamming_distance
from scipy.stats import entropy


class Cluster:
    def __init__(self, dist: NDArray[np.float_], uid: int, n_samples: int = 1) -> None:
        self._data: NDArray[np.int_] = np.array(dist * n_samples, dtype=np.int_)
        self.uid: int = uid
        self.n_samples = n_samples

    @property
    def hard_bits(self) -> NDArray[np.int_]:
        return np.array(self._data[:, 1] > self.dist[:, 0], dtype=np.int_)

    @property
    def dist(self) -> NDArray[np.float_]:
        return self._data / self.n_samples

    def add_sample(self, buffer: Bits) -> None:
        sample = np.array(buffer, dtype=np.int_)
        self._data[:, 0] += (sample == 0)
        self._data[:, 1] += (sample == 1)
        self.n_samples += 1

    def merge_cluster(self, cluster) -> None:
        self._data += cluster._data
        self.n_samples += cluster.n_samples

    def __str__(self) -> str:
        return f'cluster id: {self.uid}, with {self.n_samples} samples'

    def __len__(self) -> int:
        return self.n_samples


class BufferClassifier:
    def __init__(self, min_data_size: int, k: int, classify_dist: str = "LL", merge_dist: str = "Hellinger",
                 weight_scheme: str = "linear") -> None:
        """

        :param min_data_size: number of samples before clustering is initialized
        :param k: number of clusters
        :param classify_dist: metric used to classify a buffer
        :param merge_dist: metric used to merge clusters
        :param weight_scheme: weight scheme used to merge clusters
        """
        self.min_data_size: int = min_data_size
        self.k: int = k
        self.clusters: list[Cluster] = []
        self.init: bool = False
        self.__cluster_uid: int = 0
        self.samples: int = 0
        self.classify_dist = classify_dist
        self.merge_dist = merge_dist
        self.weight_scheme = weight_scheme

    def classify(self, buffer: Bits) -> int:
        self.samples += 1
        if self.samples < self.min_data_size:
            sample = np.array(buffer, dtype=np.int_)[np.newaxis].T
            self.__add_cluster(prob(sample))
            return -1
        if not self.init:
            self.merge_clusters()
            self.init = True

        metric = None
        for cluster in self.clusters:
            if self.classify_dist == "hamming":
                d = hamming_distance(buffer, cluster.hard_bits)
                if (metric is None) or (d < metric):
                    metric = d
                    closest: Cluster = cluster
            else:  # use log likelihood
                ll = np.mean(np.clip(
                    np.log(cluster.dist[:, 1] + np.finfo(np.float_).eps) * np.array(buffer, dtype=np.int_) + np.log(
                        cluster.dist[:, 0] + np.finfo(np.float_).eps) * (1 - np.array(buffer, dtype=np.int_)),
                    -100, 0))
                if (metric is None) or (ll > metric):
                    metric = ll
                    closest = cluster

        closest.add_sample(buffer)
        return closest.uid

    def __add_cluster(self, dist: NDArray[np.float_]) -> Cluster:
        self.clusters.append(Cluster(dist, self.__cluster_uid))
        self.__cluster_uid += 1
        return self.clusters[-1]

    def merge_clusters(self) -> None:
        while len(self.clusters) > self.k:
            # find all pairs of clusters
            indices = list(combinations(range(len(self.clusters)), 2))
            min_dist: float = None  # type: ignore
            for c1, c2 in indices:
                mx = (len(self.clusters[c1]) + len(self.clusters[c2]))/2
                if self.weight_scheme == "sqrt":
                    w = np.sqrt(mx)  # square root weights
                elif self.weight_scheme == "linear":
                    w = mx
                else:  # no weighing used
                    w = 1
                if self.merge_dist == "KlDiv":
                    d = w*np.mean(np.clip(entropy(self.clusters[c1].dist, self.clusters[c2].dist, base=2, axis=1) +
                                          entropy(self.clusters[c2].dist, self.clusters[c1].dist, base=2, axis=1),
                                          0, 10))
                else:  # hellinger distance
                    d = w*np.mean(hellinger(self.clusters[c1].dist, self.clusters[c2].dist))
                if (min_dist is None) or (d < min_dist):
                    min_dist = d
                    pair = c1, c2
            c: Cluster = self.clusters.pop(pair[1])
            self.clusters[pair[0]].merge_cluster(c)
        for idx, c, in enumerate(self.clusters):
            c.uid = idx


__all__: list[str] = ["Cluster", "BufferClassifier"]
