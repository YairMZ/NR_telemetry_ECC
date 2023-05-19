from decoders import Decoder, DecoderType
from inference import BufferClassifier
from collections.abc import Sequence
from ldpc.decoder import LogSpaDecoder
from numpy.typing import NDArray
import numpy as np
from typing import Optional
from utils.custom_exceptions import IncorrectBufferLength
from utils.information_theory import prob, entropy


class ClassifyingEntropyDecoder(Decoder):
    def __init__(self, ldpc_decoder: LogSpaDecoder, model_length: int, entropy_threshold: float, clipping_factor: float,
                 classifier_training: int, n_clusters: int,
                 window_length: Optional[int] = None, conf_center: int = 40, conf_slope: float = 0.35,
                 bit_flip: float = 0, cluster: int = 1, data_model: Optional[NDArray[np.float_]] = None
                 ) -> None:
        """
        Create a new decoder
        :param ldpc_decoder: decoder for ldpc code used
        :param model_length: length of assumed model in bits, first info bits are assumed to be model bits
        :param entropy_threshold: threshold for entropy to dictate structural elements
        :param clipping_factor: the maximum model llr is equal to the clipping_factor times the maximal channel llr
        :param window_length: number of last messages to consider when evaluating distribution and entropy. If none all
        """
        self.ldpc_decoder: LogSpaDecoder = ldpc_decoder
        self.model_length: int = model_length  # in bits
        self.entropy_threshold = entropy_threshold
        self.model_bits_idx = np.array(self.ldpc_decoder.info_idx)
        if model_length >= sum(self.model_bits_idx):
            self.model_bits_idx = np.array(range(model_length))
        else:
            self.model_bits_idx[model_length:] = False
            self.model_bits_idx = np.flatnonzero(self.model_bits_idx)  # bit indices (among codeword bits) of model bits
        self.clipping_factor = clipping_factor  # The model llr is clipped to +-clipping_factor * max_chanel_llr

        if data_model is None:
            self.distributions: list[NDArray[np.float_]] = [np.array([]) for _ in range(n_clusters)]
            self.models_entropy: list[NDArray[np.float_]] = [np.array([]) for _ in range(n_clusters)]
            self.train_models = True
        else:
            self.distributions = [data_model[:model_length,:] for _ in range(n_clusters)]
            self.models_entropy = [entropy(self.distributions[cluster_id]) for cluster_id in range(n_clusters)]
            self.train_models = False
        self.conf_center = conf_center
        self.conf_slope = conf_slope
        self.bit_flip = bit_flip
        self.window_length = window_length
        self.models_data: list[NDArray[np.uint8]] = [np.array([]) for _ in range(n_clusters)]

        self.classifier = BufferClassifier(classifier_training, n_clusters)
        self.n_clusters = n_clusters
        self.cluster = bool(cluster)
        self.running_idx = -1
        super().__init__(DecoderType.CLASSIFYING)

    def decode_buffer(self, channel_llr: Sequence[np.float_]) -> tuple[NDArray[np.int_], NDArray[np.float_], bool, int,
                                                                       NDArray[np.int_], NDArray[np.int_],
                                                                       NDArray[np.float_], NDArray[np.int_], int]:
        """decodes a buffer
        :param channel_llr: channel llr of bits to decode
        :return: return a tuple (estimated_bits, llr, decode_success, no_iterations, no of mavlink messages found)
        where:
            - estimated_bits is a 1-d np array of hard bit estimates
            - llr is a 1-d np array of soft bit estimates
            - decode_success is a boolean flag stating of the estimated_bits form a valid  code word
            - number of iterations until breaking
            - syndrome
            - vnode_validity - for each vnode how many equations failed
            - number of MAVLink messages found within buffer
            - inferred distribution of model
            - inferred structural bits of model
            - cluster label
        """
        # classify
        channel_bits = np.array(channel_llr < 0, dtype=np.int_)[self.model_bits_idx]
        if self.n_clusters > 1:
            if self.cluster:
                label: int = self.classifier.classify(channel_bits)
                if label < 0:  # label is negative during training phase of classifier, don't try to use model
                    estimate, llr, decode_success, iterations, syndrome, vnode_validity = self.ldpc_decoder.decode(channel_llr)
                    return estimate, llr, decode_success, iterations, syndrome, vnode_validity, np.array([]), np.array([]), label
            else:
                self.running_idx = (self.running_idx + 1) % self.n_clusters
                label = self.running_idx
        else:
            label = 0
        # estimate, llr, decode_success, iterations, syndrome, vnode_validity = self.ldpc_decoder.decode(channel_llr)
        # if decode_success:  # buffer fully recovered
        #     model_bits = estimate[self.model_bits_idx]
        #     self.update_model(model_bits, label)
        #     return estimate, llr, decode_success, iterations, syndrome, vnode_validity, self.distributions[label], \
        #         self.model_bits_idx[self.models_entropy[label] < self.entropy_threshold], label
        model_llr = self.model_prediction(channel_llr, label)  # type: ignore
        estimate, llr, decode_success, iterations, syndrome, vnode_validity = self.ldpc_decoder.decode(model_llr)
        model_bits = estimate[self.model_bits_idx]
        if decode_success:  # buffer fully recovered
            self.update_model(model_bits, label)
        else:  # update model from channel if data is bad
            self.update_model(channel_bits, label)
        return estimate, llr, decode_success, iterations, syndrome, vnode_validity, self.distributions[label], \
               self.model_bits_idx[self.models_entropy[label] < self.entropy_threshold], label

    def update_model(self, bits: NDArray[np.int_], cluster_id: int) -> None:
        """update model of data. model_b uses any data regardless of correctness.
        :param bits: hard estimate for bit values, assumed to be correct.
        """
        if not self.train_models:
            return
        if len(bits) != self.model_length:
            print(self.model_length)
            print(len(bits))
            raise IncorrectBufferLength()
        arr = bits[np.newaxis]

        self.models_data[cluster_id] = arr.T if self.models_data[cluster_id].size == 0 else \
            np.append(self.models_data[cluster_id], arr.T, axis=1)
        if (self.window_length is not None) and self.models_data[cluster_id].shape[1] > self.window_length:
            # trim old messages according to window
            self.models_data[cluster_id] = self.models_data[cluster_id][
                                           :, self.models_data[cluster_id].shape[1] - self.window_length:]
        dist = prob(self.models_data[cluster_id])
        v = dist[:, 1]
        p1 = np.clip((v - self.bit_flip) / (1 - 2 * self.bit_flip), 0, 1)
        self.distributions[cluster_id] = np.column_stack((1 - p1, p1))
        # mu equals p1
        if max(p1) > 1 or min(p1) < 0:
            print("problematic probability")
            raise RuntimeError("problematic probability")
        self.models_entropy[cluster_id] = entropy(self.distributions[cluster_id])

    def model_prediction(self, observation: NDArray[np.float_], cluster_id: int) -> NDArray[np.float_]:
        def model_confidence(model_size: int, center: int, slope: float) -> np.float_:
            return 0 if model_size <= 1 else 1 / (1 + np.exp(-(model_size - center) * slope, dtype=np.float_))

        llr = observation.copy()
        # infer structure
        # index of structural (low entropy) elements among codeword
        structural_elements: NDArray[np.int_] = self.model_bits_idx[
            self.models_entropy[cluster_id] < self.entropy_threshold]

        if not structural_elements.any():  # no structural elements found
            return llr
        if self.train_models:
            size = self.models_data[cluster_id].shape[1] if self.models_data[cluster_id].size > 0 else 0
            confidence = model_confidence(size, self.conf_center, self.conf_slope)  # consider window size when setting these
        else:
            confidence = 1
        clipping = self.clipping_factor * max(llr)  # llr s clipped within +-clipping
        # model llr is calculated as log(Pr(c=0 | model) / Pr(c=1| model))
        # add model llr to the observation
        if confidence > 0:
            llr[structural_elements] += confidence * np.log(
                (np.finfo(np.float_).eps + self.distributions[cluster_id][:, 0]) / (self.distributions[cluster_id][:, 1] + np.finfo(np.float_).eps)
            )[self.models_entropy[cluster_id] < self.entropy_threshold]

        return np.clip(llr, -clipping, clipping)


__all__ = ["ClassifyingEntropyDecoder"]
