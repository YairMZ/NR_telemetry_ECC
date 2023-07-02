from decoders import Decoder, DecoderType
from inference import BufferClassifier
from ldpc.decoder import LogSpaDecoder, WbfDecoder, GalBfDecoder
from numpy.typing import NDArray
import numpy as np
from typing import Optional
from utils.custom_exceptions import IncorrectBufferLength
from utils.information_theory import prob, entropy


class ClassifyingEntropyDecoder(Decoder):
    def __init__(self, ldpc_decoder: LogSpaDecoder | WbfDecoder | GalBfDecoder,
                 model_length: int, entropy_threshold: float,
                 clipping_factor: float,
                 classifier_training: int, n_clusters: int,
                 window_length: Optional[int] = None, conf_center: int = 40, conf_slope: float = 0.35,
                 bit_flip: float = 0, cluster: int = 1, data_model: Optional[NDArray[np.float_]] = None,
                 reliability_method: int = 0) -> None:
        """
        Create a new decoder
        :param ldpc_decoder:  for ldpc code used
        :param model_length: length of assumed model in bits, first info bits are assumed to be model bits
        :param entropy_threshold: threshold for entropy to dictate structural elements
        :param clipping_factor: the maximum model llr is equal to the clipping_factor times the maximal channel llr
        :param window_length: number of last messages to consider when evaluating distribution and entropy. If none all
        """
        self.ldpc_decoder: LogSpaDecoder | WbfDecoder | GalBfDecoder = ldpc_decoder
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
            self.distributions = [data_model[:model_length, :] for _ in range(n_clusters)]
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
        self.reliability_method = reliability_method
        super().__init__(DecoderType.CLASSIFYING)

    def decode_buffer(self, channel_llr: NDArray[np.float_]) -> tuple[NDArray[np.int_], NDArray[np.float_], bool, int,
    NDArray[np.int_], NDArray[np.int_], NDArray[np.float_], NDArray[np.int_], int]:
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
                    if isinstance(self.ldpc_decoder, LogSpaDecoder):
                        estimate, llr, decode_success, iterations, syndrome, vnode_validity = self.ldpc_decoder.decode(
                            channel_llr)
                        return estimate, llr, decode_success, iterations, syndrome, vnode_validity, np.array([]), np.array(
                            []), label
                    elif isinstance(self.ldpc_decoder, WbfDecoder):  # wbf decoder
                        estimate, decode_success, iterations, syndrome, vnode_validity = self.ldpc_decoder.decode(channel_llr)
                        return estimate, np.array([]), decode_success, iterations, syndrome, vnode_validity, np.array([]), \
                            np.array([]), label
                    elif isinstance(self.ldpc_decoder, GalBfDecoder):
                        hard_channel_bits = np.array(channel_llr < 0, dtype=np.int_)
                        estimate, decode_success, iterations, syndrome, vnode_validity = self.ldpc_decoder.decode(
                            hard_channel_bits)
                        return estimate, np.array([]), decode_success, iterations, syndrome, vnode_validity, np.array([]), \
                            np.array([]), label
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
        model_llr = self.model_prediction(len(channel_llr), label)  # type: ignore
        if isinstance(self.ldpc_decoder, LogSpaDecoder):
            clipping = self.clipping_factor * max(channel_llr)  # llr s clipped within +-clipping
            model_llr = np.clip(channel_llr + model_llr, -clipping, clipping)
            estimate, llr, decode_success, iterations, syndrome, vnode_validity = self.ldpc_decoder.decode(model_llr)
        elif isinstance(self.ldpc_decoder, WbfDecoder):  # wbf decoder
            # Use WBF only with AWGN since it assumes a specific relationship between priors and LLR
            # The more positive the reliability is, the less reliable the value is
            samples, priors = self._sample_and_priors(channel_llr,model_llr)
            estimate, decode_success, iterations, syndrome, vnode_validity = self.ldpc_decoder.decode(samples,
                                                                                                      priors)
            llr = np.array([])
        elif isinstance(self.ldpc_decoder, GalBfDecoder):
            samples, _ = self._sample_and_priors(channel_llr, model_llr)
            hard_channel_bits = np.array(samples < 0, dtype=np.int_)
            estimate, decode_success, iterations, syndrome, vnode_validity = self.ldpc_decoder.decode(
                hard_channel_bits)
            llr = np.array([])

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
        :param cluster_id:
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

    def model_prediction(self, observation_size: int, cluster_id: int) -> NDArray[np.float_]:
        def model_confidence(model_size: int, center: int, slope: float) -> np.float_:
            return 0 if model_size <= 1 else 1 / (1 + np.exp(-(model_size - center) * slope, dtype=np.float_))

        llr = np.zeros(observation_size, dtype=np.float_)
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
        # clipping = self.clipping_factor * max(llr)  # llr s clipped within +-clipping
        # model llr is calculated as log(Pr(c=0 | model) / Pr(c=1| model))
        # add model llr to the observation
        if confidence > 0:
            llr[structural_elements] += confidence * np.log(
                (np.finfo(np.float_).eps + self.distributions[cluster_id][:, 0]) / (
                        self.distributions[cluster_id][:, 1] + np.finfo(np.float_).eps)
            )[self.models_entropy[cluster_id] < self.entropy_threshold]

        return llr

    def _sample_and_priors(self, channel_sample: NDArray[np.float_], model_llr: NDArray[np.float_]):
        """The method returns a modified channel sample and priors depending on the method used and inputs"""
        max_sample = max(np.abs(channel_sample))
        samples = channel_sample
        priors = np.zeros_like(samples)
        if max(np.abs(model_llr)) > 0:
            if self.reliability_method == 0:  # sum LLRs and normalize
                # normalize model llr to channel sample, sum, and re-normalize
                if max(np.abs(model_llr)) > max_sample:
                    samples = channel_sample + model_llr*max_sample/max(np.abs(model_llr))
                else:
                    samples = channel_sample + model_llr
                samples *= max_sample/max(np.abs(samples))
                return samples, priors

            ## Methods 1&2 are shit. Don't USE!!! Use only method 0 until fixed.
            confidence_coefficient = 1  # when reliability_method==1
            # if self.reliability_method == 1:  # don't modify samples, use confidence_coefficient=1
            #     confidence_coefficient = 1
            if self.reliability_method == 2:  # confidence_coefficient=1
                rms_channel_sample = np.mean(np.power(channel_sample,2))**0.5
                mean_check_degree = np.sum(self.ldpc_decoder.h, axis=1).mean()
                confidence_coefficient = rms_channel_sample/mean_check_degree
            priors = -np.sign(channel_sample*model_llr) * np.abs(model_llr)
            if max(abs(priors)) > 0 and max(abs(priors)) > max_sample:  # normalize to samples
                priors *= max_sample/max(abs(priors))
            if max(abs(priors)) > 0 and max(abs(priors)) > confidence_coefficient:  # normalize to coefficient
                priors *= confidence_coefficient/max(abs(priors))
        return samples, priors

__all__ = ["ClassifyingEntropyDecoder"]
