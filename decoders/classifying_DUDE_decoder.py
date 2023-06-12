from decoders import Decoder, DecoderType
from inference import BufferClassifier, OnlineDude
from collections.abc import Sequence
from ldpc.decoder import LogSpaDecoder
from numpy.typing import NDArray
import numpy as np

class ClassifyingDudeDecoder(Decoder):
    def __init__(self, ldpc_decoder: LogSpaDecoder, model_length: int, clipping_factor: float,
                 classifier_training: int, n_clusters: int, context_length: int,
                 conf_center: int = 40, conf_slope: float = 0.35,
                 bit_flip: float = 0, cluster: int = 1, entropy_threshold: float = 0.5
                 ) -> None:
        """
        Create a new decoder
        :param ldpc_decoder: decoder for ldpc code used
        :param model_length: length of assumed model in bits, first info bits are assumed to be model bits
        :param clipping_factor: the maximum model llr is equal to the clipping_factor times the maximal channel llr
        :param classifier_training: number of buffers to use for training the classifier
        :param n_clusters: number of clusters to use in the classifier
        :param context_length: length of context to use in the DUDE
        :param conf_center: center of confidence interval
        :param conf_slope: slope of confidence interval
        :param bit_flip: probability of flipping a bit in the buffer
        :param cluster: should we use clustering or not
        :param entropy_threshold: threshold for entropy to dictate structural elements
        """
        self.ldpc_decoder: LogSpaDecoder = ldpc_decoder
        self.model_length: int = model_length  # in bits
        self.model_bits_idx = np.array(self.ldpc_decoder.info_idx)
        if model_length >= sum(self.model_bits_idx):
            self.model_bits_idx = np.array(range(model_length))
        else:
            self.model_bits_idx[model_length:] = False
            self.model_bits_idx = np.flatnonzero(self.model_bits_idx)  # bit indices (among codeword bits) of model bits

        self.clipping_factor = clipping_factor  # The model llr is clipped to +-clipping_factor * max_chanel_llr

        self.bit_flip = bit_flip
        channel_transition_matrix = np.array([[1-bit_flip, bit_flip], [bit_flip, 1-bit_flip]])
        loss_matrix = np.array([[0, 1], [1, 0]])
        self.dudes: list[list[OnlineDude]] = [[OnlineDude(channel_transition_matrix, loss_matrix, context_length,
                                                          hard_decision=False)
                                               for _ in range(model_length)] for _ in range(n_clusters)]
        self.data_size: list[int] = [0 for _ in range(n_clusters)]
        # self.entropy_threshold = entropy_threshold
        # self.distributions: list[NDArray[np.float_]] = [np.array([]) for _ in range(n_clusters)]
        # self.models_entropy: list[NDArray[np.float_]] = [np.array([]) for _ in range(n_clusters)]

        self.conf_center = conf_center
        self.conf_slope = conf_slope

        self.cluster = bool(cluster)
        if self.cluster:
            self.classifier = BufferClassifier(classifier_training, n_clusters)
        else:
            self.classifier = -1
        self.n_clusters = n_clusters
        super().__init__(DecoderType.DUDE)

    def decode_buffer(self, channel_llr: Sequence[np.float_]) -> tuple[NDArray[np.int_], NDArray[np.float_], bool, int, int,
    NDArray[np.int_]]:
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
                    return estimate, llr, decode_success, iterations, label, np.array(channel_llr < 0, dtype=np.int_)
            else:
                self.classifier = (self.classifier + 1) % self.n_clusters
                label = self.classifier
        else:
            label = 0
        self.data_size[label] += 1
        # estimate, llr, decode_success, iterations, syndrome, vnode_validity = self.ldpc_decoder.decode(channel_llr)
        # if decode_success:  # buffer fully recovered
        #     model_bits = estimate[self.model_bits_idx]
        #     self.update_model(model_bits, label)
        #     return estimate, llr, decode_success, iterations, syndrome, vnode_validity, self.distributions[label], \
        #         self.model_bits_idx[self.models_entropy[label] < self.entropy_threshold], label
        clipping = self.clipping_factor * max(np.abs(channel_llr))  # llr s clipped within +-clipping
        model_llr = self.model_prediction(channel_bits, label)  # type: ignore
        channel_llr[self.model_bits_idx] += model_llr
        channel_llr = np.clip(channel_llr, -clipping, clipping)
        denoised_bits = np.array(channel_llr < 0, dtype=np.int_)
        estimate, llr, decode_success, iterations, syndrome, vnode_validity = self.ldpc_decoder.decode(channel_llr)
        model_bits = estimate[self.model_bits_idx]
        # if decode_success:  # buffer fully recovered
        #     self.update_model(model_bits, label)
        # else:  # update model from channel if data is bad
        #     self.update_model(channel_bits, label)
        return estimate, llr, decode_success, iterations, label, denoised_bits

    def model_prediction(self, observation: NDArray[np.int_], cluster_id: int) -> NDArray[np.float_]:
        def model_confidence(model_size: int, center: int, slope: float) -> np.float_:
            return 0 if model_size <= 1 else 1 / (1 + np.exp(-(model_size - center) * slope, dtype=np.float_))

        llr = observation.copy()
        confidence = model_confidence(self.data_size[cluster_id], self.conf_center, self.conf_slope)
        clipping = self.clipping_factor * max(llr)  # llr s clipped within +-clipping
        # model llr is calculated as log(Pr(c=0 | model) / Pr(c=1| model))
        # For soft decision DUDES
        p1 = np.array([dude.denoise_sample(observation[i])[0] for i, dude in enumerate(self.dudes[cluster_id])])[:,:,1].flatten()
        # For hard decision DUDES
        # p1 = np.array([dude.denoise_sample(observation[i])[0]for i, dude in enumerate(self.dudes[cluster_id])])
        # compute model llr to the observation
        model_llr = confidence * np.log((1 - p1 + np.finfo(np.float_).eps) / (p1 + np.finfo(np.float_).eps))

        return model_llr


__all__ = ["ClassifyingDudeDecoder"]
