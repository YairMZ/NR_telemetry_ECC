from decoders import Decoder, DecoderType
from inference import BufferSegmentation, MsgParts
from protocol_meta import dialect_meta as meta
from collections.abc import Sequence
from ldpc.decoder import LogSpaDecoder
from numpy.typing import NDArray
import numpy as np
from typing import Optional
from utils.custom_exceptions import IncorrectBufferLength
from utils.information_theory import prob, entropy


class EntropyBitwiseDecoder(Decoder):
    """
    This decoder creates a model of the data, classifying them into structural and non-structural bits, based on their
    entropy across buffers.
    The decoder rectifies the channel llr per the model for structural bits.
    If a bit is deemed structural its llr is inferred from the model, and added to the channel llr.
    """

    def __init__(self, ldpc_decoder: LogSpaDecoder, model_length: int, entropy_threshold: float, clipping_factor: int,
                 min_data: int, window_length: Optional[int] = None, model: Optional[NDArray[np.float_]] = None) -> None:
        """
        Create a new decoder
        :param ldpc_decoder: decoder for ldpc code used
        :param model_length: length of assumed model in bits, first info bits are assumed to be model bits
        :param entropy_threshold: threshold for entropy to dictate structural elements
        :param clipping_factor: the maximum model llr is equal to the clipping_factor times the maximal channel llr
        :param min_data: the minimum amount of good buffers to be used in the learning stage before attempting to rectify llr
        :param window_length: number of last messages to consider when evaluating distribution and entropy. If none all
        :param model: if not None, holds a distribution of bits instead of learning it.
        previous messages are considered.
        """
        self.segmentor: BufferSegmentation = BufferSegmentation(meta.protocol_parser)
        self.ldpc_decoder: LogSpaDecoder = ldpc_decoder
        self.model_length: int = model_length  # in bits
        self.entropy_threshold = entropy_threshold
        self.model_bits_idx = np.array(self.ldpc_decoder.info_idx)
        self.model_bits_idx[model_length:] = False
        self.model_bits_idx = np.flatnonzero(self.model_bits_idx)  # bit indices (among codeword bits) of model bits
        self.clipping_factor = clipping_factor  # The model llr is clipped to +-clipping_factor * max_chanel_llr
        self.window_length = window_length
        if model is not None:
            self.model_length = min(model.shape[0], self.model_bits_idx.size)
            self.distribution: NDArray[np.float_] = model[:self.model_length, :]   # estimated distribution model
            self.entropy: NDArray[np.float_] = entropy(self.distribution)  # estimated entropy of distribution model
            self.predefined_model = True
        else:
            self.distribution = np.array([])
            self.entropy = np.array([])
            self.predefined_model = False
        self.model_data: NDArray[np.uint8] = np.array([])  # 2d array, each column is a sample
        self.structural_elements: NDArray[np.int_] = np.array([])  # index of structural (low entropy) elements among codeword
        self.min_data = min_data  # minimum amount of good buffers used in the learning stage before attempting to rectify llr
        super().__init__(DecoderType.ENTROPY)

    def decode_buffer(self, channel_llr: Sequence[np.float_]) -> tuple[NDArray[np.int_], NDArray[np.float_], bool, int, int]:
        """decodes a buffer
        :param channel_llr: channel llr of bits to decode
        :return: return a tuple (estimated_bits, llr, decode_success, no_iterations, no of mavlink messages found)
        where:
            - estimated_bits is a 1-d np array of hard bit estimates
            - llr is a 1-d np array of soft bit estimates
            - decode_success is a boolean flag stating of the estimated_bits form a valid  code word
            - number of iterations until breaking
            - number of MAVLink messages found within buffer
        """
        estimate, llr, decode_success, iterations, syndrome, vnode_validity = self.ldpc_decoder.decode(channel_llr)
        if decode_success:
            model_bits = estimate[self.model_bits_idx]
            model_bytes: bytes = np.packbits(model_bits).tobytes()
            msg_parts, validity, structure = self.segmentor.segment_buffer(model_bytes)
            if MsgParts.UNKNOWN not in msg_parts:  # buffer fully recovered
                self.update_model(model_bits)
            return estimate, llr, decode_success, iterations, len(structure)

        # rectify llr
        model_llr = self.model_prediction(channel_llr)  # type: ignore
        estimate, llr, decode_success, iterations, syndrome, vnode_validity = self.ldpc_decoder.decode(model_llr)
        model_bits = estimate[self.model_bits_idx]
        model_bytes = np.packbits(model_bits).tobytes()
        msg_parts, validity, structure = self.segmentor.segment_buffer(model_bytes)
        if MsgParts.UNKNOWN not in msg_parts:  # buffer fully recovered
            self.update_model(model_bits)
        return estimate, llr, decode_success, iterations, len(structure)

    def update_model(self, bits: NDArray[np.int_]) -> None:
        """update model of data
        :param bits: hard estimate for bit values, assumed to be correct.
        """
        if self.predefined_model:
            return
        if len(bits) != self.model_length:
            raise IncorrectBufferLength()
        arr = bits[np.newaxis]
        self.model_data = arr.T if self.model_data.size == 0 else np.append(self.model_data, arr.T, axis=1)
        if (self.window_length is not None) and self.model_data.shape[1] > self.window_length:
            # trim old messages according to window
            self.model_data = self.model_data[:, self.model_data.shape[1] - self.window_length:]
        self.distribution = prob(self.model_data)
        self.entropy = entropy(self.distribution)

    def model_prediction(self, observation: NDArray[np.float_]) -> NDArray[np.float_]:
        """Responsible for making predictions regarding originally sent data, based on recent observations and model.
        If sufficient data exists, the llr is computed based on the model, and is added to the observation.
        :param observation: recent observation regrading which a prediction is required.
        :return: an array of llr based on model predictions
        """
        # infer structure
        # index of structural (low entropy) elements among codeword
        self.structural_elements = self.model_bits_idx[self.entropy < self.entropy_threshold]
        # model llr is calculated as log(Pr(c=0 | model) / Pr(c=1| model))
        llr = observation.copy()
        if self.predefined_model or (self.model_data.size > 0 and self.model_data.shape[1] >= self.min_data):
            # if sufficient previous data exists
            clipping = self.clipping_factor * max(llr)  # llr s clipped within +-clipping
            llr[self.structural_elements] += np.clip(  # add model llr to the observation
                np.log(
                    (np.finfo(np.float_).eps + self.distribution[:, 0])/(self.distribution[:, 1] + np.finfo(np.float_).eps)
                ),
                -clipping, clipping)[self.entropy < self.entropy_threshold]
        return llr


class EntropyBitwiseFlippingDecoder(Decoder):
    """
    This decoder creates a model of the data, classifying them into structural and non-structural bits, based on their
    entropy across buffers.
    The decoder receives hard input, and flips structural bits if needed.
    """

    def __init__(self, ldpc_decoder: LogSpaDecoder, model_length: int, entropy_threshold: float, min_data: int,
                 window_length: Optional[int] = None, model: Optional[NDArray[np.float_]] = None) -> None:
        """
        Create a new decoder
        :param ldpc_decoder: decoder for ldpc code used
        :param model_length: length of assumed model in bits, first info bits are assumed to be model bits
        :param entropy_threshold: threshold for entropy to dictate structural elements
        :param min_data: the minimum amount of good buffers to be used in the learning stage before attempting to rectify llr
        :param window_length: number of last messages to consider when evaluating distribution and entropy. If none all
        previous messages are considered.
        :param model: if not None, holds a distribution of bits instead of learning it.
        """
        self.segmentor: BufferSegmentation = BufferSegmentation(meta.protocol_parser)
        self.ldpc_decoder: LogSpaDecoder = ldpc_decoder
        self.model_length: int = model_length  # in bits
        self.entropy_threshold = entropy_threshold
        self.model_bits_idx = np.array(self.ldpc_decoder.info_idx)
        self.model_bits_idx[model_length:] = False
        self.model_bits_idx = np.flatnonzero(self.model_bits_idx)  # bit indices (among codeword bits) of model bits
        self.window_length = window_length
        if model is not None:
            self.model_length = min(model.shape[0], self.model_bits_idx.size)
            self.distribution: NDArray[np.float_] = model[:self.model_length, :]   # estimated distribution model
            self.entropy: NDArray[np.float_] = entropy(self.distribution)  # estimated entropy of distribution model
            self.predefined_model = True
        else:
            self.distribution = np.array([])
            self.entropy = np.array([])
            self.predefined_model = False
        self.model_data: NDArray[np.uint8] = np.array([])  # 2d array, each column is a sample
        self.structural_elements: NDArray[np.int_] = np.array([])  # index of structural (low entropy) elements among codeword
        self.structural_values: NDArray[np.int_] = np.array([])  # value of structural (low entropy) elements among codeword
        self.min_data = min_data  # minimum amount of good buffers used in the learning stage before attempting to rectify llr
        super().__init__(DecoderType.ENTROPY)

    def decode_buffer(self, channel_word: Sequence[np.float_]) -> tuple[NDArray[np.int_], NDArray[np.float_], bool, int, int]:
        """decodes a buffer
        :param channel_word: channel of hard bits to decode
        :return: return a tuple (estimated_bits, llr, decode_success, no_iterations, no of mavlink messages found)
        where:
            - estimated_bits is a 1-d np array of hard bit estimates
            - llr is a 1-d np array of soft bit estimates
            - decode_success is a boolean flag stating of the estimated_bits form a valid  code word
            - number of iterations until breaking
            - number of MAVLink messages found within buffer
        """
        estimate, llr, decode_success, iterations, syndrome, vnode_validity = self.ldpc_decoder.decode(channel_word)
        if decode_success:
            model_bits = estimate[self.model_bits_idx]
            model_bytes: bytes = np.packbits(model_bits).tobytes()
            msg_parts, validity, structure = self.segmentor.segment_buffer(model_bytes)
            if MsgParts.UNKNOWN not in msg_parts:  # buffer fully recovered
                self.update_model(model_bits)
            return estimate, llr, decode_success, iterations, len(structure)

        # use model
        model_word = self.model_prediction(channel_word)  # type: ignore
        estimate, llr, decode_success, iterations, syndrome, vnode_validity = self.ldpc_decoder.decode(model_word)
        model_bits = estimate[self.model_bits_idx]
        model_bytes = np.packbits(model_bits).tobytes()
        msg_parts, validity, structure = self.segmentor.segment_buffer(model_bytes)
        if MsgParts.UNKNOWN not in msg_parts:  # buffer fully recovered
            self.update_model(model_bits)
        return estimate, llr, decode_success, iterations, len(structure)

    def update_model(self, bits: NDArray[np.int_]) -> None:
        """update model of data
        :param bits: hard estimate for bit values, assumed to be correct.
        """
        if self.predefined_model:
            return
        if len(bits) != self.model_length:
            raise IncorrectBufferLength()
        arr = bits[np.newaxis]
        self.model_data = arr.T if self.model_data.size == 0 else np.append(self.model_data, arr.T, axis=1)
        if (self.window_length is not None) and self.model_data.shape[1] > self.window_length:
            # trim old messages according to window
            self.model_data = self.model_data[:, self.model_data.shape[1] - self.window_length:]
        self.distribution = prob(self.model_data)
        self.entropy = entropy(self.distribution)

    def model_prediction(self, observation: NDArray[np.int_]) -> NDArray[np.int_]:
        """Responsible for making predictions regarding originally sent data, based on recent observations and model.
        If sufficient data exists, the llr is computed based on the model, and is added to the observation.
        :param observation: recent observation regrading which a prediction is required.
        :return: an array of llr based on model predictions
        """
        if (not self.predefined_model) and (self.model_data.size <= 0 or self.model_data.shape[1] < self.min_data):
            return observation
        # infer structure
        # index of structural (low entropy) elements among codeword
        self.structural_elements = self.model_bits_idx[self.entropy < self.entropy_threshold]
        # round the probability of Pr(c=1)
        self.structural_values = np.array(self.distribution[:, 1] > 0.5, dtype=np.int_)[self.structural_elements]
        prediction = np.array(observation, dtype=np.int_)
        prediction[self.structural_elements] = self.structural_values
        return prediction


class EntropyBitwiseWeightedDecoder(Decoder):
    """
    This decoder creates a model of the data, classifying them into structural and non-structural bits, based on their
    entropy across buffers.
    The decoder rectifies the channel llr per the model for structural bits.
    If a bit is deemed structural its llr is inferred from the model, and added to the channel llr.

    The decoder is similar to EntropyBitwiseDecoder, except that it trains both on good and bad data.
    When enough good data exists the model based on bad data is discarded.
    """

    def __init__(self, ldpc_decoder: LogSpaDecoder, model_length: int, entropy_threshold: float, clipping_factor: float,
                 window_length: Optional[int] = None, a_conf_center: int = 20, a_conf_slope: float = 0.35,
                 b_conf_center: int = 40, b_conf_slope: float = 0.35, confidence: int = 0,
                 estimator: str = "MLE", bit_flip: float = 0, corrected_dist: bool = False) -> None:
        """
        Create a new decoder
        :param ldpc_decoder: decoder for ldpc code used
        :param model_length: length of assumed model in bits, first info bits are assumed to be model bits
        :param entropy_threshold: threshold for entropy to dictate structural elements
        :param clipping_factor: the maximum model llr is equal to the clipping_factor times the maximal channel llr
        :param window_length: number of last messages to consider when evaluating distribution and entropy. If none all
        """
        self.segmentor: BufferSegmentation = BufferSegmentation(meta.protocol_parser)
        self.ldpc_decoder: LogSpaDecoder = ldpc_decoder
        self.model_length: int = model_length  # in bits
        self.entropy_threshold = entropy_threshold
        self.model_bits_idx = np.array(self.ldpc_decoder.info_idx)
        self.model_bits_idx[model_length:] = False
        self.model_bits_idx = np.flatnonzero(self.model_bits_idx)  # bit indices (among codeword bits) of model bits
        self.clipping_factor = clipping_factor  # The model llr is clipped to +-clipping_factor * max_chanel_llr

        self.a_distribution: NDArray[np.float_] = np.array([])
        self.a_entropy: NDArray[np.float_] = np.array([])
        self.b_distribution: NDArray[np.float_] = np.array([])
        self.b_entropy: NDArray[np.float_] = np.array([])
        self.a_conf_center = a_conf_center
        self.a_conf_slope = a_conf_slope
        self.b_conf_center = b_conf_center
        self.b_conf_slope = b_conf_slope
        self.confidence_scheme = confidence
        self.estimator = estimator
        self.bit_flip = bit_flip
        self.corrected_dist = corrected_dist
        if estimator == "Bayes":
            self.a_ones_prior = np.ones(len(self.model_bits_idx), dtype=np.int_)
            self.a_zeros_prior = np.ones(len(self.model_bits_idx), dtype=np.int_)
            self.b_ones_prior = np.ones(len(self.model_bits_idx), dtype=np.int_)
            self.b_zeros_prior = np.ones(len(self.model_bits_idx), dtype=np.int_)
            self.a_normalized_variance = 0.0
            self.b_normalized_variance = 0.0
            self.min_size = window_length
            self.max_size = 2*window_length
        else:  # use MLE
            self.window_length = window_length
            self.model_a_data: NDArray[np.uint8] = np.array([])  # 2d array, each column is a sample
            self.model_b_data: NDArray[np.uint8] = np.array([])  # 2d array, each column is a sample
        super().__init__(DecoderType.ENTROPY)

    def decode_buffer(self, channel_llr: Sequence[np.float_]) -> tuple[NDArray[np.int_], NDArray[np.float_], bool, int,
                                                                       NDArray[np.int_], NDArray[np.int_], int,
                                                                       NDArray[np.float_], NDArray[np.float_],
                                                                       NDArray[np.int_], NDArray[np.int_]]:
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
            - inferred distribution of "a_model"
            - inferred distribution of "b_model"
            - inferred structural bits of "a_model"
            - inferred structural bits of "b_model"
        """
        estimate, llr, decode_success, iterations, syndrome, vnode_validity = self.ldpc_decoder.decode(channel_llr)
        if decode_success:
            model_bits = estimate[self.model_bits_idx]
            model_bytes: bytes = np.packbits(model_bits).tobytes()
            msg_parts, validity, structure = self.segmentor.segment_buffer(model_bytes)
            if MsgParts.UNKNOWN not in msg_parts:  # buffer fully recovered
                self.update_model_a(model_bits)
            self.update_model_b(model_bits)

            return estimate, llr, decode_success, iterations, syndrome, vnode_validity, len(structure), \
                   self.a_distribution, self.b_distribution, \
                   self.model_bits_idx[self.a_entropy < self.entropy_threshold], \
                   self.model_bits_idx[self.b_entropy < self.entropy_threshold]

        # rectify llr
        model_llr = self.model_prediction(channel_llr)  # type: ignore
        estimate, llr, decode_success, iterations, syndrome, vnode_validity = self.ldpc_decoder.decode(model_llr)
        model_bits = estimate[self.model_bits_idx]
        model_bytes = np.packbits(model_bits).tobytes()
        msg_parts, validity, structure = self.segmentor.segment_buffer(model_bytes)
        if MsgParts.UNKNOWN not in msg_parts:  # buffer fully recovered
            self.update_model_a(model_bits)
            self.update_model_b(model_bits)
        else:  # update model from channel if data is bad
            channel_bits = np.array(channel_llr < 0, dtype=np.int_)[self.model_bits_idx]
            self.update_model_b(channel_bits)
        return estimate, llr, decode_success, iterations, syndrome, vnode_validity, len(structure), \
               self.a_distribution, self.b_distribution, self.model_bits_idx[self.a_entropy < self.entropy_threshold], \
               self.model_bits_idx[self.b_entropy < self.entropy_threshold]

    def update_model_a(self, bits: NDArray[np.int_]) -> None:
        """update model of data. model_a uses only data which mavlink passed crc (and valid codeword)
        :param bits: hard estimate for bit values, assumed to be correct.
        """
        if len(bits) != self.model_length:
            raise IncorrectBufferLength()
        arr = bits[np.newaxis]

        if self.estimator == "Bayes":
            self.a_ones_prior += bits
            self.a_zeros_prior += 1 - bits
            # to allow for changes in data divide by common divisor
            if self.a_ones_prior[0] + self.a_zeros_prior[0] > self.max_size:
                gcd = np.gcd(self.a_ones_prior, self.a_zeros_prior)
                self.a_ones_prior //= gcd
                self.a_zeros_prior //= gcd
                if self.a_ones_prior[0] + self.a_zeros_prior[0] < self.min_size:
                    m = self.min_size // self.a_ones_prior[0] + self.a_zeros_prior[0]
                    self.a_ones_prior *= m
                    self.a_zeros_prior *= m

            # estimate v using mode of beta variable
            v = np.nan_to_num((self.a_ones_prior - 1) / (self.a_ones_prior + self.a_zeros_prior - 2))
            p1 = np.clip((v - self.bit_flip) / (1 - 2 * self.bit_flip), 0, 1)
            # find  th mean variance of estimates as confidence measure
            var = ((1 / (1 - 2 * self.bit_flip)) ** 2) * np.mean(self.a_ones_prior * self.a_zeros_prior / (
                    (self.a_ones_prior + self.a_zeros_prior + 1) * np.power(self.a_ones_prior + self.a_zeros_prior, 2)))
            # for a uniform variable over the domain [0,1] (if we assumed uniform p) variance would be 1/12.
            # Thus, we measure the variance with respect to a uniform distribution in percentage
            self.a_normalized_variance = 12 * var  # percentage of uniform variance
            self.a_distribution = np.column_stack((1 - p1, p1))
        else:  # Use MLE
            self.model_a_data = arr.T if self.model_a_data.size == 0 else np.append(self.model_a_data, arr.T, axis=1)
            if (self.window_length is not None) and self.model_a_data.shape[1] > self.window_length:
                # trim old messages according to window
                self.model_a_data = self.model_a_data[:, self.model_a_data.shape[1] - self.window_length:]
            self.a_distribution = prob(self.model_a_data)
            if self.corrected_dist:
                v = self.a_distribution[:, 1]
                p1 = np.clip((v - self.bit_flip) / (1 - 2 * self.bit_flip), 0, 1)
                self.a_distribution = np.column_stack((1 - p1, p1))
            else:
                p1 = self.a_distribution[:, 1]
            # mu equals p1
            # for a Bernoulli variable the variance p(1-p), which maximizes at 1/2.
            # Thus, we measure the variance with respect to a uniform distribution in percentage
            self.a_normalized_variance = 2 * np.mean(p1 - np.power(p1, 2))
        if max(p1) > 1 or min(p1) < 0:
            print("problematic probability")
            raise RuntimeError("problematic probability")
        self.a_entropy = entropy(self.a_distribution)

    def update_model_b(self, bits: NDArray[np.int_]) -> None:
        """update model of data. model_b uses any data regardless of correctness.
        :param bits: hard estimate for bit values, assumed to be correct.
        """
        if len(bits) != self.model_length:
            raise IncorrectBufferLength()
        arr = bits[np.newaxis]

        if self.estimator == "Bayes":
            self.b_ones_prior += bits
            self.b_zeros_prior += 1 - bits
            # to allow for changes in data divide by common divisor
            if self.b_ones_prior[0] + self.b_zeros_prior[0] > self.max_size:
                gcd = np.gcd(self.b_ones_prior, self.b_zeros_prior)
                self.b_ones_prior //= gcd
                self.b_zeros_prior //= gcd
                if self.b_ones_prior[0] + self.b_zeros_prior[0] < self.min_size:
                    m = self.min_size // self.b_ones_prior[0] + self.b_zeros_prior[0]
                    self.b_ones_prior *= m
                    self.b_zeros_prior *= m
            # estimate v using mode of beta variable
            v = np.nan_to_num((self.b_ones_prior - 1) / (self.b_ones_prior + self.b_zeros_prior - 2))

            p1 = np.clip((v - self.bit_flip) / (1 - 2 * self.bit_flip), 0, 1)
            # find th mean variance of estimates as confidence measure
            var = ((1 / (1 - 2 * self.bit_flip)) ** 2) * np.mean(self.b_ones_prior * self.b_zeros_prior / (
                    (self.b_ones_prior + self.b_zeros_prior + 1) * np.power(self.b_ones_prior + self.b_zeros_prior, 2)))
            # for a uniform continuous variable over the domain [0,1] (if we assumed uniform p) variance would be 1/12.
            # Thus, we measure the variance with respect to a uniform distribution in percentage
            self.b_normalized_variance = 12 * var  # percentage of uniform variance
            self.b_distribution = np.column_stack((1 - p1, p1))
        else:  # Use MLE
            self.model_b_data = arr.T if self.model_b_data.size == 0 else np.append(self.model_b_data, arr.T, axis=1)
            if (self.window_length is not None) and self.model_b_data.shape[1] > self.window_length:
                # trim old messages according to window
                self.model_b_data = self.model_b_data[:, self.model_b_data.shape[1] - self.window_length:]
            self.b_distribution = prob(self.model_b_data)
            if self.corrected_dist:
                v = self.b_distribution[:, 1]
                p1 = np.clip((v - self.bit_flip) / (1 - 2 * self.bit_flip), 0, 1)
                self.b_distribution = np.column_stack((1 - p1, p1))
            else:
                p1 = self.b_distribution[:, 1]
            # mu equals p1
            # for a Bernoulli variable the variance p(1-p), which maximizes at 1/2.
            # Thus, we measure the variance with respect to a uniform distribution in percentage
            self.b_normalized_variance = 2 * np.mean(p1 - np.power(p1, 2))
        if max(p1) > 1 or min(p1) < 0:
            print("problematic probability")
            raise RuntimeError("problematic probability")
        self.b_entropy = entropy(self.b_distribution)

    def model_prediction(self, observation: NDArray[np.float_]) -> NDArray[np.float_]:
        """Responsible for making predictions regarding originally sent data, based on recent observations and model.
        If sufficient data exists, the llr is computed based on the model, and is added to the observation.
        :param llr: recent observation regrading which a prediction is required.
        :return: an array of llr based on model predictions
        """
        def model_confidence(model_size: int, center: int, slope: float) -> np.float_:
            return 0 if model_size <= 1 else 1/(1+np.exp(-(model_size-center)*slope, dtype=np.float_))

        llr = observation.copy()
        # infer structure
        # index of structural (low entropy) elements among codeword
        a_structural_elements: NDArray[np.int_] = self.model_bits_idx[self.a_entropy < self.entropy_threshold]
        b_structural_elements: NDArray[np.int_] = self.model_bits_idx[self.b_entropy < self.entropy_threshold]

        if (not a_structural_elements.any()) and (not b_structural_elements.any()):  # no structural elements found
            return llr

        if self.estimator == "Bayes":
            a_size = self.a_ones_prior[0] + self.a_zeros_prior[0] - 2
            b_size = self.b_ones_prior[0] + self.b_zeros_prior[0] - 2
        else:   # Use MLE
            a_size = self.model_a_data.shape[1] if self.model_a_data.size > 0 else 0
            b_size = self.model_b_data.shape[1] if self.model_b_data.size > 0 else 0

        a_confidence = model_confidence(a_size, self.a_conf_center, self.a_conf_slope)
        b_confidence = model_confidence(b_size, self.b_conf_center, self.b_conf_slope)  # consider window size when setting these

        if self.confidence_scheme == 0:
            pass  # use separate confidence measures
        elif self.confidence_scheme == 1:  # normalize sum of confidence to unity
            s = a_confidence + b_confidence
            if s > 0:
                a_confidence *= a_confidence / s
                b_confidence *= b_confidence / s
        elif self.confidence_scheme == 1:  # normalize sum but prefer "good" model
            s = 2*a_confidence + b_confidence
            if s > 0:
                a_confidence *= 2 * a_confidence / s
                b_confidence *= b_confidence / s
        elif self.confidence_scheme == 3:  # ignore bad model
            b_confidence = 0
        elif self.confidence_scheme == 4:  # scale using variance
            a_confidence *= 1 - np.exp(-0.1 / self.a_normalized_variance)
            b_confidence *= 1 - np.exp(-0.1 / self.b_normalized_variance)
        elif self.confidence_scheme == 5:  # ignore good model
            a_confidence = 0

        clipping = self.clipping_factor * max(llr)  # llr s clipped within +-clipping
        # model llr is calculated as log(Pr(c=0 | model) / Pr(c=1| model))
        # add model llr to the observation
        if a_confidence > 0:
            llr[a_structural_elements] += a_confidence*np.log(
                (np.finfo(np.float_).eps + self.a_distribution[:, 0])/(self.a_distribution[:, 1] + np.finfo(np.float_).eps)
            )[self.a_entropy < self.entropy_threshold]
        if b_confidence > 0:
            llr[b_structural_elements] += b_confidence*np.log(
                (np.finfo(np.float_).eps + self.b_distribution[:, 0]) / (self.b_distribution[:, 1] + np.finfo(np.float_).eps)
            )[self.b_entropy < self.entropy_threshold]

        return np.clip(llr, -clipping, clipping)


__all__ = ["EntropyBitwiseDecoder", "EntropyBitwiseFlippingDecoder", "EntropyBitwiseWeightedDecoder"]
