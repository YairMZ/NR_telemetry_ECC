from decoders import Decoder, DecoderType
from inference import BufferSegmentation, MsgParts
from protocol_meta import dialect_meta as meta
from collections.abc import Sequence
from ldpc.decoder import LogSpaDecoder, bsc_llr
from numpy.typing import NDArray
import numpy as np
from typing import Optional
from utils.custom_exceptions import IncorrectBufferLength
from utils.information_theory import prob, entropy


class CombinedDecoder(Decoder):
    """
    This decoder creates a model of the data, classifying bits into structural and non-structural bits, based on their
    entropy across buffers.
    The decoder rectifies the channel llr per the model for structural bits.
    If a bit is deemed structural its llr is inferred from the model, and added to the channel llr.

    Additionally, segmentation is done to good and bad segments. Parts recognized as valid MAVLink messages are flagged as
    valid, and are assigned llr reflecting it.
    """

    def __init__(self, ldpc_decoder: LogSpaDecoder, model_length: int, entropy_threshold: float, clipping_factor: int,
                 feedback_iterations: int, stable_factor: float,
                 window_length: Optional[int] = None, a_conf_center: int = 20, a_conf_slope: float = 0.35,
                 b_conf_center: int = 40, b_conf_slope: float = 0.35, confidence: int = 0) -> None:
        """
        Create a new decoder
        :param ldpc_decoder: decoder for ldpc code used
        :param model_length: length of assumed model in bits, first info bits are assumed to be model bits
        :param entropy_threshold: threshold for entropy to dictate structural elements
        :param clipping_factor: the maximum model llr is equal to the clipping_factor times the maximal channel llr
        :param feedback_iterations: number of times segmentation is done.
        :param window_length: number of last messages to consider when evaluating distribution and entropy. If none all
        previous messages are considered.
        :param
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
        self.model_a_data: NDArray[np.uint8] = np.array([])  # 2d array, each column is a sample
        self.model_b_data: NDArray[np.uint8] = np.array([])  # 2d array, each column is a sample
        self.a_distribution: NDArray[np.float_] = np.array([])
        self.a_entropy: NDArray[np.float_] = np.array([])
        self.b_distribution: NDArray[np.float_] = np.array([])
        self.b_entropy: NDArray[np.float_] = np.array([])
        self.a_conf_center = a_conf_center
        self.a_conf_slope = a_conf_slope
        self.b_conf_center = b_conf_center
        self.b_conf_slope = b_conf_slope
        self.confidence = confidence
        self.feedback_iterations = feedback_iterations
        self.stable_factor = stable_factor
        super().__init__(DecoderType.COMBINED)

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
        channel_input: NDArray[np.float_] = np.array(channel_llr, dtype=np.float_)
        hard_channel_input: NDArray[np.int_] = np.array(channel_input < 0, dtype=np.int_)
        iterations_to_convergence = 0
        for _ in range(self.feedback_iterations):
            estimate, llr, decode_success, iterations, syndrome, vnode_validity = self.ldpc_decoder.decode(channel_input)
            iterations_to_convergence += iterations
            model_bits = estimate[self.model_bits_idx]
            model_bytes: bytes = np.packbits(model_bits).tobytes()
            msg_parts, validity, structure = self.segmentor.segment_buffer(model_bytes)
            if decode_success:
                if MsgParts.UNKNOWN not in msg_parts:  # buffer fully recovered
                    self.update_model_a(model_bits)
                self.update_model_b(model_bits)
                return estimate, llr, decode_success, iterations, len(structure)

            # use model to rectify llr
            channel_input = self.model_prediction(channel_input)
            # use mavlink to rectify llr
            good_bits = np.flatnonzero(np.repeat(msg_parts != MsgParts.UNKNOWN, 8))
            if good_bits.size > 0:
                max_llr = np.abs(channel_input).max()
                # if good bits are found, their values override previous estimates, with credibility based on stable_factor.
                # Good bits are get llr which is equal to the max llr times stable_factor.
                # The expression: 1 - 2*(estimate[good_bits] == 1  maps bits in estimate such that: 1 -> -1 and 0 -> 1.
                channel_input[good_bits] = self.stable_factor * max_llr * (1 - 2*(estimate[good_bits] == 1))

        return estimate, llr, decode_success, iterations, len(structure)

    def update_model_a(self, bits: NDArray[np.int_]) -> None:
        """update model of data. model_a uses only data which mavlink passed crc (and valid codeword)
        :param bits: hard estimate for bit values, assumed to be correct.
        """
        if len(bits) != self.model_length:
            raise IncorrectBufferLength()
        arr = bits[np.newaxis]
        self.model_a_data = arr.T if self.model_a_data.size == 0 else np.append(self.model_a_data, arr.T, axis=1)
        if (self.window_length is not None) and self.model_a_data.shape[1] > self.window_length:
            # trim old messages according to window
            self.model_a_data = self.model_a_data[:, self.model_a_data.shape[1] - self.window_length:]
        self.a_distribution = prob(self.model_a_data)
        self.a_entropy = entropy(self.a_distribution)

    def update_model_b(self, bits: NDArray[np.int_]) -> None:
        """update model of data. model_b uses any data regardless of correctness.
        :param bits: hard estimate for bit values, assumed to be correct.
        """
        if len(bits) != self.model_length:
            raise IncorrectBufferLength()
        arr = bits[np.newaxis]
        self.model_b_data = arr.T if self.model_b_data.size == 0 else np.append(self.model_b_data, arr.T, axis=1)
        if (self.window_length is not None) and self.model_b_data.shape[1] > self.window_length:
            # trim old messages according to window
            self.model_b_data = self.model_b_data[:, self.model_b_data.shape[1] - self.window_length:]
        self.b_distribution = prob(self.model_b_data)
        self.b_entropy = entropy(self.b_distribution)

    def model_prediction(self, observation: NDArray[np.float_]) -> NDArray[np.float_]:
        """Responsible for making predictions regarding originally sent data, based on recent observations and model.
        If sufficient data exists, the llr is computed based on the model, and is added to the observation.
        :param llr: recent observation regrading which a prediction is required.
        :return: an array of llr based on model predictions
        """

        def model_confidence(model_size: int, center: int, slope: float) -> np.float_:
            return 0 if model_size <= 1 else 1 / (1 + np.exp(-(model_size - center) * slope, dtype=np.float_))

        llr = observation.copy()
        # infer structure
        # index of structural (low entropy) elements among codeword
        a_structural_elements: NDArray[np.int_] = self.model_bits_idx[self.a_entropy < self.entropy_threshold]
        b_structural_elements: NDArray[np.int_] = self.model_bits_idx[self.b_entropy < self.entropy_threshold]
        # model llr is calculated as log(Pr(c=0 | model) / Pr(c=1| model))
        a_size = self.model_a_data.shape[1] if self.model_a_data.size > 0 else 0
        b_size = self.model_b_data.shape[1] if self.model_b_data.size > 0 else 0
        a_confidence = model_confidence(a_size, self.a_conf_center, self.a_conf_slope)
        b_confidence = model_confidence(b_size, self.b_conf_center,
                                        self.b_conf_slope)  # consider window size when setting these

        clipping = self.clipping_factor * max(llr)  # llr s clipped within +-clipping

        if self.confidence == 0:
            pass  # use separate confidence measures
        elif self.confidence == 1:  # normalize sum of confidence to unity
            s = a_confidence + b_confidence
            if s > 0:
                a_confidence *= a_confidence / s
                b_confidence *= b_confidence / s
        elif self.confidence == 1:  # normalize sum but prefer "good" model
            s = 2 * a_confidence + b_confidence
            if s > 0:
                a_confidence *= 2 * a_confidence / s
                b_confidence *= b_confidence / s
        elif self.confidence == 3:  # ignore bad model
            b_confidence = 0
        elif self.confidence == 3:  # use predetermined clipping
            clipping = self.clipping_factor

        # add model llr to the observation
        if a_confidence > 0:
            llr[a_structural_elements] += a_confidence * np.log(
                (np.finfo(np.float_).eps + self.a_distribution[:, 0]) / (self.a_distribution[:, 1] + np.finfo(np.float_).eps)
            )[self.a_entropy < self.entropy_threshold]
        if b_confidence > 0:
            llr[b_structural_elements] += b_confidence * np.log(
                (np.finfo(np.float_).eps + self.b_distribution[:, 0]) / (self.b_distribution[:, 1] + np.finfo(np.float_).eps)
            )[self.b_entropy < self.entropy_threshold]

        return np.clip(llr, -clipping, clipping)
