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
            self.distribution: NDArray[np.float_] = model   # estimated distribution model
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
                 window_length: Optional[int] = None) -> None:
        """
        Create a new decoder
        :param ldpc_decoder: decoder for ldpc code used
        :param model_length: length of assumed model in bits, first info bits are assumed to be model bits
        :param entropy_threshold: threshold for entropy to dictate structural elements
        :param min_data: the minimum amount of good buffers to be used in the learning stage before attempting to rectify llr
        :param window_length: number of last messages to consider when evaluating distribution and entropy. If none all
        previous messages are considered.
        """
        self.segmentor: BufferSegmentation = BufferSegmentation(meta.protocol_parser)
        self.ldpc_decoder: LogSpaDecoder = ldpc_decoder
        self.model_length: int = model_length  # in bits
        self.entropy_threshold = entropy_threshold
        self.model_bits_idx = np.array(self.ldpc_decoder.info_idx)
        self.model_bits_idx[model_length:] = False
        self.model_bits_idx = np.flatnonzero(self.model_bits_idx)  # bit indices (among codeword bits) of model bits
        self.window_length = window_length
        self.model_data: NDArray[np.uint8] = np.array([])  # 2d array, each column is a sample
        self.distribution: NDArray[np.float_] = np.array([])  # estimated distribution model
        self.entropy: NDArray[np.float_] = np.array([])  # estimated entropy of distribution model
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
        if self.model_data.size <= 0 or self.model_data.shape[1] < self.min_data:
            return observation
        # infer structure
        # index of structural (low entropy) elements among codeword
        self.structural_elements = self.model_bits_idx[self.entropy < self.entropy_threshold]
        # round the probability of Pr(c=1)
        self.structural_values = np.array(self.distribution[:, 1] > 0.5, dtype=np.int_)[self.structural_elements]
        prediction = np.array(observation, dtype=np.int_)
        prediction[self.structural_elements] = self.structural_values
        return prediction


__all__ = ["EntropyBitwiseDecoder", "EntropyBitwiseFlippingDecoder"]
