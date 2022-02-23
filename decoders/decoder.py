"""general base class for all decoders"""
from abc import ABC, abstractmethod
from enum import Enum, auto
from numpy.typing import NDArray
from collections.abc import Sequence
import numpy as np


class DecoderType(Enum):
    """Types of decoders"""
    ENTROPY = auto()
    RECTIFYING = auto()


class Decoder(ABC):
    """The class serves as a base class for all decoders, and serves as an interface"""
    def __init__(self, decoder_type: DecoderType):
        self.decoder_type = decoder_type

    @abstractmethod
    def decode_buffer(self, channel_word: Sequence[np.float_]) -> tuple[NDArray[np.int_], NDArray[np.float_], bool, int, int]:
        """decodes a buffer
        :param channel_word: buffer to decode, input can be decimal (int) byte or bit values, or channel llr (float).
        :return: return a tuple (estimated_bits, llr, decode_success, no_iterations, no of mavlink messages found)
        where:
            - estimated_bits is a 1-d np array of hard bit estimates
            - llr is a 1-d np array of soft bit estimates
            - decode_success is a boolean flag stating of the estimated_bits form a valid  code word
            - number of iterations until breaking
            - number of MAVLink messages found within buffer
        """
