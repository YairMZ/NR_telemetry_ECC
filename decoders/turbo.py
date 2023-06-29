from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from decoders import Decoder, DecoderType
from ldpc.decoder import LogSpaDecoder


def track_errors(error_idx: NDArray[np.int_], rx: NDArray[np.uint8], estimate: NDArray[np.uint8]) -> int:
    """Used for debug to track convergence of error"""
    errors_bool = np.zeros(rx.shape, dtype=np.bool_)
    errors_bool[error_idx] = True
    tx = rx ^ errors_bool
    return sum(tx != estimate)


class TurboDecoder(Decoder):
    """skeleton for turbo decoder"""
    def __init__(self, first_decoder: Decoder, second_decoder: LogSpaDecoder, max_iter:int, debug: bool = False) -> None:
        self.first_decoder = first_decoder
        self.second_decoder = second_decoder
        self.max_iter = max_iter
        self.debug = debug
        super().__init__(DecoderType.TURBO)
    def decode_buffer(self, channel_word: NDArray[np.float_], error_idx: NDArray[np.int_]) -> tuple[NDArray[np.int_], NDArray[np.float_], bool, int, int]:
        """

        :param channel_word: received word
        :param error_idx: error index, for debug only
        :return: a tuple (estimated_bits, llr, decode_success, which decoder returned final value, number of final error bits)
        """
        first_decoder_prior = np.zeros_like(channel_word)
        second_decoder_prior = np.zeros_like(channel_word)

        for _ in range(self.max_iter):
            first_decoder_output = self.first_decoder.decode_buffer(channel_word.copy() + first_decoder_prior)
            estimate1, llr1, decode_success1 = first_decoder_output[:3]
            if decode_success1: # early exit
                return estimate1, llr1, decode_success1, 1, 0
            second_decoder_prior = llr1 - channel_word - first_decoder_prior  # update prior, since we don't use an
            # interleaver, extrinsic information is the same as the prior
            second_decoder_output = self.second_decoder.decode(second_decoder_prior + channel_word.copy())
            estimate2, llr2, decode_success2 = second_decoder_output[:3]
            if decode_success2: # early exit
                return estimate2, llr2, decode_success2, 2, 0
            first_decoder_prior = llr2 - channel_word - second_decoder_prior  # update prior, since we don't use an
            # interleaver, extrinsic information is the same as the prior
            if self.debug:
                err1 = track_errors(error_idx, np.array(channel_word<0,dtype=np.uint8), estimate1.astype(np.uint8))
                err2 = track_errors(error_idx, np.array(channel_word<0,dtype=np.uint8), estimate2.astype(np.uint8))

        # if we got here, both decoders failed
        if self.debug:
            return estimate2, llr2, decode_success2, 2, err2
        else:
            return estimate2, llr2, decode_success2, 2, 0

