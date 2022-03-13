"""Mavlink Rectifying decoder"""
from decoders import Decoder, DecoderType
from collections.abc import Sequence
from ldpc.decoder import LogSpaDecoder, bsc_llr
import numpy as np
from inference import BufferSegmentation, MsgParts
from protocol_meta import dialect_meta as meta
from numpy.typing import NDArray


class MavlinkRectifyingDecoder(Decoder):
    """
    This decoder assumes all buffers must contain at least one full MAVLink message.
    Thus, it breaks down buffers to "good" and "bad" parts. It then updates the llr per part.
    Since a buffer may contain padding at the end which cannot be interpreted as messages even without errors, it is best not
    to assume too high bit flip probability even for "bad parts"


    """
    def __init__(self, ldpc_decoder: LogSpaDecoder, segmentation_iterations: int, ldpc_iterations: int, k: int,
                 bad_p: float, good_p: float) -> None:
        """
        :param ldpc_decoder: decoder object for BP decoding
        :param segmentation_iterations: number of times segmentation is done.
        :param ldpc_iterations: iterations between segmentations
        :param k: number of information bearing bits
        :param bad_p:
        :param good_p:
        """

        self.ldpc_decoder = ldpc_decoder
        self.bs = BufferSegmentation(meta.protocol_parser)
        self.segmentation_iterations = segmentation_iterations
        self.ldpc_iterations = ldpc_iterations
        self.k = k
        self.bad_p = bad_p
        self.good_p = bsc_llr(p=good_p)
        self.v_node_uids = [node.uid for node in self.ldpc_decoder.ordered_vnodes()][:self.k]  # it is assumed first k vnodes
        # hold information
        super().__init__(DecoderType.MAVLINK)

    def decode_buffer(self, channel_llr: Sequence[np.float_]) -> tuple[NDArray[np.int_], NDArray[np.float_], bool, int, int]:
        """decodes a buffer

        :param channel_llr: bits to decode
        :return: return a tuple (estimated_bits, llr, decode_success, no_iterations, no of mavlink messages found)
        where:
            - estimated_bits is a 1-d np array of hard bit estimates
            - llr is a 1-d np array of soft bit estimates
            - decode_success is a boolean flag stating of the estimated_bits form a valid  code word
            - number of MAVLink messages found within buffer
        """
        channel_input: NDArray[np.float_] = np.array(channel_llr, dtype=np.float_)
        hard_channel_input: NDArray[np.int_] = np.array(channel_input < 0, dtype=np.int_)
        decode_success = False
        iterations_to_convergence = 0
        for idx in range(self.segmentation_iterations + 1):
            estimate, llr, decode_success, iterations, syndrome, vnode_validity = self.ldpc_decoder.decode(channel_input, self.ldpc_iterations)
            iterations_to_convergence += iterations
            info_bytes = self.ldpc_decoder.info_bits(estimate).tobytes()
            parts, validity, structure = self.bs.segment_buffer(info_bytes)
            if decode_success:
                break
            good_bits = np.flatnonzero(np.repeat(parts != MsgParts.UNKNOWN, 8))
            if good_bits.size > 0 and idx < self.segmentation_iterations:
                n = channel_input.size
                bad_bits = n - good_bits.size
                bad_p = bsc_llr(p=self.bad_p*n/bad_bits)
                channel_input = bad_p(hard_channel_input)
                channel_input[good_bits] = self.good_p(estimate[good_bits])
                # for debug
                # o = np.array(channel_input[good_bits] < 0, dtype=np.int_)
                # new = np.array(estimate[good_bits] < 0, dtype=np.int_)
                # t = sum(new != o)
            else:
                break

        return estimate, llr, decode_success, iterations_to_convergence, len(structure)
