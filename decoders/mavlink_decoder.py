"""Mavlink Rectifying decoder"""
from typing import Any, Tuple, List

from numpy import ndarray, dtype, float_, long, bool_

from decoders import Decoder, DecoderType
from collections.abc import Sequence
from ldpc.decoder import LogSpaDecoder
from numpy.typing import NDArray
import numpy as np
from inference import BufferSegmentation, BufferModel, BufferClassifier
from protocol_meta import dialect_meta as meta


class MavlinkRectifyingDecoder(Decoder):
    """
    This decoder assumes all buffers must contain at least one full MAVLink message.
    Thus, it breaks down buffers to "good" and "bad" parts. It then updates the llr per part.
    Since a buffer may contain padding at the end which cannot be interpreted as messages even without errors, it is best not
    to assume too high bit flip probability even for "bad parts"
    """

    def __init__(self, ldpc_decoder: LogSpaDecoder, model_length: int, threshold: float, n_clusters: int,
                 valid_factor: float, invalid_factor: float, classifier_training_size: int | None = None,
                 cluster: bool = True, window_length: int | None = None, data_model: BufferModel | None = None,
                 conf_center: float = 0, conf_slope: float = 0) -> None:
        """
        :param ldpc_decoder: decoder for ldpc code used
        :param model_length: length of assumed model in bits, first info bits are assumed to be model bits
        :param threshold: threshold for classifying a field as valid or invalid
        :param classifier_training_size: number of buffers to use for training the classifier
        :param n_clusters: number of clusters to use for clustering
        :param valid_factor: factor to multiply llr of valid bits, must be > 1
        :param invalid_factor: factor to multiply llr of invalid bits, must be < 1
        :param cluster: actually use the clustering algo, or assume order of buffers is deterministic
        :param window_length: number of last messages to consider when evaluating distribution and entropy. If none all
        :param data_model: optional model for data bits, if model is given, the decoder will not learn but use the model
        :param conf_center: center of model confidence sigmoid
        :param conf_slope: slope of model confidence sigmoid
        """
        self.buffer_structures: Sequence[dict[int, int]] = [None] * n_clusters
        self.ldpc_decoder = ldpc_decoder
        self.model_length = model_length
        self.threshold = threshold
        self.window_length = window_length
        self.bs = BufferSegmentation(meta.protocol_parser)
        self.model: BufferModel = data_model if data_model is not None else BufferModel(window_size=window_length)
        self.learning = data_model is None
        self.model_bits_idx = np.array(self.ldpc_decoder.info_idx)
        if model_length >= sum(self.model_bits_idx):
            self.model_bits_idx = np.array(range(model_length))
        else:
            self.model_bits_idx[model_length:] = False
            self.model_bits_idx = np.flatnonzero(self.model_bits_idx)  # bit indices (among codeword bits) of model bits

        self.running_idx = -1
        self.classifier_training_size = classifier_training_size
        self.n_clusters = n_clusters
        self.cluster = cluster
        self.classifier = BufferClassifier(classifier_training_size, n_clusters) if cluster else None

        self.valid_factor = valid_factor
        self.invalid_factor = invalid_factor

        self.conf_center = conf_center
        self.conf_slope = conf_slope
        super().__init__(DecoderType.MAVLINK)

    def decode_buffer(self, channel_llr: Sequence[np.float_], error_idx: NDArray[np.int_]) -> tuple[
        NDArray[np.int_], NDArray[np.float_], bool, int, NDArray[np.int_], NDArray[np.int_], int, NDArray[np.bool_],
        NDArray[np.bool_], NDArray[np.bool_], list[int], list[int], list[tuple[str,int]]]:
        """decodes a buffer

        :param channel_llr: bits to decode
        :param error_idx: indices of bits that are in error. Used for debugging, and analysis of classifier performance.
        :return: return a tuple (estimated_bits, llr, decode_success, no_iterations, no of mavlink messages found)
        where:
            - estimated_bits is a 1-d np array of hard bit estimates
            - llr is a 1-d np array of soft bit estimates
            - decode_success is a boolean flag stating of the estimated_bits form a valid  code word
            - number of MAVLink messages found within buffer
        """
        # TODO: add feedback loop
        # channel_input: NDArray[np.float_] = np.array(channel_llr, dtype=np.float_)
        channel_bits: NDArray[np.int_] = np.array(channel_llr < 0, dtype=np.int_)
        if self.n_clusters > 1:
            if self.cluster:
                label: int = self.classifier.classify(channel_bits[self.model_bits_idx])
                if label < 0:  # label is negative during training phase of classifier, don't try to use model to rectify
                    return self.decode_and_update_model(channel_llr, label, channel_bits) + (np.array([]), np.array([]),
                                                                                             np.array([]), [], [], [])
            else:
                self.running_idx = (self.running_idx + 1) % self.n_clusters
                label = self.running_idx
        else:  # no clustering, single buffer type
            label = 0

        modified_llr, good_bits, bad_bits, good_field_idx, bad_field_idx, \
                segmented_bits = self.model_prediction(channel_llr, label)
        good_classification, bad_classification = 0, 0
        if self.buffer_structures[label] is not None:
            damaged_fields = self.model.find_damaged_fields(error_idx, self.buffer_structures[label], len(channel_llr)//8)
            if damaged_fields:
                damaged_fields_idx = tuple({field[1] for field in damaged_fields})
                for field in good_field_idx:
                    if field in damaged_fields_idx:
                        bad_classification += 1
                    else:
                        good_classification += 1
                for field in bad_field_idx:
                    if field in damaged_fields_idx:
                        good_classification += 1
                    else:
                        bad_classification += 1
            else:
                good_classification = len(good_field_idx)
                bad_classification = len(bad_field_idx)
        else:
            damaged_fields = []
        return self.decode_and_update_model(modified_llr, label, channel_bits) + \
                (good_bits, bad_bits, segmented_bits, good_field_idx, bad_field_idx, damaged_fields)

    def decode_and_update_model(self, llr, label, channel_bits) -> tuple[
        NDArray[np.int_], NDArray[np.float_], bool, int, NDArray[np.int_], NDArray[np.int_], int]:
        estimate, llr, decode_success, iterations, syndrome, vnode_validity = self.ldpc_decoder.decode(llr)
        if self.learning:
            if decode_success:  # buffer fully recovered
                if self.buffer_structures[label] is None:
                    info_bytes = self.ldpc_decoder.info_bits(estimate).tobytes()
                    parts, validity, structure = self.bs.segment_buffer(info_bytes)
                    self.buffer_structures[label] = structure
                self.model.add_buffer(estimate, self.buffer_structures[label])
            elif self.buffer_structures[label] is not None:  # update model from channel if data is bad
                self.model.add_buffer(channel_bits, self.buffer_structures[label])
        return estimate, llr, decode_success, iterations, syndrome, vnode_validity, label

    def set_buffer_structures(self, buffer_structures: Sequence[dict[int, int]]) -> None:
        """sets the buffer structures to use for decoding

        :param buffer_structures: list of buffer structures to use for decoding
        """
        if len(buffer_structures) != self.n_clusters:
            raise ValueError(f"Number of buffer structures ({len(buffer_structures)}) does not match number of clusters "
                             f"({self.n_clusters})")
        self.buffer_structures = buffer_structures

    def model_prediction(self, observation: NDArray[np.float_], cluster_id: int) -> \
            tuple[NDArray[np.float_], NDArray[np.bool_], NDArray[np.bool_], list[int], list[int], NDArray[np.bool_]]:
        """rectifies the observation using the model

        :param observation: observation to rectify
        :param cluster_id: cluster to use for rectification
        :return: rectified observation, good bits, bad bits, good fields, bad fields, segmented bits
        """
        if self.buffer_structures[cluster_id] is None:
            return observation, np.array([]), np.array([]), [], [], np.array([])
        # non mavlink bits are nan
        valid_field_p, valid_bits_p = self.model.predict(np.array(observation < 0, dtype=np.int_),
                                                         self.buffer_structures[cluster_id])
        good_bits: NDArray[np.bool_] = valid_bits_p > 1 - self.threshold
        bad_bits: NDArray[np.bool_] = valid_bits_p < self.threshold
        good_fields: list[int] = []
        for idx, vfp in enumerate(valid_field_p):
            if vfp[1] > 1 - self.threshold:
                good_fields.append(idx)
        bad_fields: list[int] = []
        for idx, vfp in enumerate(valid_field_p):
            if vfp[1] < self.threshold:
                bad_fields.append(idx)
        if self.learning:  # if model is being learned, need to add a confidence factor to the rectification
            def model_confidence(model_size: int, center: float, slope: float) -> np.float_:
                return 0 if model_size <= 1 else 1 / (1 + np.exp(-(model_size - center) * slope, dtype=np.float_))

            confidence = model_confidence(self.model.model_size, self.conf_center, self.conf_slope)
            valid_factor = (self.valid_factor - 1) * confidence + 1
            invalid_factor = (self.invalid_factor - 1) * confidence + 1
        else:
            valid_factor = self.valid_factor
            invalid_factor = self.invalid_factor
        # rectify
        modified_llr: NDArray[np.float_] = observation.copy()
        modified_llr[good_bits] *= valid_factor
        modified_llr[bad_bits] *= invalid_factor

        # find bits with valid model prediction of 0. These are bits that are wrong for sure and should be forced to the
        # model's value
        forced_bits = np.where(valid_bits_p <= 0)[0]
        if forced_bits.size > 0:
            model_bits = self.model.bitwise_model_mean(forced_bits, len(observation) // 8, self.buffer_structures[cluster_id])
            max_llr = np.max(np.abs(observation))
            model_llr = np.power(-1, model_bits) * max_llr
            # modified_llr[forced_bits] = model_llr * max(valid_factor, 1)

        # try to run buffer segmentation on the modified llr and on the original llr. If segmentation is able to locate
        # a mavlink message, these bits are good and should be forced.
        info_bytes = self.ldpc_decoder.info_bits(np.array(observation < 0, dtype=np.int_)).tobytes()
        parts, valid_info_bits, structure = self.bs.segment_buffer(info_bytes)
        if structure:
            modified_llr[np.where(valid_info_bits == 1)[0]] = observation[np.where(valid_info_bits == 1)[0]] * 2
            # multiply by 2 for confidence
        else:
            info_bytes = self.ldpc_decoder.info_bits(np.array(modified_llr < 0, dtype=np.int_)).tobytes()
            parts, valid_info_bits, structure = self.bs.segment_buffer(info_bytes)
            if structure:
                pass
                # modified_llr[np.where(valid_info_bits == 1)[0]] = modified_llr[np.where(valid_info_bits == 1)] * 2
                # multiply by 2 for confidence
        return modified_llr, good_bits, bad_bits, good_fields, bad_fields, \
            np.pad(valid_info_bits, (0, len(good_bits) - len(valid_info_bits))).astype(np.bool_)

# decode_success = False
# iterations_to_convergence = 0
# for idx in range(self.segmentation_iterations):
#     rectified_llr = self.model.predict(channel_input) if self.model is not None else channel_input
#     estimate, llr, decode_success, iterations, syndrome, vnode_validity = self.ldpc_decoder.decode(channel_input,
#                                                                                                    self.ldpc_iterations)
#     iterations_to_convergence += iterations
#     info_bytes = self.ldpc_decoder.info_bits(estimate).tobytes()
#     parts, validity, structure = self.bs.segment_buffer(info_bytes)
#     if decode_success:
#         break
#     good_bits = np.flatnonzero(np.repeat(parts != MsgParts.UNKNOWN, 8))
#     if good_bits.size > 0 and idx < self.segmentation_iterations:
#         n = channel_input.size
#         bad_bits = n - good_bits.size
#         bad_p = self.bad_p * n / bad_bits
#         channel_input = bad_p(hard_channel_input)
#         channel_input[good_bits] = self.good_p(estimate[good_bits])
#         # for debug
#         # o = np.array(channel_input[good_bits] < 0, dtype=np.int_)
#         # new = np.array(estimate[good_bits] < 0, dtype=np.int_)
#         # t = sum(new != o)
#     else:
#         break
#
# return estimate, llr, decode_success, iterations_to_convergence, len(structure)
