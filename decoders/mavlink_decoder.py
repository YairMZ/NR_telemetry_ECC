"""Mavlink Rectifying decoder"""
from typing import Any
from decoders import Decoder, DecoderType
from collections.abc import Sequence
from ldpc.decoder import LogSpaDecoder
from numpy.typing import NDArray, ArrayLike
import numpy as np
from inference import BufferSegmentation, BufferModel, BufferClassifier
from protocol_meta import dialect_meta as meta


def confusion(predicted_positive: ArrayLike, actual_positive: NDArray[np.int_], negative: int) -> NDArray[np.int_]:
    true_positive = np.intersect1d(predicted_positive, actual_positive).size
    false_positive = np.setdiff1d(predicted_positive, actual_positive).size
    false_negative = np.setdiff1d(actual_positive, predicted_positive).size
    true_negative = negative - false_positive
    # true positive, false positive, true negative, false negative, positive, negative
    return np.array([true_positive, false_positive, true_negative, false_negative, actual_positive.size, negative],
                    dtype=np.int_)


class MavlinkRectifyingDecoder(Decoder):
    """
    This decoder assumes all buffers must contain at least one full MAVLink message.
    Thus, it breaks down buffers to "good" and "bad" parts. It then updates the llr per part.
    Since a buffer may contain padding at the end which cannot be interpreted as messages even without errors, it is best not
    to assume too high bit flip probability even for "bad parts"
    """

    def __init__(self, ldpc_decoder: LogSpaDecoder, model_length: int, valid_thr: float, invalid_thr: float, n_clusters: int,
                 valid_factor: float, invalid_factor: float, classifier_training_size: int | None = None,
                 cluster: bool = True, window_length: int | None = None, data_model: BufferModel | None = None,
                 conf_center: float = 0, conf_slope: float = 0, debug: bool = False, segmentation_iterations: int = 1
                 ) -> None:
        """
        :param ldpc_decoder: decoder for ldpc code used
        :param model_length: length of assumed model in bits, first info bits are assumed to be model bits
        :param valid_thr: valid_threshold for classifying a field as valid
        :param invalid_thr: invalid_threshold for classifying a field as invalid
        :param classifier_training_size: number of buffers to use for training the classifier
        :param n_clusters: number of clusters to use for clustering
        :param valid_factor: factor to multiply llr of valid bits, must be > 1
        :param invalid_factor: factor to multiply llr of invalid bits, must be < 1
        :param cluster: actually use the clustering algo, or assume order of buffers is deterministic
        :param window_length: number of last messages to consider when evaluating distribution and entropy. If none all
        :param data_model: optional model for data bits, if model is given, the decoder will not learn but use the model
        :param conf_center: center of model confidence sigmoid
        :param conf_slope: slope of model confidence sigmoid
        :param debug: analyze classifier and bit forcing performance
        :param segmentation_iterations: number of times to run segmentation (feedback loop)
        """
        self.invalid_thr = invalid_thr
        self.buffer_structures: Sequence[dict[int, int]] = [None] * n_clusters
        self.ldpc_decoder = ldpc_decoder
        self.model_length = model_length
        self.valid_threshold = valid_thr
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
        self.debug = debug
        self.segmentation_iterations = segmentation_iterations
        super().__init__(DecoderType.MAVLINK)

    def decode_buffer(self, channel_llr: NDArray[np.float_], error_idx: NDArray[np.int_]) -> \
            tuple[NDArray[np.int_], NDArray[np.float_], bool, int, NDArray[np.int_], NDArray[np.int_], int, NDArray[np.bool_],
            list[int], list[int], dict[str,Any], dict[str,Any]]:
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
        channel_bits: NDArray[np.int_] = np.array(channel_llr < 0, dtype=np.int_)
        decoder_input = channel_llr.copy()
        all_iteration = 0
        for _ in range(self.segmentation_iterations):
            if self.n_clusters > 1:
                if self.cluster:
                    label: int = self.classifier.classify(channel_bits[self.model_bits_idx])
                    if label < 0:  # label is negative during training phase of classifier, don't try to use model to rectify
                        return self.decode_and_update_model(channel_llr, label, channel_bits) + (np.array([]), [], [], {}, {})
                else:
                    self.running_idx = (self.running_idx + 1) % self.n_clusters
                    label = self.running_idx
            else:  # no clustering, single buffer type
                label = 0

            segmented_bits: NDArray[np.bool_]
            good_fields_idx: list[int]
            bad_fields_idx: list[int]
            n_fields: int
            forced_bits: NDArray[np.int_]
            good_bits: NDArray[np.int_]
            bad_bits: NDArray[np.int_]
            decoder_input, good_bits, bad_bits, good_fields_idx, bad_fields_idx, n_fields, forced_bits,\
                segmented_bits = self.model_prediction(decoder_input, label)
            estimate, decoder_input, decode_success, iterations, syndrome, vnode_validity, label = self.decode_and_update_model(
                decoder_input, label, channel_bits)
            all_iteration += iterations
            if decode_success:
                break
        if self.debug:
            classifier_performance = self.classifier_analysis(good_fields_idx, bad_fields_idx, error_idx, n_fields,
                                                              label)
            forcing_performance = self.forcing_analysis(forced_bits, error_idx, channel_bits, label)
        else:
            classifier_performance = {}
            forcing_performance = {}

        return estimate, decoder_input, decode_success, all_iteration, syndrome, vnode_validity, label,\
            segmented_bits, good_fields_idx, bad_fields_idx, classifier_performance, forcing_performance

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
            tuple[NDArray[np.float_], NDArray[np.bool_], NDArray[np.bool_], list[int], list[int], int, NDArray[np.int_], NDArray[np.bool_]]:
        """rectifies the observation using the model

        :param observation: observation to rectify
        :param cluster_id: cluster to use for rectification
        :return: rectified observation, good bits, bad bits, good fields, bad fields, segmented bits
        """
        if self.buffer_structures[cluster_id] is None:
            return observation, np.array([]), np.array([]), [], [], 0, np.array([]), np.array([])
        # non mavlink bits are nan
        valid_field_p, valid_bits_p, bitwise_std = self.model.predict(np.array(observation < 0, dtype=np.int_),
                                                                      self.buffer_structures[cluster_id])
        good_bits: NDArray[np.bool_] = valid_bits_p > 1 - self.valid_threshold
        bad_bits: NDArray[np.bool_] = valid_bits_p < self.invalid_thr
        good_fields_idx: list[int] = [idx for idx, vfp in enumerate(valid_field_p) if vfp[1] > 1 - self.valid_threshold]
        bad_fields_idx: list[int] = [idx for idx, vfp in enumerate(valid_field_p) if vfp[1] < self.invalid_thr]

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
        use_soft_rectification = False
        if use_soft_rectification:
            factor = np.ones(observation.shape, dtype=np.float_)
            factor[good_bits] += (valid_bits_p[good_bits] - 0.5) * 2
            # factor += (valid_bits_p - 0.5) * 2 * (valid_factor - 1)
            modified_llr *= factor
            modified_llr[bad_bits] *= invalid_factor
        else:
            modified_llr[:len(good_bits)][good_bits] *= valid_factor
            modified_llr[:len(bad_bits)][bad_bits] *= invalid_factor

        # find bits with valid model prediction of 0. These are bits that are wrong for sure and should be forced to the
        # model's value
        forced_bits: NDArray[np.int_] = np.where((valid_bits_p <= 0) & (bitwise_std == 0))[0]
        if forced_bits.size > 0:
            model_bits = self.model.bitwise_model_mean(forced_bits, len(observation) // 8, self.buffer_structures[cluster_id])
            max_llr = np.max(np.abs(observation))
            model_llr = np.power(-1, model_bits) * max_llr
            modified_llr[forced_bits] = model_llr * max(valid_factor, 1)

        # try to run buffer segmentation on the modified llr and on the original llr. If segmentation is able to locate
        # a mavlink message, these bits are good and should be forced.
        info_bytes = self.ldpc_decoder.info_bits(np.array(observation < 0, dtype=np.int_)).tobytes()
        parts, valid_info_bits, structure = self.bs.segment_buffer(info_bytes)
        if structure:
            modified_llr[np.where(valid_info_bits == 1)[0]] = observation[np.where(valid_info_bits == 1)[0]] * 2 * valid_factor
            print("found segmentation")
            # multiply by 2 for confidence
        else:
            info_bytes = self.ldpc_decoder.info_bits(np.array(modified_llr < 0, dtype=np.int_)).tobytes()
            parts, valid_info_bits, structure = self.bs.segment_buffer(info_bytes)
            if structure:
                modified_llr[np.where(valid_info_bits == 1)[0]] *= 2
                print("found segmentation")
                # multiply by 2 for confidence

        return modified_llr, good_bits, bad_bits, good_fields_idx, bad_fields_idx, len(valid_field_p), forced_bits,\
            np.pad(valid_info_bits, (0, len(modified_llr) - len(valid_info_bits))).astype(np.bool_)

    def classifier_analysis(self, good_fields_idx, bad_fields_idx, error_idx, n_fields_in_buffer, label=0) -> dict[str, Any]:

        if self.buffer_structures[label] is None:
            return {}
        damaged = self.model.find_damaged_fields(error_idx, self.buffer_structures[label], self.ldpc_decoder.n // 8)
        if not damaged:
            return {}
        damaged_fields = np.array(tuple({field[1] for field in damaged}))
        bad_fields_performance = confusion(bad_fields_idx, damaged_fields, n_fields_in_buffer - len(damaged))
        actual_good_fields = np.setdiff1d(np.arange(n_fields_in_buffer), damaged_fields)
        good_fields_performance = confusion(good_fields_idx, actual_good_fields, len(damaged))
        return {"n_fields_in_buffer": n_fields_in_buffer, "n_damaged_fields_in_buffer": len(damaged_fields),
                "good_fields_performance": good_fields_performance, "bad_fields_performance": bad_fields_performance}

    def forcing_analysis(self,  # valid_field_p,
                         forced_bits, error_idx, rx: NDArray[np.uint8], label=0) -> dict[str, Any]:
        # damaged = self.model.find_damaged_fields(error_idx, self.buffer_structures[label], self.ldpc_decoder.n // 8)
        # if not damaged:
        #     return {}
        # damaged_fields = np.array(tuple({field[1] for field in damaged}))
        buffer_len_bits = len(rx)
        bits_confusion_matrix = confusion(forced_bits, error_idx, buffer_len_bits-len(error_idx))
        # forced_fields = np.array([idx for idx, vfp in enumerate(valid_field_p) if (vfp[1] <= 0 and vfp[2] == 0)],
        #                          dtype=np.int_)
        # fields_confusion_matrix = confusion(forced_fields, damaged_fields,len(valid_field_p))
        model_bits = self.model.bitwise_model_mean(forced_bits, buffer_len_bits // 8, self.buffer_structures[label])
        bits_flipped_from_forced = forced_bits[(model_bits ^ rx[forced_bits]).astype(np.bool_)]
        flipping_confusion_matrix = confusion(bits_flipped_from_forced, error_idx, buffer_len_bits-len(error_idx))
        n_bits_flipped = bits_flipped_from_forced.size
        errors_bool = np.zeros(rx.shape, dtype=np.bool_)
        errors_bool[error_idx] = True
        tx = rx ^ errors_bool
        if forced_bits.size > 0:
            forcing_quality = np.array([error_idx.size, forced_bits.size, n_bits_flipped, flipping_confusion_matrix[0],
                                        flipping_confusion_matrix[1], sum(tx[forced_bits] ^ model_bits) / forced_bits.size],
                                       dtype=np.float_)
        else:
            forcing_quality = np.array([error_idx.size, forced_bits.size, n_bits_flipped, flipping_confusion_matrix[0],
                                        flipping_confusion_matrix[1], 0], dtype=np.float_)

        return {"bits_confusion_matrix": bits_confusion_matrix,  # "fields_confusion_matrix": fields_confusion_matrix,
                "flipping_confusion_matrix": flipping_confusion_matrix, "forcing_quality": forcing_quality}

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
