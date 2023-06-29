from inference import BufferSegmentation, BufferModel, BufferClassifier
from protocol_meta import dialect_meta as meta
from decoders import Decoder, DecoderType
from collections.abc import Sequence
from ldpc.decoder import LogSpaDecoder
from numpy.typing import NDArray, ArrayLike
import numpy as np
from typing import Any
from utils.custom_exceptions import IncorrectBufferLength
from utils.information_theory import prob, entropy


def confusion(predicted_positive: ArrayLike, actual_positive: NDArray[np.int_], negative: int) -> NDArray[np.int_]:
    true_positive = np.intersect1d(predicted_positive, actual_positive).size
    false_positive = np.setdiff1d(predicted_positive, actual_positive).size
    false_negative = np.setdiff1d(actual_positive, predicted_positive).size
    true_negative = negative - false_positive
    # true positive, false positive, true negative, false negative, positive, negative
    return np.array([true_positive, false_positive, true_negative, false_negative, actual_positive.size, negative],
                    dtype=np.int_)


class CombinedDecoder(Decoder):
    """
    This decoder combines the classifying decoder and the segmentation decoder.
    """

    def __init__(self, ldpc_decoder: LogSpaDecoder, model_length: int,
                 valid_thr: float, invalid_thr: float, n_clusters: int,
                 valid_factor: float, invalid_factor: float,
                 entropy_threshold: float, clipping_factor: float,
                 classifier_training_size: int | None = None,
                 cluster: bool = True, window_length: int | None = None, data_model: BufferModel | None = None,
                 conf_center: int = 0, conf_slope: float = 0, debug: bool = False, bit_flip: float = 0
                 ) -> None:
        self.invalid_thr = invalid_thr
        self.buffer_structures: Sequence[dict[int, int]] = [None] * n_clusters
        self.ldpc_decoder = ldpc_decoder
        self.model_length = model_length  # in bits
        self.valid_threshold = valid_thr
        self.window_length = window_length
        self.bs = BufferSegmentation(meta.protocol_parser)
        self.model: BufferModel = data_model if data_model is not None else BufferModel(window_size=window_length)
        self.mavlink_learning = data_model is None
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

        self.entropy_models_data: list[NDArray[np.uint8]] = [np.array([]) for _ in range(n_clusters)]
        self.entropy_threshold = entropy_threshold
        self.clipping_factor = clipping_factor  # The model llr is clipped to +-clipping_factor * max_chanel_llr
        self.bit_flip = bit_flip
        self.distributions: list[NDArray[np.float_]] = [np.array([]) for _ in range(n_clusters)]
        self.models_entropy: list[NDArray[np.float_]] = [np.array([]) for _ in range(n_clusters)]

        super().__init__(DecoderType.COMBINED)

    def entropy_model_prediction(self, cluster_id: int) -> NDArray[np.float_] | None:
        def model_confidence(model_size: int, center: int, slope: float) -> np.float_:
            return 0 if model_size <= 1 else 1 / (1 + np.exp(-(model_size - center) * slope, dtype=np.float_))

        # infer structure
        # index of structural (low entropy) elements among codeword
        structural_elements: NDArray[np.int_] = self.model_bits_idx[
            self.models_entropy[cluster_id] < self.entropy_threshold]

        if not structural_elements.any():  # no structural elements found
            return None

        size = self.entropy_models_data[cluster_id].shape[1] if self.entropy_models_data[cluster_id].size > 0 else 0
        confidence = model_confidence(size, self.conf_center, self.conf_slope)  # consider window size when setting these

        # model llr is calculated as log(Pr(c=0 | model) / Pr(c=1| model))
        # add model llr to the observation
        if confidence > 0:
            llr = confidence * np.log(
                (np.finfo(np.float_).eps + self.distributions[cluster_id][:, 0]) / (
                            self.distributions[cluster_id][:, 1] + np.finfo(np.float_).eps)
            )[self.models_entropy[cluster_id] < self.entropy_threshold]
            return llr, structural_elements

    def update_entropy_model(self, bits: NDArray[np.int_], cluster_id: int) -> None:
        """update model of data. model_b uses any data regardless of correctness.
        :param bits: hard estimate for bit values, assumed to be correct.
        """
        if len(bits) != self.model_length:
            print(self.model_length)
            print(len(bits))
            raise IncorrectBufferLength()
        arr = bits[np.newaxis]

        self.entropy_models_data[cluster_id] = arr.T if self.entropy_models_data[cluster_id].size == 0 else \
            np.append(self.entropy_models_data[cluster_id], arr.T, axis=1)
        if (self.window_length is not None) and self.entropy_models_data[cluster_id].shape[1] > self.window_length:
            # trim old messages according to window
            self.entropy_models_data[cluster_id] = self.entropy_models_data[cluster_id][
                                                   :, self.entropy_models_data[cluster_id].shape[1] - self.window_length:]
        dist = prob(self.entropy_models_data[cluster_id])
        v = dist[:, 1]
        p1 = np.clip((v - self.bit_flip) / (1 - 2 * self.bit_flip), 0, 1)
        self.distributions[cluster_id] = np.column_stack((1 - p1, p1))
        # mu equals p1
        if max(p1) > 1 or min(p1) < 0:
            print("problematic probability")
            raise RuntimeError("problematic probability")
        self.models_entropy[cluster_id] = entropy(self.distributions[cluster_id])

    def set_buffer_structures(self, buffer_structures: Sequence[dict[int, int]]) -> None:
        """sets the buffer structures to use for decoding

        :param buffer_structures: list of buffer structures to use for decoding
        """
        if len(buffer_structures) != self.n_clusters:
            raise ValueError(f"Number of buffer structures ({len(buffer_structures)}) does not match number of clusters "
                             f"({self.n_clusters})")
        self.buffer_structures = buffer_structures

    def mavlink_model_prediction(self, observation: NDArray[np.float_], cluster_id: int) -> \
            tuple[
                NDArray[np.float_], NDArray[np.bool_], NDArray[np.bool_], list[int], list[int], int, NDArray[np.int_], NDArray[
                    np.bool_]]:
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

        if self.mavlink_learning:  # if model is being learned, need to add a confidence factor to the rectification
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

        return modified_llr, good_bits, bad_bits, good_fields_idx, bad_fields_idx, len(valid_field_p), forced_bits, \
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
        bits_confusion_matrix = confusion(forced_bits, error_idx, buffer_len_bits - len(error_idx))
        # forced_fields = np.array([idx for idx, vfp in enumerate(valid_field_p) if (vfp[1] <= 0 and vfp[2] == 0)],
        #                          dtype=np.int_)
        # fields_confusion_matrix = confusion(forced_fields, damaged_fields,len(valid_field_p))
        model_bits = self.model.bitwise_model_mean(forced_bits, buffer_len_bits // 8, self.buffer_structures[label])
        bits_flipped_from_forced = forced_bits[(model_bits ^ rx[forced_bits]).astype(np.bool_)]
        flipping_confusion_matrix = confusion(bits_flipped_from_forced, error_idx, buffer_len_bits - len(error_idx))
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

    def decode_buffer(self, channel_llr: NDArray[np.float_], error_idx: NDArray[np.int_]) -> Any:
        channel_bits: NDArray[np.int_] = np.array(channel_llr < 0, dtype=np.int_)
        decoder_input = channel_llr.copy()
        if self.n_clusters > 1:
            if self.cluster:
                label: int = self.classifier.classify(channel_bits[self.model_bits_idx])
                if label < 0:  # label is negative during training phase of classifier, don't try to use model to rectify
                    return self.decode_and_update_model(channel_llr, label, channel_bits)
            else:
                self.running_idx = (self.running_idx + 1) % self.n_clusters
                label = self.running_idx
        else:  # no clustering, single buffer type
            label = 0

        # mavlink based prediction
        segmented_bits: NDArray[np.bool_]
        good_fields_idx: list[int]
        bad_fields_idx: list[int]
        n_fields: int
        forced_bits: NDArray[np.int_]
        good_bits: NDArray[np.int_]
        bad_bits: NDArray[np.int_]
        decoder_input, good_bits, bad_bits, good_fields_idx, bad_fields_idx, n_fields, forced_bits, \
            segmented_bits = self.mavlink_model_prediction(decoder_input, label)
        if self.debug:
            classifier_performance = self.classifier_analysis(good_fields_idx, bad_fields_idx, error_idx, n_fields,
                                                              label)
            forcing_performance = self.forcing_analysis(forced_bits, error_idx, channel_bits, label)

        # entropy based prediction
        model_result = self.entropy_model_prediction(label)  # type: ignore
        if model_result is not None:
            clipping = self.clipping_factor * max(abs(channel_llr))  # llr s clipped within +-clipping
            decoder_input[model_result[1]] += np.clip(model_result[0], -clipping, clipping)

        return self.decode_and_update_model(decoder_input, label, channel_bits)

    def decode_and_update_model(self, decoder_input: NDArray[np.float_], label: int, channel_bits: NDArray[np.int_]) -> Any:
        estimate, llr, decode_success, iterations, syndrome, vnode_validity = self.ldpc_decoder.decode(decoder_input)
        if label >= 0:
            if self.mavlink_learning:
                if decode_success:  # buffer fully recovered
                    if self.buffer_structures[label] is None:
                        info_bytes = self.ldpc_decoder.info_bits(estimate).tobytes()
                        parts, validity, structure = self.bs.segment_buffer(info_bytes)
                        self.buffer_structures[label] = structure
                    self.model.add_buffer(estimate, self.buffer_structures[label])
                elif self.buffer_structures[label] is not None:  # update model from channel if data is bad
                    self.model.add_buffer(channel_bits, self.buffer_structures[label])
            if decode_success:  # buffer fully recovered
                self.update_entropy_model(estimate, label)
            else:  # update model from channel if data is bad
                self.update_entropy_model(channel_bits, label)
        return estimate, llr, decode_success, iterations, label
