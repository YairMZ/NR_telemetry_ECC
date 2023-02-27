from numpy import ndarray
from scipy.stats import norm
from scipy.special import erf
from protocol_meta import field_lengths, dialect_meta, format_strings, inv_format_strings
import json
from numpy.typing import NDArray
import numpy as np
from mavlink_utils.HC_dialect import mavlink_map
from collections.abc import Sequence
import struct


def num2bits(num, field_type: str) -> NDArray[np.float_]:
    fmt = format_strings[field_type]
    to_bytes = struct.pack(fmt, num)
    return np.unpackbits(np.frombuffer(to_bytes, dtype=np.uint8))


class FieldModel:
    def __init__(self, name: str, field_type: str, window_size: int | None = None):
        if field_type not in field_lengths.keys():
            raise ValueError(f"field_type {field_type} is not supported")
        self.name = name
        self.field_type = field_type
        self.field_len = field_lengths[field_type]
        self._mean = 0
        self._std = 0
        self.samples = []
        self._up2date = False
        self.window_size = window_size
        if "uint" in field_type or "char" in field_type:
            self.int_type = True
            self.signed = False
        elif "int" in field_type:
            self.int_type = True
            self.signed = True
        elif "float" in field_type:
            self.int_type = False
            self.signed = True
        else:
            raise ValueError(f"field_type {field_type} is not supported")

    def add_sample(self, sample) -> None:
        if isinstance(sample, list):
            self.samples.extend(sample)
        else:
            self.samples.append(sample)
        if (self.window_size is not None) and (len(self.samples) > self.window_size):
            self.samples = self.samples[-self.window_size:]
        self._up2date = False

    @property
    def mean(self) -> float:
        if not self._up2date:
            self._update()
        return self._mean

    @property
    def std(self) -> float:
        if not self._up2date:
            self._update()
        return self._std

    def _update(self) -> None:
        if self.samples:
            self._mean, self._std = norm.fit(self.samples)
            self._up2date = True

    def pdf(self, x: int) -> float:
        if not self._up2date:
            self._update()
        return norm.pdf(x, self._mean, self._std)

    def cdf(self, x: int) -> float:
        if not self._up2date:
            self._update()
        return norm.cdf(x, self._mean, self._std)

    def classify_value(self, value: float) -> float:
        """returns the probability that the value is not an outlier assuming a gaussian model"""
        if not self._up2date:
            self._update()
        if self._std <= 0:
            return 1 if value == self._mean else 0
        normalized_distance = abs(self._mean - value) / self._std  # as number of standard deviations
        return 1 - erf(normalized_distance / np.sqrt(2))  # the probability that the value is not an outlier

    def p_of_sign_bit(self) -> float:
        # TODO: this is isn't fully implemented yet
        """returns the probability that the sign bit of a measurement (leftmost bit) is correct assuming a gaussian model.
        For unsigned fields, returns 1"""
        if not self.signed:
            return 1
        if not self._up2date:
            self._update()
        # the probability the sign of changes from that of the mean
        p_change = norm.cdf(-abs(self._mean) / self._std)
        return 1 - p_change

    def bitwise_classify_value(self, value: float) -> NDArray[np.float_]:
        # TODO: this is isn't fully implemented yet
        if not self._up2date:
            self._update()
        bits = num2bits(value, self.field_type)
        return np.zeros((1,))

    def std_significant_bits(self, num_std: float = 1) -> int:
        # TODO: this is isn't fully implemented yet
        if not self._up2date:
            self._update()
        std_in_bits = num2bits(num_std * self._std, self.field_type)
        return int(np.ceil(np.log2(1 / (num_std * self._std))))

    def __str__(self) -> str:
        return f"{self.name} ({self.field_type}): {self.mean} +- {self.std}"

    @property
    def model_size(self) -> int:
        return len(self.samples)


class BufferModel:
    def __init__(self, models: dict[str, FieldModel] = None, window_size: int | None = None):
        self.ordered_field_names = None
        self._buffer_description = None
        self._field_models: dict[str, FieldModel] = {} if models is None else models
        self.window_size = window_size

    def add_sample(self, field_name: str, sample, field_type: str = "") -> None:
        if field_name not in self._field_models.keys():
            self._field_models[field_name] = FieldModel(field_name, field_type, self.window_size)
        self._field_models[field_name].add_sample(sample)

    def __getitem__(self, key: str) -> FieldModel:
        return self._field_models[key]

    def __setitem__(self, key: str, value: FieldModel) -> None:
        self._field_models[key] = value

    def __len__(self) -> int:
        return len(self._field_models)

    def __iter__(self):
        return iter(self._field_models.values())

    def __contains__(self, item: str) -> bool:
        return item in self._field_models.keys()

    def fields_names(self) -> list[str]:
        return list(self._field_models.keys())

    def set_buffer_description(self, description: list) -> None:
        """a buffer is described by a list, where rach element is either a string holding a field name,
        or a bytes object holding an expected mavlink message header"""
        self._buffer_description = description
        self.ordered_field_names = []
        for item in description:
            if isinstance(description, Sequence) and not isinstance(item, str) and not isinstance(item, bytes):
                self.ordered_field_names.append(item[0])

    def get_buffer_description(self) -> list:
        return self._buffer_description

    def get_field_model(self, field_name: str) -> FieldModel:
        return self._field_models.get(field_name)

    def predict(self, buffer: bytes| NDArray[np.int_], buffer_structure: dict[int, int], bitwise: bool = True) -> \
            list[tuple[str, float]]| NDArray[np.float_]:
        """predicts the probability of bits originating from a valid (not an outlier) field
        :param buffer: the buffer to predict
        :param buffer_structure: a dictionary mapping the byte index in the buffer to the relevant mavlink message id
        :param bitwise: if True, the probability of each bit is calculated, otherwise the probability of the whole field
        :return: probabilities of fields being valid (not an outlier). If bitwise is True, and array probabilities are for each
        bit within the field, otherwise a list of tuples with filed name and valid probability.
        """
        if isinstance(buffer, ndarray):  # if buffer is a numpy array, convert to bytes
            if bitwise:
                bit_2_val_idx = np.array([-1] * len(buffer))
            buffer = [
                int(''.join(map(str, buffer[i: i + 8])), 2)
                for i in range(0, len(buffer), 8)
            ]
            buffer = bytes(buffer)  # convert to bytes
        elif isinstance(buffer, bytes):
            if bitwise:
                bit_2_val_idx = np.array([-1] * len(buffer) * 8)
        else:
            raise ValueError("buffer must be either a numpy array of bits or bytes object")

        # unpack bytes to fields using mavlink message unpacker
        # for example
        # {0: 212, 27: 218, 45: 33, 97: 234}
        fields = tuple()
        ordered_field_names = []

        field_idx = 0
        for byte_idx, msg_id in buffer_structure.items():
            fields += mavlink_map[msg_id].unpacker.unpack(
                buffer[byte_idx + dialect_meta.header_len:
                       byte_idx + dialect_meta.header_len + dialect_meta.msgs_payload_length[msg_id]])
            ordered_field_names += [f'{mavlink_map[msg_id].name}_{name}' for name in mavlink_map[msg_id].ordered_fieldnames]
            if bitwise:
                format_string = mavlink_map[msg_id].unpacker.format
                start_bit = (byte_idx + dialect_meta.header_len) * 8
                for char in format_string[1:]:  # skip the first character which is the endianness
                    field_len_bits = field_lengths.get(inv_format_strings.get(char)) * 8
                    last_bit = start_bit + field_len_bits
                    bit_2_val_idx[start_bit:last_bit] = [field_idx] * (last_bit - start_bit)
                    field_idx += 1
                    start_bit = last_bit

        # calculate the valid probability of each field
        if bitwise:
            valid_probability = np.array([None] * len(bit_2_val_idx)).astype(np.float_)
        else:
            valid_probability = [None] * len(fields)
        for field_idx, field_name, value in zip(range(len(fields)), ordered_field_names, fields):
            if bitwise:
                valid_probability[bit_2_val_idx == field_idx] = self._field_models[field_name].classify_value(value)
            else:
                valid_probability[field_idx] = (field_name, self._field_models[field_name].classify_value(value))
        return valid_probability
        # TODO: implement

    def save(self, filename: str) -> None:
        for model in self._field_models.values():
            model._update()
        with open(filename, 'w') as f:
            json.dump({"field_models": self._field_models, "buffer_description": self._buffer_description,
                       "window_size": self.window_size}, f,
                      default=lambda o: o.hex() if isinstance(o, bytes) else o.__dict__, indent=4, sort_keys=True)

    @classmethod
    def load(cls, filename: str) -> "BufferModel":
        with open(filename, 'r') as f:
            data = json.load(f)
            models = data.get("field_models")
            buffer_description = data.get("buffer_description")
            window_size = data.get("window_size")
            obj = cls()
            for field_name, vals in models.items():
                model = FieldModel(field_name, vals["field_type"], window_size=vals["window_size"])
                model.add_sample(vals["samples"])
                model._update()
                obj[field_name] = model
            # encode bytes objects
            for i, val in enumerate(buffer_description):
                if isinstance(val, str) and val != 'crc':
                    buffer_description[i] = bytes.fromhex(val)
            obj.set_buffer_description(buffer_description)
            obj.window_size = window_size
            return obj

    @staticmethod
    def find_damaged_fields(error_indices, structure):
        """finds the damaged fields in a buffer based on the error indices. Use for debugging / analysis purposes only."""
        ordered_field_names = []
        damaged_fields = []
        field_idx = 0
        for byte_idx, msg_id in structure.items():
            ordered_field_names += [f'{mavlink_map[msg_id].name}_{name}' for name in mavlink_map[msg_id].ordered_fieldnames]
            format_string = mavlink_map[msg_id].unpacker.format
            start_bit = (byte_idx + dialect_meta.header_len) * 8
            for char in format_string[1:]:
                field_len_bits = field_lengths.get(inv_format_strings.get(char)) * 8
                last_bit = start_bit + field_len_bits
                if np.any((error_indices >= start_bit) & (error_indices < last_bit)):
                    damaged_fields.append((ordered_field_names[field_idx], field_idx))
                start_bit = last_bit
                field_idx += 1
        return damaged_fields

    def add_buffer(self, buffer: bytes| NDArray[np.int_], structure: dict[int, int]) -> None:
        """add observations to the model
        :param buffer: the buffer to add
        :param structure: a dictionary mapping the byte index in the buffer to the relevant mavlink message id
        """
        if isinstance(buffer, ndarray):  # if buffer is a numpy array, convert to bytes
            buffer = [
                int(''.join(map(str, buffer[i: i + 8])), 2)
                for i in range(0, len(buffer), 8)
            ]
            buffer = bytes(buffer)  # convert to bytes
        elif not isinstance(buffer, bytes):
            raise ValueError("buffer must be either a numpy array of bits or bytes object")
        fields = tuple()
        ordered_field_names = []
        ordered_field_types = []
        for byte_idx, msg_id in structure.items():
            fields += mavlink_map[msg_id].unpacker.unpack(
                buffer[byte_idx + dialect_meta.header_len:
                       byte_idx + dialect_meta.header_len + dialect_meta.msgs_payload_length[msg_id]])
            ordered_field_names += [f'{mavlink_map[msg_id].name}_{name}' for name in mavlink_map[msg_id].ordered_fieldnames]
            for name in mavlink_map[msg_id].ordered_fieldnames:
                type_idx = mavlink_map[msg_id].fieldnames.index(name)
                ordered_field_types.append(mavlink_map[msg_id].fieldtypes[type_idx])
        for field_name, value, field_type in zip(ordered_field_names, fields, ordered_field_types):
            self.add_sample(field_name, value, field_type)


__all__: list[str] = ["FieldModel", "BufferModel", "num2bits"]
