from scipy.stats import norm
from protocol_meta import field_lengths
import json


class FieldModel:
    def __init__(self, name: str, field_type: str):
        if field_type not in field_lengths.keys():
            raise ValueError(f"field_type {field_type} is not supported")
        self.name = name
        self.field_type = field_type
        self.field_len = field_lengths[field_type]
        self._mean = 0
        self._std = 0
        self.samples = []
        self._up2date = False

    def add_sample(self, sample) -> None:
        if isinstance(sample, list):
            self.samples.extend(sample)
        else:
            self.samples.append(sample)
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

    def classify_value(self, value: float, number_of_stds=2) -> int:
        if not self._up2date:
            self._update()
        if value < self._mean - number_of_stds * self._std or value > self._mean + number_of_stds * self._std:
            return 0
        else:
            return 1

    def __str__(self) -> str:
        return f"{self.name} ({self.field_type}): {self.mean} +- {self.std}"


class BufferModel:
    def __init__(self, models: dict[str, FieldModel] = None):
        self._buffer_description = None
        self._field_models: dict[str, FieldModel] = {} if models is None else models

    def add_sample(self, field_name: str, sample, field_type: str = "") -> None:
        if field_name not in self._field_models.keys():
            self._field_models[field_name] = FieldModel(field_name, field_type)
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
        """a buffer is described by a list, where rach element is either a string holding a fieldname,
        or a bytes object holding an expected mavlink message header"""
        self._buffer_description = description

    def save(self, filename: str) -> None:
        with open(filename, 'w') as f:
            json.dump([self._field_models, self._buffer_description], f,
                      default=lambda o: o.hex() if isinstance(o, bytes) else o.__dict__
                      , indent=4, sort_keys=True)

    @classmethod
    def load(cls, filename: str) -> "BufferModel":
        with open(filename, 'r') as f:
            data = json.load(f)
            models = data[0]
            buffer_description = data[1]
            obj = cls()
            for fieldname, vals in models.items():
                model = FieldModel(fieldname, vals["field_type"])
                model.add_sample(vals["samples"])
                model._update()
                obj[fieldname] = model
            # encode bytes objects
            for i, val in enumerate(buffer_description):
                if isinstance(val, str):
                    if val != 'crc':
                        buffer_description[i] = bytes.fromhex(val)
            obj.set_buffer_description(buffer_description)
            return obj


__all__: list[str] = ["FieldModel", "BufferModel"]
