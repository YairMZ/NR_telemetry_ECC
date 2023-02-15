from scipy.stats import norm
from protocol_meta import field_lengths


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
        self.up2date = False

    def add_sample(self, sample: int) -> None:
        self.samples.append(sample)
        self.up2date = False

    @property
    def mean(self) -> float:
        if not self.up2date:
            self._update()
        return self._mean

    @property
    def std(self) -> float:
        if not self.up2date:
            self._update()
        return self._std

    def _update(self) -> None:
        self._mean, self._std = norm.fit(self.samples)
        self.up2date = True

    def pdf(self, x: int) -> float:
        if not self.up2date:
            self._update()
        return norm.pdf(x, self._mean, self._std)

    def cdf(self, x: int) -> float:
        if not self.up2date:
            self._update()
        return norm.cdf(x, self._mean, self._std)

    def classify_value(self, value: float, number_of_stds=2) -> int:
        if not self.up2date:
            self._update()
        if value < self._mean - number_of_stds * self._std or value > self._mean + number_of_stds * self._std:
            return 0
        else:
            return 1
