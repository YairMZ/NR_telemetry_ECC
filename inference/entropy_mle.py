import numpy as np
from numpy.typing import NDArray
from typing import Callable
from utils.information_theory import entropy


def bsc_llr(p: float) -> Callable[[NDArray[np.float_]], NDArray[np.float_]]:
    """
    bsc llr is defined as:
        L(c_i) = log(Pr(c_i=0| y_i) / Pr(c_i=1| y_i)) = (-1)^y log((1-p)/p)
    :param float p: the llr is parameterized by the bit flip probability of the channel p.
    :returns: return a callable which accepts a binary input argument - y_i (bits from the channel), and returns its llr
    """
    return lambda y: np.power(-1, y) * np.log((1 - p) / p)


class EMLE:
    """Implementation of the entropy aided MLE denoiser"""

    def __init__(self, f: float, threshold: float = 0.5,
                 window_length: int = 0,
                 conf_center: int = 40, conf_slope: float = 0.35,
                 hard_decision: bool = False, soft_input: bool = True) -> None:
        """
        :param f: channel bit flip probability
        :param hard_decision: whether to use hard decision or soft decision
        """
        self.f = f
        self.hard_decision = hard_decision
        self.soft_input = soft_input
        self.threshold = threshold
        self.conf_center = conf_center
        self.conf_slope = conf_slope
        self.window_length: int = window_length

        self.model_size: int = 0
        self.model_entropy: NDArray[np.float_] = np.array([])
        self.p1: NDArray[np.float_] = np.array([])
        self.model_data: NDArray[np.int_] = np.array([])

    def _model_confidence(self) -> np.float_:
        return 0 if self.model_size <= 1 else 1 / (1 + np.exp(-(self.model_size - self.conf_center) * self.conf_slope,
                                                              dtype=np.float_))

    def reset(self) -> None:
        """resets the denoiser"""
        self.model_size = 0
        self.model_entropy = np.array([])
        self.p1 = np.array([])
        self.model_data = np.array([])

    def predict(self, observation_llr: NDArray[np.float_]) -> NDArray[np.float_]:
        """denoises a buffer
        :param observation_llr: the llr of the buffer to denoise
        :return: a tuple (estimate, cost) where estimate is the denoised sample and cost is the cost of the estimate
        """
        structural_elements: NDArray[np.int_] = self.model_entropy < self.threshold
        if not structural_elements.any():  # no structural elements found
            return observation_llr
        c = self._model_confidence()
        if c > 0:
            observation_llr[structural_elements] += c * np.log(
                (1 - self.p1 + np.finfo(np.float_).eps) / (self.p1 + np.finfo(np.float_).eps))[structural_elements]
        return observation_llr

    def update_model(self, observation_bits: NDArray[np.int_]) -> None:
        """updates the model with a new observation
        :param observation_bits: observed bits
        """
        arr = observation_bits[np.newaxis]
        self.model_data = arr.T if self.model_data.size == 0 else np.append(self.model_data, arr.T, axis=1)
        if 0 < self.window_length < self.model_data.shape[1]:
            # trim old messages according to window
            self.model_data = self.model_data[:, -self.window_length:]
        v = np.mean(self.model_data, axis=1)
        self.p1 = np.clip((v - self.f) / (1 - 2 * self.f), 0, 1)
        self.model_entropy = entropy(np.column_stack((1 - self.p1, self.p1)))
        if self.model_size < self.window_length or self.window_length == 0:
            self.model_size += 1

    def denoise_buffer(self, observation: NDArray[np.float_]) -> tuple[NDArray, np.int_]:
        """denoises a buffer
        :param observation: the llr of the buffer to denoise, or hard bits if hard input is used
        :return: a tuple (estimate, cost) where estimate is the denoised sample and cost is the Hamming distance between the
        estimate and the input (hard values)
        """

        llr = observation.copy() if self.soft_input else bsc_llr(p=self.f)(observation)
        hard_bits = np.array(llr < 0, dtype=np.int_) if self.soft_input else observation.copy()
        estimated_llr = self.predict(llr)
        estimated_bits = np.array(estimated_llr < 0, dtype=np.int_)
        self.update_model(hard_bits)
        cost: np.int_ = np.sum(np.abs(hard_bits - estimated_bits))
        return (estimated_bits, cost) if self.hard_decision else (estimated_llr, cost)

    def denoise_sequence(self, buffer_sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """denoises a sequence of buffers
        :param buffer_sequence: a 2D array of buffers to denoise, each row is a buffer
        :return: a tuple (estimates, costs) where estimates is the denoised 2D array and costs is a 1D array of costs.
        For hard decision estimates is a 2D array of hard bits, for soft decision estimates is a 2D array of llrs
        """
        if self.hard_decision:
            estimates = np.zeros(buffer_sequence.shape, dtype=np.int_)
        else:
            estimates = np.zeros(buffer_sequence.shape, dtype=np.float_)
        costs = np.zeros(buffer_sequence.shape[0], dtype=np.int_)
        for i in range(buffer_sequence.shape[0]):
            estimates[i], costs[i] = self.denoise_buffer(buffer_sequence[i])
        return estimates, costs


__all__ = ['EMLE']
