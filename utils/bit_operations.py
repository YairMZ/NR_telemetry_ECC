"""functions for performing operations on the binary representation of variables."""
import bitstring
from typing import Union
from numpy.typing import NDArray
import numpy as np


def hamming_distance(a: Union[bitstring.Bits, bytes, int, NDArray[np.int_]],
                     b: Union[bitstring.Bits, bytes, int, NDArray[np.int_]],
                     uint_len: int = 8) -> int:
    """Calculate the Hamming distance. Accept multiple types of inputs

    :param a: Union[bitstring.Bits, bytes, int] first object for distance calculation
    :param b: Union[bitstring.Bits, bytes, int] second object for distance calculation
    :param uint_len: length (in bits) of required bitstring. Used for integer inputs, else ignored.
    :rtype: int
    :return: The Hamming distance
    """
    if isinstance(a, bytes):
        a = bitstring.Bits(bytes=a)
    if isinstance(b, bytes):
        b = bitstring.Bits(bytes=b)
    if isinstance(a, int):
        a = bitstring.Bits(uint=a, length=uint_len)
    if isinstance(b, int):
        b = bitstring.Bits(uint=b, length=uint_len)
    if isinstance(a, np.ndarray):
        a = bitstring.Bits(auto=a)
    if isinstance(b, np.ndarray):
        b = bitstring.Bits(auto=b)
    if not (isinstance(a, bitstring.Bits) and isinstance(b, bitstring.Bits)):
        raise TypeError()
    return int((a ^ b).count(True))
