"""The package contains all the relevant metadata regarding a protocol."""
from .protocol_meta import MAVError, field_lengths
from .protocol_meta import dialect_meta as dialect_meta
from .msg_header import hamming_distance_2_valid_header, FrameHeader, is_valid_header, NonExistentMsdId
__all__: list[str] = ["MAVError", "dialect_meta", "hamming_distance_2_valid_header", "FrameHeader",
                      "is_valid_header", "NonExistentMsdId", "field_lengths"]
