"""package for breaking data a message to tokens, and attempt reconstruction"""
from .buffer_segmentation import MsgParts, BufferStructure, BufferSegmentation
__all__: list[str] = ["MsgParts", "BufferStructure", "BufferSegmentation"]
