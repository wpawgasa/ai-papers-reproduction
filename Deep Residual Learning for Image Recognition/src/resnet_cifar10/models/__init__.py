"""Model architectures for ResNet and PlainNet."""

from .resnet import ResNetCIFAR, BasicBlock
from .plainnet import PlainNetCIFAR, PlainBlock

__all__ = ["ResNetCIFAR", "BasicBlock", "PlainNetCIFAR", "PlainBlock"]
