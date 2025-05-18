import logging
from dataclasses import dataclass
from typing import Union, List, Dict, Optional, TypeVar, Any

from transformers import BatchEncoding, ProcessorMixin
from transformers.utils import PaddingStrategy, add_end_docstrings

logger = logging.getLogger(__name__)

ProcessorType = TypeVar("ProcessorType", bound=ProcessorMixin)

__all__ = [
    "BaseCollator",
    "BASE_COLLATOR_DOCSTRING"
]

BASE_COLLATOR_DOCSTRING = """
A collator class for processing inputs with dynamic padding, truncation, and tensor conversion.

Args:
    processor ([`ProcessorMixin`]):
        The processor used to encode the data (e.g., tokenizer).
    padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `'max_length'`):
        Padding strategy to apply. It determines whether to pad, and if so, how.
        Can be a boolean or a padding strategy ('longest', 'max_length', 'do_not_pad').
    truncation (`bool`, *optional*, defaults to `True`):
        Whether to truncate the inputs to the maximum length. If True, inputs will be truncated
        to the max_length specified.
    max_length (`int`, *optional*, defaults to 64):
        Maximum length of the inputs after padding or truncation. Inputs longer than this will be
        truncated, and shorter ones will be padded.
    pad_to_multiple_of (`int`, *optional*):
        If specified, pads the input to a multiple of this value. This is useful when working
        with certain models that require input sequences to have a specific length that is a multiple
        of a particular value.
    return_tensors (`str`, *optional*, defaults to `'pt'`):
        The format of the tensors to return. Can be 'np' for NumPy arrays, 'pt' for PyTorch tensors,
        or 'tf' for TensorFlow tensors.
"""


@add_end_docstrings(BASE_COLLATOR_DOCSTRING)
@dataclass
class BaseCollator:
    """
    An abstract base class for collators that handle dynamic padding, truncation, and tensor conversion.
    This class provides the common structure for subclasses that need to process input data with these
    features before passing the data to a model processor.

    Subclasses should implement the `__call__` method to define how input data is processed.

    Raises:
        NotImplementedError:
            If the `__call__` method is not implemented in a subclass.
    """
    processor: ProcessorType
    padding: Union[bool, str, PaddingStrategy] = 'max_length'
    truncation: bool = True
    max_length: Optional[int] = 77
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    color_model: str = "RGB"

    def __call__(self, inputs: List[Dict[str, Any]]) -> BatchEncoding:
        raise NotImplementedError
