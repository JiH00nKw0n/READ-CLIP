import logging
from typing import Optional, Union

from datasets import IterableDataset, Dataset
from pydantic import BaseModel
from src.utils import suppress_errors
logger = logging.getLogger(__name__)

__all__ = [
    "BaseBuilder",
]


class BaseBuilder(BaseModel):
    """
    A base class for building datasets. The `build_dataset` method must be implemented by subclasses
    to handle dataset creation.
    """
    split: Optional[str] = None
    name: Optional[str] = None

    def build_dataset(self) -> Union[Dataset, IterableDataset]:
        """
        Builds the dataset. This method must be implemented by subclasses to define how the dataset
        should be constructed.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError
