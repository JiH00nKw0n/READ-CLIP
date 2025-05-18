from typing import List, Optional, Union

from datasets import concatenate_datasets, Dataset, load_dataset

from src.common import registry
from .base import (
    BaseBuilder
)

__all__ = [
    "COCOCaptionsDatasetBuilder",
    "COCOCaptionsWithImageDatasetBuilder",
]


@registry.register_builder('COCOCaptionsDatasetBuilder')
class COCOCaptionsDatasetBuilder(BaseBuilder):
    """
    A builder class for creating a dataset for COCO captions with non-iterable format.
    It extends `SequenceTextDatasetFeaturesWithImageURL`.

    Attributes:
        split (Union[str, List[str]]): The dataset split(s) to load.
        name (Optional[str]): The name of the dataset.
    """
    split: Union[str, List[str]] = ['train', 'restval']
    name: Optional[str] = 'coco'

    def build_dataset(self) -> Dataset:
        """
        Builds and returns the COCO captions dataset.

        Returns:
            Dataset: The COCO captions dataset.
        """
        if isinstance(self.split, list):
            dataset = concatenate_datasets(
                load_dataset(
                    "yerevann/coco-karpathy", trust_remote_code=True, split=self.split
                )
            )
        else:
            dataset = load_dataset(
                "yerevann/coco-karpathy", trust_remote_code=True, split=self.split
            )
        dataset = dataset.rename_columns({"sentences": 'text', "url": 'images'})
        dataset = dataset.select_columns(['images', 'text'])

        return dataset


@registry.register_builder('COCOCaptionsWithImageDatasetBuilder')
class COCOCaptionsWithImageDatasetBuilder(BaseBuilder):
    """
    A builder class for creating a dataset for COCO captions with non-iterable format.
    It extends `SequenceTextDatasetFeaturesWithImageURL`.

    Attributes:
        split (Union[str, List[str]]): The dataset split(s) to load.
        name (Optional[str]): The name of the dataset.
    """
    split: str = 'train'
    name: Optional[str] = 'coco'

    def build_dataset(self) -> Dataset:
        """
        Builds and returns the COCO captions dataset.

        Returns:
            Dataset: The COCO captions dataset.
        """
        dataset = load_dataset(
            "Mayfull/coco-karpathy-with-image", trust_remote_code=True, split=self.split
        )

        return dataset
