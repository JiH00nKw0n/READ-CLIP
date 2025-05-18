import os
from typing import Optional

from datasets import load_dataset, Dataset

from src.builders.base import BaseBuilder
from src.common import registry


@registry.register_builder('SugarCrepeBuilder')
class SugarCrepeBuilder(BaseBuilder):
    """
    A builder class for creating the SugarCrepe dataset.

    Attributes:
        split (Optional[str]): The dataset split to load (default: 'train').
        name (Optional[str]): The name of the dataset (default: 'SugarCrepe').
    """
    split: Optional[str] = 'test'
    name: Optional[str] = 'SugarCrepe'

    def build_dataset(self) -> Dataset:
        """
        Builds and returns the SUGARCREPE dataset.

        Returns:
            Dataset: The SUGARCREPE dataset with sequences of text and images.
        """
        dataset = load_dataset(
            path="Mayfull/sugarcrepe_vlms",
            trust_remote_code=True,
            split=self.split,
            token=str(os.getenv("HF_TOKEN"))
        )

        return dataset


@registry.register_builder('SugarCrepePPBuilder')
class SugarCrepePPBuilder(BaseBuilder):
    """
    A builder class for creating the SugarCrepe dataset.

    Attributes:
        split (Optional[str]): The dataset split to load (default: 'train').
        name (Optional[str]): The name of the dataset (default: 'SugarCrepe').
    """
    split: Optional[str] = 'test'
    name: Optional[str] = 'SugarCrepe++'

    def build_dataset(self) -> Dataset:
        """
        Builds and returns the SUGARCREPE dataset.

        Returns:
            Dataset: The SUGARCREPE dataset with sequences of text and images.
        """
        dataset = load_dataset(
            path="Mayfull/sugarcrepepp_vlms",
            trust_remote_code=True,
            split=self.split,
            token=str(os.getenv("HF_TOKEN"))
        )

        return dataset