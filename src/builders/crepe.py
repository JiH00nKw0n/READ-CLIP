import os
from typing import Optional

from datasets import load_dataset, Dataset

from src.builders.base import BaseBuilder
from src.common import registry


@registry.register_builder('CrepeBuilder')
class CrepeBuilder(BaseBuilder):
    """
    A builder class for creating the Crêpe dataset.

    Attributes:
        split (Optional[str]): The dataset split to load (default: 'train').
        name (Optional[str]): The name of the dataset (default: 'Crêpe').
    """
    split: Optional[str] = 'test'
    name: Optional[str] = 'Crepe'

    def build_dataset(self) -> Dataset:
        """
        Builds and returns the CRÊPE dataset.

        Returns:
            Dataset: The CRÊPE dataset with sequences of text and images.
        """
        dataset = load_dataset(
            path="Mayfull/crepe_vlms",
            trust_remote_code=True,
            split=self.split,
        )

        return dataset