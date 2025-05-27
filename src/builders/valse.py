import os
from typing import Optional

from datasets import load_dataset, Dataset

from src.builders.base import BaseBuilder
from src.common import registry


@registry.register_builder('ValseBuilder')
class ValseBuilder(BaseBuilder):
    """
    A builder class for creating the Valse dataset.

    Attributes:
        split (Optional[str]): The dataset split to load (default: 'train').
        name (Optional[str]): The name of the dataset (default: 'Valse').
    """
    split: Optional[str] = 'test'
    name: Optional[str] = 'Valse'

    def build_dataset(self) -> Dataset:
        """
        Builds and returns the Valse dataset.

        Returns:
            Dataset: The Valse dataset with sequences of text and images.
        """
        dataset = load_dataset(
            path="Mayfull/valse_vlms",
            trust_remote_code=True,
            split=self.split,
        )

        return dataset