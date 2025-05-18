import os
from typing import Optional

from datasets import load_dataset, Dataset

from src.builders.base import BaseBuilder
from src.common import registry


@registry.register_builder('WhatsUpBuilder')
class WhatsUpBuilder(BaseBuilder):
    """
    A builder class for creating the WhatsUp dataset.

    Attributes:
        split (Optional[str]): The dataset split to load (default: 'train').
        name (Optional[str]): The name of the dataset (default: 'WhatsUp').
    """
    split: Optional[str] = 'test'
    name: Optional[str] = 'WhatsUp'

    def build_dataset(self) -> Dataset:
        """
        Builds and returns the WhatsUp dataset.

        Returns:
            Dataset: The WhatsUp dataset with sequences of text and images.
        """
        dataset = load_dataset(
            path="Mayfull/whats_up_vlms",
            trust_remote_code=True,
            split=self.split,
            token=str(os.getenv("HF_TOKEN"))
        )

        return dataset