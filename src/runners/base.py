import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from typing import TypeVar

import torch
from PIL.Image import Image
from datasets import Dataset
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedModel
from transformers import Trainer
from transformers.utils import logging

from src.common.registry import registry
from src.utils import dummy_collator

# from src.collators import BaseCollator

logger = logging.get_logger(__name__)

# CollatorType = Type[BaseCollator]
CollatorType = TypeVar("CollatorType", bound="BaseCollator")

__all__ = ["BaseTrainer", "BaseEvaluator"]


class BaseTrainer(Trainer):
    """
    A subclass of the Hugging Face `Trainer` class, designed to be extended with additional
    training logic and customized behavior.
    """
    pass


@dataclass
class BaseEvaluator:
    """
    A base class for model evaluation.

    Attributes:
        model (Optional[PreTrainedModel]): The model to be evaluated.
        data_collator (Optional[CollatorType]): Custom collator for preparing the data.
        builder_cls_name (Optional[str]): The builder class name for dataset creation.
        overwrite_results (Optional[bool]): Whether to overwrite existing results in the output directory.
        output_dir (Optional[Union[str, os.PathLike]]): Directory where evaluation results will be saved.
    """
    model: Optional[PreTrainedModel] = None
    data_collator: Optional[CollatorType] = None
    builder_cls_name: Optional[str] = None
    overwrite_results: Optional[bool] = False
    output_dir: Optional[Union[str, os.PathLike]] = None
    default_filename: Optional[str] = None

    def __post_init__(self):
        """
        Validates initialization and prepares the model for evaluation.
        """
        if not self.model:
            raise ValueError("A model must be provided for evaluation.")
        self.model.eval()
        logger.info(f"Preparing {self.builder_cls_name.replace('Builder', '').upper()} Benchmark")

    def _get_eval_dataset(self) -> Dataset:
        """
        Retrieves the dataset for evaluation based on the builder class.

        Returns:
            Dataset: The evaluation dataset.
        """
        if self.builder_cls_name is None:
            raise ValueError("A Builder must be provided for evaluation.")
        builder_cls = registry.get_builder_class(self.builder_cls_name)

        if not builder_cls:
            logger.error(f"Model class {self.builder_cls_name} not registered.")
            raise ValueError(f"Model class {self.builder_cls_name} not registered.")

        return builder_cls().build_dataset()

    def _prepare_dataloader(self):
        """
        Prepares a DataLoader for the evaluation dataset.

        Returns:
            DataLoader: The prepared DataLoader.
        """
        dataset = self._get_eval_dataset()

        dataloader = tqdm(
            DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=dummy_collator
            )
        )
        dataloader.set_description(f"Computing {self.builder_cls_name.replace('Builder', '')} scores")

        return dataloader

    def evaluate(self):
        """
        Abstract method for evaluation. Must be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement the evaluate method.")

    def _prepare_inputs(
            self,
            images: Optional[List[Image]] = None,
            text: Optional[List[str]] = None
    ) -> BatchEncoding:
        """
        Prepares inputs for the model by collating and moving to the correct device.

        Args:
            images (List[Image]): List of image inputs.
            text (List[str]): List of text inputs.

        Returns:
            BatchEncoding: The prepared inputs.
        """
        inputs: BatchEncoding = self.data_collator({"images": images, "text": text})
        return inputs.to(self.model.device)

    @torch.no_grad()
    def _compute_similarity(
            self,
            images: List[Image],
            text: List[str],
            return_outputs: bool = False
    ) -> Union[Tensor | tuple[Tensor, Any]]:
        """
        Computes similarity scores between images and text.

        Args:
            images (List[Image]): List of image inputs.
            text (List[str]): List of text inputs.
            return_outputs (bool): Whether to return the output of the model (default: 'False').

        Returns:
            torch.Tensor: Similarity scores between image and text embeddings.
            outputs: Output of the model.
        """
        inputs = self._prepare_inputs(images, text)
        outputs = self.model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        if not return_outputs:
            return torch.matmul(image_embeds, text_embeds.t().to(image_embeds.device))
        else:
            return torch.matmul(image_embeds, text_embeds.t().to(image_embeds.device)), outputs

    @torch.no_grad()
    def get_image_to_text_score(
            self,
            images: List[Image],
            text: List[str],
            return_tot: bool = False,
            return_raw_scores: bool = False
    ) -> tuple[int, Any, int, list[int | float | bool]] | tuple[int, Any] | int | tuple[int, int]:
        """
        Determines if the first text matches the single image.

        Args:
            images (List[Image]): A single image wrapped in a list.
            text (List[str]): Multiple text inputs.
            return_tot (bool): if return tot score (default: 'False').
            return_raw_scores (bool): if return raw similarity scores (default: 'False').

        Returns:
            int: 1 if the first text matches the image, otherwise 0.
            or
            tuple: (score, similarity_scores, text_similarity_scores)
        """
        if len(images) != 1:
            raise ValueError("`get_image_to_text_score` requires exactly one image.")
        if len(text) <= 1:
            raise ValueError("`get_image_to_text_score` requires more than one text input.")

        similarity_scores, outputs = self._compute_similarity(images, text, return_outputs=True)

        if return_raw_scores:
            if not return_tot:
                return int(similarity_scores.argmax() == 0), similarity_scores
            else:
                i0_c0 = similarity_scores[0, 0].item()
                i0_c1 = similarity_scores[0, 1].item()
                i0_c2 = similarity_scores[0, 2].item()

                image_correct = i0_c0 > i0_c2 and i0_c1 > i0_c2

                text_similarity_scores = torch.matmul(outputs.text_embeds, outputs.text_embeds.t())
                c0_c1 = text_similarity_scores[0, 1].item()
                c0_c2 = text_similarity_scores[0, 2].item()
                c1_c2 = text_similarity_scores[1, 2].item()

                text_correct = c0_c1 > c0_c2 and c0_c1 > c1_c2

                return int(image_correct), similarity_scores, int(text_correct), [c0_c1, c0_c2, c1_c2]
        else:
            if not return_tot:
                return int(similarity_scores.argmax() == 0)
            else:
                i0_c0 = similarity_scores[0, 0].item()
                i0_c1 = similarity_scores[0, 1].item()
                i0_c2 = similarity_scores[0, 2].item()

                image_correct = i0_c0 > i0_c2 and i0_c1 > i0_c2

                text_similarity_scores = torch.matmul(outputs.text_embeds, outputs.text_embeds.t())

                c0_c1 = text_similarity_scores[0, 1].item()
                c0_c2 = text_similarity_scores[0, 2].item()
                c1_c2 = text_similarity_scores[1, 2].item()

                text_correct = c0_c1 > c0_c2 and c0_c1 > c1_c2

                return int(image_correct), int(text_correct)

    @torch.no_grad()
    def get_text_to_image_score(self, images: List[Image], text: List[str]) -> int:
        """
        Determines if the first image matches the single text.

        Args:
            images (List[Image]): Multiple images.
            text (List[str]): A single text input wrapped in a list.

        Returns:
            int: 1 if the first image matches the text, otherwise 0.
        """
        if len(images) <= 1:
            raise ValueError("`get_text_to_image_score` requires more than one image.")
        if len(text) != 1:
            raise ValueError("`get_text_to_image_score` requires exactly one text input.")

        similarity_scores = self._compute_similarity(images, text)
        return int(similarity_scores.argmax() == 0)

    @torch.no_grad()
    def get_group_score(self, images: List[Image], text: List[str]) -> Dict[str, bool]:
        """
        Computes correctness scores for a group of 2 images and 2 text inputs.

        Args:
            images (List[Image]): A list of exactly 2 images.
            text (List[str]): A list of exactly 2 text inputs.

        Returns:
            Dict[str, bool]: A dictionary containing correctness scores.
        """
        if len(images) != 2:
            raise ValueError("`get_group_score` requires exactly 2 images.")
        if len(text) != 2:
            raise ValueError("`get_group_score` requires exactly 2 text inputs.")

        similarity_scores = self._compute_similarity(images, text)

        # Extract scores
        c0_i0 = similarity_scores[0, 0].item()
        c0_i1 = similarity_scores[0, 1].item()
        c1_i0 = similarity_scores[1, 0].item()
        c1_i1 = similarity_scores[1, 1].item()

        # Compute correctness
        text_correct = c0_i0 > c1_i0 and c1_i1 > c0_i1
        image_correct = c0_i0 > c0_i1 and c1_i1 > c1_i0
        group_correct = text_correct and image_correct

        return {
            "text_correct": text_correct,
            "image_correct": image_correct,
            "group_correct": group_correct,
        }

    def _save_result(self, result: Dict, filename: Optional[str] = None) -> None:
        """
        Saves evaluation results to a JSON file.

        Args:
            result (Dict): The evaluation results.
            filename (Optional[str]): The name of the result file.
        """
        if not self.output_dir:
            raise ValueError("Output directory is not set.")

        os.makedirs(self.output_dir, exist_ok=True)
        file_path = os.path.join(self.output_dir, f"{filename or self.default_filename}.json")

        try:
            with open(file_path, "w") as f:
                json.dump(result, f, indent=2)
        except IOError as e:
            raise IOError(f"Failed to save results to {file_path}: {e}")

    def _save_scores_to_csv(self, scores_list, filename: str) -> None:
        if not self.output_dir:
            raise ValueError("Output directory is not set.")

        os.makedirs(self.output_dir, exist_ok=True)
        file_path = os.path.join(self.output_dir, f"{filename}.csv")

        try:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for row in scores_list:
                    writer.writerow(row)
            logger.info(f"Scores saved to {file_path}")
        except IOError as e:
            raise IOError(f"Failed to save scores to {file_path}: {e}")
