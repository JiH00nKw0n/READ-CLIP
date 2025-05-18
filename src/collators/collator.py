import json
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

import torch
from PIL import Image
from src.common.mixin import NegCLIPNegativeTextMining
from transformers import BatchEncoding
from transformers.utils import add_end_docstrings
from transformers.utils import logging

from src.common.registry import registry
from .base import BaseCollator, BASE_COLLATOR_DOCSTRING

logger = logging.get_logger(__name__)

__all__ = [
    "ImageCollator",
    "COCOImageCollatorWithT5Tokenizer",
]


@add_end_docstrings(BASE_COLLATOR_DOCSTRING)
@registry.register_collator('ImageCollator')
class ImageCollator(BaseCollator):
    """
    Collator for processing input data containing 'images' and 'text' keys.

    The 'images' key should hold a list of `PIL.Image` objects, and the 'text' key should hold a list of strings.
    This collator processes the images and text, applies padding and truncation, and prepares
    the batch for model input.

    Raises:
        TypeError:
            If the 'images' key contains objects not of type `PIL.Image.Image`, or if the 'text' key
            contains objects not of type `str`.
        ValueError:
            If the input does not contain exactly the keys 'images' and 'text', or if the corresponding values
            are not lists.
    """

    def __call__(self, inputs: Dict[str, Any]) -> BatchEncoding:
        """
        Processes a batch of input dictionaries with 'images' and 'text' keys.

        Validates that 'images' is a list of `PIL.Image.Image` objects and 'text' is a list of strings,
        then encodes the processed data with the specified processor.

        Args:
            inputs (`Dict[str, Any]`):
                A dictionary containing 'images' and 'text' keys with corresponding list values.

        Returns:
            `BatchEncoding`:
                A batch of processed and encoded inputs.

        Raises:
            TypeError:
                If the data types of the values are invalid.
            ValueError:
                If the input dictionary contains invalid keys or the values are not lists.
        """

        def _validate_and_process_images(images: Any) -> List[Image]:
            """
            Validates that the 'images' key contains a list of `PIL.Image.Image` objects.

            Args:
                images (`Any`): The value associated with the 'images' key.

            Returns:
                `List[PIL.Image.Image]`: Processed list of images.

            Raises:
                TypeError: If the value is not a list or contains non-image elements.
            """
            if not isinstance(images, list):
                raise TypeError(f"Expected `list` for key 'images', but got {type(images)}.")
            for img in images:
                if not isinstance(img, Image):
                    raise TypeError(f"Expected elements of 'images' to be `PIL.Image.Image`, but got {type(img)}.")
            return [img.convert("RGB") for img in images]

        def _validate_and_process_texts(texts: Any) -> List[str]:
            """
            Validates that the 'text' key contains a list of strings.

            Args:
                texts (`Any`): The value associated with the 'text' key.

            Returns:
                `List[str]`: Processed list of texts.

            Raises:
                TypeError: If the value is not a list or contains non-string elements.
            """
            if not isinstance(texts, list):
                raise TypeError(f"Expected `list` for key 'text', but got {type(texts)}.")
            for text in texts:
                if not isinstance(text, str):
                    raise TypeError(f"Expected elements of 'text' to be `str`, but got {type(text)}.")
            return texts

        # Validate keys
        allowed_keys = {'images', 'text'}
        if set(inputs.keys()) != allowed_keys:
            raise ValueError(f"Input dictionary must only contain the keys 'images' and 'text'. Found: {inputs.keys()}")

        # Validate and process inputs
        processed_images = _validate_and_process_images(inputs['images'])
        processed_texts = _validate_and_process_texts(inputs['text'])

        # Create arguments for the processor
        kwargs = {
            'return_tensors': self.return_tensors,
            'padding': self.padding,
            'truncation': self.truncation,
            'max_length': self.max_length,
            'pad_to_multiple_of': self.pad_to_multiple_of,
        }

        processor_input = {'images': processed_images, 'text': processed_texts, **kwargs}

        return self.processor(**processor_input)


@add_end_docstrings(BASE_COLLATOR_DOCSTRING)
@dataclass
@registry.register_collator('COCONegCLIPImageCollatorWithTokenizer')
class COCOImageCollatorWithT5Tokenizer(BaseCollator, NegCLIPNegativeTextMining):
    tokenizer: Optional[str] = None
    num_hard_negs: Optional[int] = 0
    num_labels: Optional[int] = 0
    use_negative_caption: Optional[bool] = False
    use_synthetic_caption: Optional[bool] = False
    use_different_label: Optional[bool] = False
    use_text_loss: Optional[bool] = False
    synthetic_type: Optional[str] = None
    synthetic_data: Optional[Dict] = None

    def __post_init__(self):
        from transformers import AutoTokenizer

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xl')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)

        NegCLIPNegativeTextMining.__init__(self)
        self.synthetic_data = {}
        if (self.synthetic_type is not None
                and self.synthetic_type != 'None'
                and os.path.isfile(f"./data/{self.synthetic_type}.json")
        ):
            data = {}
            logger.info(f"Paraphrased label loaded: ./data/{self.synthetic_type}.json")
            with open(f"./data/{self.synthetic_type}.json", "r", encoding="utf-8") as f:
                data.update(json.load(f))
            self.synthetic_data = data

    def __call__(self, inputs: List[Dict[str, Any]]) -> BatchEncoding:
        valid_images = []
        valid_texts = []
        valid_neg_texts = []
        valid_labels = []
        valid_paraphrased_texts = []

        for i, input_dict in enumerate(inputs):
            if (
                    'images' in input_dict and 'sentences' and 'cocoid' in input_dict
            ):
                cocoid = input_dict['cocoid']
                _image = input_dict['images'].convert("RGB")
                captions = input_dict['sentences'][:]

                if self.use_text_loss:
                    if self.synthetic_type is not None and str(cocoid) in self.synthetic_data:
                        if self.use_synthetic_caption:
                            if i == 0:
                                logger.info(
                                    "(use_synthetic_caption) We will use synthetic caption for text contrastive loss."
                                )
                            _text = str(
                                self.rng.choice(
                                    captions,
                                    size=1,
                                    replace=False
                                ).tolist()[0]
                            )
                            # 2. paraphrased_text 후보 리스트 생성 (captions + synthetic_data)
                            candidate_list = [*captions[:], *self.synthetic_data[str(cocoid)]]

                            if _text in candidate_list:
                                candidate_list.remove(_text)

                            # 4. _paraphrased_text 추출
                            _paraphrased_text = str(
                                self.rng.choice(
                                    candidate_list,
                                    size=1,
                                    replace=False
                                ).tolist()[0]
                            )
                        else:
                            if i == 0:
                                logger.info("(use_synthetic_label) We will use synthetic label only for loss.")
                            candidates = self.rng.choice(captions, size=2, replace=False)
                            _text = str(candidates[0])
                            _paraphrased_text = str(candidates[1])
                        valid_paraphrased_texts.append(_paraphrased_text)
                    else:
                        if i == 0:
                            logger.info("We will not use synthetic data for loss.")
                        if self.synthetic_type is not None:
                            if i == 0:
                                logger.info(
                                    (f"{self.synthetic_type} is provided but `{cocoid}` not in synthetic data."
                                     f"We will use the original caption.")
                                )
                        candidates = self.rng.choice(captions, size=2, replace=False).tolist()
                        _text = str(candidates[0])
                        _paraphrased_text = str(candidates[1])
                        valid_paraphrased_texts.append(_paraphrased_text)
                else:
                    _text = str(self.rng.choice(captions, size=1, replace=False)[0])

                valid_images.append(_image)
                valid_texts.append(_text)

                if self.use_different_label:
                    label_captions = captions[:]

                    if self.num_labels > 0:
                        _labels = self.rng.choice(label_captions, size=self.num_labels, replace=False).tolist()
                        valid_labels.append(_labels)
                    elif self.num_labels == 0:
                        pass
                    else:
                        raise ValueError("Please specify the number of labels.")
                else:
                    if self.num_labels == 0:
                        pass
                    elif self.num_labels == 1:
                        _labels = [_text]
                        valid_labels.append(_labels)
                    else:
                        raise ValueError(
                            f"If self.use_different_label is {self.use_different_label}, "
                            f"self.num_labels must be 1."
                        )

                if self.use_negative_caption:
                    if self.num_hard_negs is None or self.num_hard_negs <= 0:
                        raise ValueError("Please specify the valid number of hard negatives.")
                    else:
                        valid_neg_texts.extend(self.negative_caption(_text) for _ in range(self.num_hard_negs))
            else:
                raise ValueError("Invalid input dictionary. Please check the input dictionary.")

        valid_texts.extend(valid_neg_texts)
        logger.info(f"images: {len(valid_images)}")
        logger.info(f"texts: {len(valid_texts)}")
        logger.info(f"paraphrased texts: {len(valid_paraphrased_texts)}")
        logger.info(f"labels: {len(valid_labels)}")

        labels_list = list(map(list, zip(*valid_labels)))

        # Create kwargs for processor, including padding, truncation, etc.
        kwargs = {
            'return_tensors': self.return_tensors,
            'padding': self.padding,
            'truncation': self.truncation,
            'max_length': self.max_length,
            'pad_to_multiple_of': self.pad_to_multiple_of,
        }

        # Create a dictionary to store processed data for processor
        processed_dict = {'images': valid_images, 'text': valid_texts}

        # Merge processed inputs with kwargs and pass to the processor
        processor_input = dict(processed_dict, **kwargs)
        batch_output = self.processor(**processor_input)

        if labels_list and self.num_labels > 0:
            labels = []
            decoder_input_ids = []
            for valid_sub_labels in labels_list:
                sub_label_dict = {'text': valid_sub_labels}
                sub_label_input = dict(sub_label_dict, **kwargs)
                sub_label_output = self.tokenizer(**sub_label_input)
                sub_labels = sub_label_output.data['input_ids'].clone()
                sub_decoder_input_ids = self._shift_right(sub_label_output.data['input_ids'].clone())
                sub_labels[sub_labels == self.tokenizer.pad_token_id] = -100

                labels.append(sub_labels)
                decoder_input_ids.append(sub_decoder_input_ids)

            labels_tensor = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
            batch_output.data['labels'] = labels_tensor
            decoder_input_ids_tensor = torch.nn.utils.rnn.pad_sequence(
                decoder_input_ids, batch_first=True, padding_value=0
            )
            batch_output.data['decoder_input_ids'] = decoder_input_ids_tensor

        if self.use_text_loss:
            paraphrased_dict = {'text': valid_paraphrased_texts}
            paraphrased_input = dict(paraphrased_dict, **kwargs)
            paraphrased_output = self.processor(**paraphrased_input)
            paraphrased_input_ids = paraphrased_output.data['input_ids'].clone()
            paraphrased_attention_mask = paraphrased_output.data['attention_mask'].clone()

            batch_output.data['paraphrased_input_ids'] = paraphrased_input_ids
            batch_output.data['paraphrased_attention_mask'] = paraphrased_attention_mask

        return batch_output
