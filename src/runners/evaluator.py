import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Type

from transformers.utils import logging

from src.collators import BaseCollator
from src.common.registry import registry
from .base import BaseEvaluator

logger = logging.get_logger(__name__)

CollatorType = Type[BaseCollator]

__all__ = [
    "AROEvaluator",
    "CrepeEvaluator",
    "SugarCrepeEvaluator",
    "SugarCrepePPEvaluator",
    "ValseEvaluator",
    "WhatsUpEvaluator",
    "WinogroundEvaluator"
]


@dataclass
@registry.register_evaluator("CrepeEvaluator")
class CrepeEvaluator(BaseEvaluator):
    builder_cls_name: Optional[str] = 'CrepeBuilder'
    default_filename: str = 'CREPE'

    def evaluate(self):
        if os.path.isfile(os.path.join(self.output_dir, f"{self.default_filename}.json")):
            logger.info(
                f"The file '{self.default_filename}.json' already exists. Skipping transfer or consider renaming."
            )
            return None
        dataloader = self._prepare_dataloader()

        result = defaultdict(list)

        # 점수 저장을 위한 리스트 생성
        itt_scores_list = []

        for sample in dataloader:
            images = [
                i.crop(
                    (
                        sample["x"],
                        sample["y"],
                        sample["x"] + sample["width"],
                        sample["y"] + sample["height"]
                    )
                ) for i in sample["images"]
            ]

            text = [*sample["positive_caption"], *sample["negative_caption"]]

            # 원시 점수도 반환
            score, similarity_scores = self.get_image_to_text_score(images, text, return_raw_scores=True)

            # 텐서를 리스트로 변환하여 저장
            itt_scores_list.append(similarity_scores.flatten().tolist())

            result[sample["original_file_name"]].append(score)

        average_result = {
            key: 100 * round(sum(values) / len(values), 5) if values else 0 for key, values in result.items()
        }
        average_result.update({"total": round(sum(average_result.values()) / len(average_result), 3)})

        # 결과와 점수 저장
        self._save_result(average_result)
        self._save_scores_to_csv(itt_scores_list, f"{self.default_filename}_itt")

    def _save_result(self, result: Dict, filename: str = 'CREPE') -> None:
        super()._save_result(result, filename)


@dataclass
@registry.register_evaluator("SugarCrepeEvaluator")
class SugarCrepeEvaluator(BaseEvaluator):
    builder_cls_name: Optional[str] = 'SugarCrepeBuilder'
    default_filename: str = 'SUGARCREPE'

    def evaluate(self):
        if os.path.isfile(os.path.join(self.output_dir, f"{self.default_filename}.json")):
            logger.info(
                f"The file '{self.default_filename}.json' already exists. Skipping transfer or consider renaming."
            )
            return None
        dataloader = self._prepare_dataloader()

        result = defaultdict(list)

        # 점수 저장을 위한 리스트 생성
        itt_scores_list = []

        for sample in dataloader:
            images = sample["images"]
            text = [*sample["positive_caption"], *sample["negative_caption"]]

            # 원시 점수도 반환
            score, similarity_scores = self.get_image_to_text_score(images, text, return_raw_scores=True)

            # 텐서를 리스트로 변환하여 저장
            itt_scores_list.append(similarity_scores.flatten().tolist())

            result[sample["original_file_name"]].append(score)

        average_result = {
            key: 100 * round(sum(values) / len(values), 5) if values else 0 for key, values in result.items()
        }
        average_result.update({"total": round(sum(average_result.values()) / len(average_result), 3)})

        # 결과와 점수 저장
        self._save_result(average_result)
        self._save_scores_to_csv(itt_scores_list, f"{self.default_filename}_itt")

    def _save_result(self, result: Dict, filename: str = 'SUGARCREPE') -> None:
        super()._save_result(result, filename)


@dataclass
@registry.register_evaluator("SugarCrepePPEvaluator")
class SugarCrepePPEvaluator(BaseEvaluator):
    builder_cls_name: Optional[str] = 'SugarCrepePPBuilder'
    default_filename: str = 'SUGARCREPEPP'

    def evaluate(self):
        if os.path.isfile(os.path.join(self.output_dir, f"{self.default_filename}.json")):
            logger.info(
                f"The file '{self.default_filename}.json' already exists. Skipping transfer or consider renaming."
            )
            return None
        dataloader = self._prepare_dataloader()

        result = defaultdict(list)

        # 점수 저장을 위한 리스트 생성
        itt_scores_list = []
        tot_scores_list = []

        for sample in dataloader:
            images = sample["images"]
            text = [*sample["positive_caption_1"], *sample["positive_caption_2"], *sample["negative_caption"]]

            # 원시 점수도 반환
            itt_score, similarity_scores, tot_score, text_similarity_scores = self.get_image_to_text_score(
                images, text, return_tot=True, return_raw_scores=True
            )

            # 텐서를 리스트로 변환하여 저장
            itt_scores_list.append(similarity_scores.flatten().tolist())
            tot_scores_list.append(text_similarity_scores)

            result[sample["original_file_name"] + "_itt"].append(itt_score)
            result[sample["original_file_name"] + "_tot"].append(tot_score)

        average_result = {
            key: 100 * round(sum(values) / len(values), 5) if values else 0 for key, values in result.items()
        }
        average_result.update(
            {
                "total_itt": round(
                    sum(v for k, v in average_result.items() if k.endswith("itt")) / sum(
                        k.endswith("itt") for k in average_result
                    ), 3
                )
            }
        )
        average_result.update(
            {
                "total_tot": round(
                    sum(v for k, v in average_result.items() if k.endswith("tot")) / sum(
                        k.endswith("tot") for k in average_result
                    ), 3
                )
            }
        )

        # 결과와 점수 저장
        self._save_result(average_result)
        self._save_scores_to_csv(itt_scores_list, f"{self.default_filename}_itt")
        self._save_scores_to_csv(tot_scores_list, f"{self.default_filename}_tot")

    def _save_result(self, result: Dict, filename: str = 'SUGARCREPEPP') -> None:
        super()._save_result(result, filename)


@dataclass
@registry.register_evaluator("ValseEvaluator")
class ValseEvaluator(BaseEvaluator):
    builder_cls_name: Optional[str] = 'ValseBuilder'
    default_filename: str = 'VALSE'

    def evaluate(self):
        if os.path.isfile(os.path.join(self.output_dir, f"{self.default_filename}.json")):
            logger.info(
                f"The file '{self.default_filename}.json' already exists. Skipping transfer or consider renaming."
            )
            return None
        dataloader = self._prepare_dataloader()

        result = defaultdict(list)

        # 점수 저장을 위한 리스트 생성
        itt_scores_list = []

        for sample in dataloader:
            images = sample["images"]
            text = [*sample["positive_caption"], *sample["negative_caption"]]

            # 원시 점수도 반환
            score, similarity_scores = self.get_image_to_text_score(images, text, return_raw_scores=True)

            # 텐서를 리스트로 변환하여 저장
            itt_scores_list.append(similarity_scores.flatten().tolist())

            result[sample["linguistic_phenomena"]].append(score)
            result["total"].append(score)

        average_result = {
            key: 100 * round(sum(values) / len(values), 5) if values else 0 for key, values in result.items()
        }

        # 결과와 점수 저장
        self._save_result(average_result)
        self._save_scores_to_csv(itt_scores_list, f"{self.default_filename}_itt")

    def _save_result(self, result: Dict, filename: str = 'VALSE') -> None:
        super()._save_result(result, filename)


@dataclass
@registry.register_evaluator("WhatsUpEvaluator")
class WhatsUpEvaluator(BaseEvaluator):
    builder_cls_name: Optional[str] = 'WhatsUpBuilder'
    default_filename: str = 'WHATSUP'

    def evaluate(self):
        if os.path.isfile(os.path.join(self.output_dir, f"{self.default_filename}.json")):
            logger.info(
                f"The file '{self.default_filename}.json' already exists. Skipping transfer or consider renaming."
            )
            return None
        dataloader = self._prepare_dataloader()

        result = defaultdict(list)

        # 점수 저장을 위한 리스트 생성
        itt_scores_list = []

        for sample in dataloader:
            images = sample["images"]
            text = [*sample["positive_caption"], *sample["negative_caption"]]

            # 원시 점수도 반환
            score, similarity_scores = self.get_image_to_text_score(images, text, return_raw_scores=True)

            # 텐서를 리스트로 변환하여 저장
            itt_scores_list.append(similarity_scores.flatten().tolist())

            if sample["original_file_name"].startswith("coco"):
                name = "coco"
            elif sample["original_file_name"].startswith("vg"):
                name = "vg"
            else:
                name = "whatsup"

            result[name].append(score)

        average_result = {
            key: 100 * round(sum(values) / len(values), 5) if values else 0 for key, values in result.items()
        }
        average_result.update({"total": round(sum(average_result.values()) / len(average_result), 3)})

        # 결과와 점수 저장
        self._save_result(average_result)
        self._save_scores_to_csv(itt_scores_list, f"{self.default_filename}_itt")

    def _save_result(self, result: Dict, filename: str = 'WHATSUP') -> None:
        super()._save_result(result, filename)


@dataclass
@registry.register_evaluator("WinogroundEvaluator")
class WinogroundEvaluator(BaseEvaluator):
    builder_cls_name: Optional[str] = 'WinogroundBuilder'
    default_filename: str = 'WINOGROUND'

    def evaluate(self):
        if os.path.isfile(os.path.join(self.output_dir, f"{self.default_filename}.json")):
            logger.info(
                f"The file '{self.default_filename}.json' already exists. Skipping transfer or consider renaming."
            )
            return None
        dataloader = self._prepare_dataloader()

        result = defaultdict(list)

        for sample in dataloader:
            images = [sample["image_0"], sample["image_1"]]
            text = [sample["caption_0"], sample["caption_1"]]

            score = self.get_group_score(images, text)

            result['text_score'].append(score['text_correct'])
            result['image_score'].append(score['image_correct'])
            result['group_score'].append(score['group_correct'])

        average_result = {
            key: 100 * round(sum(values) / len(values), 5) if values else 0 for key, values in result.items()
        }

        self._save_result(average_result)

    def _save_result(self, result: Dict, filename: str = 'WINOGROUND') -> None:
        super()._save_result(result, filename)
