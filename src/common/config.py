from typing import Dict, List, Optional

from omegaconf import OmegaConf, DictConfig
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

__all__ = ["TrainConfig", "EvaluateConfig"]


@dataclass(
    config=ConfigDict(
        extra='ignore', frozen=True, strict=True, validate_assignment=True
    )
)
class BaseConfig:
    """
    Base configuration class that contains settings for model, processor, dataset, collator, and run configurations.

    Args:
        model (Dict):
            Dictionary containing model configuration settings.
        processor (Dict):
            Dictionary containing processor configuration settings.
        dataset (Dict):
            Dictionary containing dataset configuration settings.
        run (Dict):
            Dictionary containing run settings.

    Properties:
        model_config (DictConfig):
            Returns the model configuration as an `OmegaConf` object.
        processor_config (DictConfig):
            Returns the processor configuration as an `OmegaConf` object.
        run_config (DictConfig):
            Returns the run configuration as an `OmegaConf` object.
    """
    model: Dict
    processor: Dict
    run: Dict

    @property
    def model_config(self) -> DictConfig:
        return OmegaConf.create(self.model)

    @property
    def processor_config(self) -> DictConfig:
        return OmegaConf.create(self.processor)

    @property
    def run_config(self) -> DictConfig:
        return OmegaConf.create(self.run)


@dataclass(
    config=ConfigDict(
        extra='ignore', frozen=True, strict=True, validate_assignment=True
    )
)
class TrainConfig(BaseConfig):
    """
    Configuration class for training, which extends `BaseConfig` by adding trainer-specific settings.

    Args:
        trainer (Dict):
            Dictionary containing trainer configuration settings.
        collator (Dict):
            Dictionary containing collator configuration settings.
        dataset (Dict):
            Dictionary containing dataset configuration settings.

    Properties:
        trainer_config (Dict):
            Returns the trainer configuration as a dictionary.
        collator_config (DictConfig):
            Returns the collator configuration as an `OmegaConf` object.
        dataset_config (Dict):
            Returns the dataset configuration as a Python dictionary.
    """
    collator: Dict
    trainer: Dict
    dataset: Dict

    @property
    def trainer_config(self) -> Dict:
        return self.trainer

    @property
    def collator_config(self) -> DictConfig:
        return OmegaConf.create(self.collator)

    @property
    def dataset_config(self) -> Dict:
        return self.dataset


@dataclass(
    config=ConfigDict(
        extra='ignore', frozen=True, strict=True, validate_assignment=True
    )
)
class EvaluateConfig(BaseConfig):
    """
    Configuration class for evaluation, which extends `BaseConfig` by adding evaluator-specific settings.

    Args:
        evaluator (List):
            List containing evaluator configuration settings.

    Properties:
        evaluator_config (List):
            Returns the evaluator configuration as a list.
    """
    evaluator: Optional[List]

    @property
    def evaluator_config(self) -> List:
        return self.evaluator
