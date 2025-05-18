from transformers import CLIPModel, CLIPConfig, CLIPProcessor, T5Config

from src.common.registry import registry
from .configuration_clipwithdecoder import CLIPWithDecoderConfig
from .modeling_clipwithdecoder import CLIPWithDecoderModel

__all__ = [
    "CLIPWithDecoderConfig",
    "CLIPWithDecoderModel",
    "CLIPModel",
    "CLIPConfig",
    "CLIPProcessor",
]

registry.register_model("CLIPModel")(CLIPModel)
registry.register_model_config("CLIPConfig")(CLIPConfig)
registry.register_processor("CLIPProcessor")(CLIPProcessor)
registry.register_model_config("T5Config")(T5Config)
