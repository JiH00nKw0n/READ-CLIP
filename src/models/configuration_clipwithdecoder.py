"""CLIPInstructor model configuration"""

import os
from typing import Optional, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from src.common import registry

logger = logging.get_logger(__name__)
__all__ = ["CLIPWithDecoderConfig"]


@registry.register_model_config("CLIPWithDecoderConfig")
class CLIPWithDecoderConfig(PretrainedConfig):
    model_type = "clip_with_decoder"

    def __init__(
            self,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
            decoder_pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
            use_decoder: Optional[bool] = True,
            **kwargs
    ):
        clip_config = kwargs.pop('clip_config', None)
        decoder_config = kwargs.pop('decoder_config', None)

        super().__init__(**kwargs)

        self.clip_config = clip_config if clip_config is not None else dict(
            {"pretrained_model_name_or_path": pretrained_model_name_or_path}, **kwargs
        )
        self.decoder_config = decoder_config if decoder_config is not None else dict(
            {"pretrained_model_name_or_path": decoder_pretrained_model_name_or_path}, **kwargs
        )

        self.initializer_factor = 1.0
        self.use_decoder = use_decoder