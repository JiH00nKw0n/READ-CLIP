from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
from transformers import (
    PreTrainedModel,
    CLIPConfig, CLIPModel, AutoConfig,
    T5ForConditionalGeneration
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.clip.modeling_clip import _get_vector_norm
from transformers.utils import ModelOutput, logging

from src.common import registry
from .configuration_clipwithdecoder import CLIPWithDecoderConfig
from src.utils.utils import neg_clip_loss

logger = logging.get_logger(__name__)

__all__ = [
    "CLIPWithDecoderOutput",
    "CLIPWithDecoderPreTrainedModel",
    "CLIPWithDecoderModel",
]


@dataclass
class CLIPWithDecoderOutput(ModelOutput):
    cont_loss: Optional[torch.FloatTensor] = None
    positive_lm_loss: Optional[torch.FloatTensor] = None
    negative_lm_loss: Optional[torch.FloatTensor] = None
    lm_logits: Optional[List[torch.FloatTensor]] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None


# Copied from transformers.models.clip.modeling_clip.CLIPPreTrainedModel with CLIP -> Base
class CLIPWithDecoderPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CLIPWithDecoderConfig
    base_model_prefix = "clip_with_decoder"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


@registry.register_model("CLIPWithDecoderModel")
class CLIPWithDecoderModel(CLIPWithDecoderPreTrainedModel):
    config_class = CLIPWithDecoderConfig

    def __init__(self, config: CLIPWithDecoderConfig):
        super().__init__(config)

        clip_config = CLIPConfig.from_pretrained(**config.clip_config)
        decoder_config = AutoConfig.from_pretrained(**config.decoder_config)

        self.decoder_config = decoder_config

        self.projection_dim = clip_config.projection_dim
        self.text_hidden_size = clip_config.text_config.hidden_size

        self.decoder_dim = self.decoder_config.d_model
        if config.use_decoder:
            self.decoder_text_projection = nn.Linear(
                self.text_hidden_size, self.decoder_dim, bias=False
            )
        # Initialize weights and apply final processing
        super().init_weights()

        self.clip = CLIPModel.from_pretrained(**config.clip_config, config=clip_config)

        if config.use_decoder:
            decoder_config.tie_encoder_decoder = True
            self.decoder = T5ForConditionalGeneration.from_pretrained(**config.decoder_config, config=decoder_config)

        super()._backward_compatibility_gradient_checkpointing()

        if config.use_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False

    def get_text_features(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:

        return self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def get_text_contrastive_loss(
            self,
            input_ids: Optional[torch.Tensor] = None,
            paraphrased_input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            paraphrased_attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            paraphrased_position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            use_negative_in_text_contrastive: Optional[bool] = False,
    ):
        if not use_negative_in_text_contrastive:
            logger.info("Using only positive in text contrastive loss")
            input_ids = input_ids[:paraphrased_input_ids.shape[0]]
            attention_mask = attention_mask[:paraphrased_input_ids.shape[0]]

        text_embeds = self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        text_embeds = text_embeds / _get_vector_norm(text_embeds)

        paraphrased_text_embeds = self.clip.get_text_features(
            input_ids=paraphrased_input_ids,
            attention_mask=paraphrased_attention_mask,
            position_ids=paraphrased_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        paraphrased_text_embeds = paraphrased_text_embeds / _get_vector_norm(paraphrased_text_embeds)

        logits = torch.matmul(paraphrased_text_embeds, text_embeds.t().to(text_embeds.device))
        logits = logits * self.clip.logit_scale.exp().to(text_embeds.device)

        loss = nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

        return loss

    def get_image_features(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            interpolate_pos_encoding: bool = False,
            return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:

        return self.clip.get_image_features(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

    def get_clip_output(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        clip_output = self.clip(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            return_loss=False,
        )

        logits_per_image = clip_output.logits_per_image

        cont_loss = neg_clip_loss(logits_per_image)

        clip_output.loss = cont_loss

        encoder_text_outputs = (
            self.decoder_text_projection(
                clip_output.text_model_output.pooler_output[:clip_output.image_embeds.shape[0]]
            ).unsqueeze(1),
        )

        return clip_output, encoder_text_outputs

    def get_decoder_output(
            self,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        decoder_output = self.decoder(
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return decoder_output

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            negative_decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[List[torch.LongTensor]] = None,
            negative_labels: Optional[List[torch.LongTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPWithDecoderOutput]:

        clip_output = self.clip(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits_per_image = clip_output.logits_per_image

        cont_loss = neg_clip_loss(logits_per_image)

        positive_lm_loss = torch.tensor(0.0, device=cont_loss.device)
        negative_lm_loss = torch.tensor(0.0, device=cont_loss.device)
        lm_logits = []

        if labels is not None:
            num_labels = float(len(labels))

            encoder_text_outputs = (
                self.decoder_text_projection(
                    clip_output.text_model_output.pooler_output[:clip_output.image_embeds.shape[0]]
                ).unsqueeze(1),
            )

            for sub_labels, sub_decoder_input_ids in zip(labels, decoder_input_ids):
                decoder_output = self.get_decoder_output(
                    decoder_input_ids=sub_decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    decoder_head_mask=decoder_head_mask,
                    cross_attn_head_mask=cross_attn_head_mask,
                    encoder_outputs=encoder_text_outputs,
                    past_key_values=past_key_values,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    labels=sub_labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

                positive_lm_loss += decoder_output.loss / num_labels
                lm_logits.append(decoder_output.logits)

            if negative_decoder_input_ids is not None and negative_labels is not None:
                for sub_neg_labels, sub_neg_decoder_input_ids in zip(negative_labels, negative_decoder_input_ids):
                    decoder_output = self.get_decoder_output(
                        decoder_input_ids=sub_neg_decoder_input_ids,
                        decoder_attention_mask=decoder_attention_mask,
                        decoder_head_mask=decoder_head_mask,
                        cross_attn_head_mask=cross_attn_head_mask,
                        encoder_outputs=encoder_text_outputs,
                        past_key_values=past_key_values,
                        decoder_inputs_embeds=decoder_inputs_embeds,
                        labels=sub_neg_labels,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                    )

                    negative_lm_loss += decoder_output.loss / num_labels

        if not return_dict:
            output = (
                cont_loss,
                positive_lm_loss,
                negative_lm_loss,
                lm_logits,)
            return output

        return CLIPWithDecoderOutput(
            cont_loss=cont_loss,
            positive_lm_loss=positive_lm_loss,
            negative_lm_loss=negative_lm_loss,
            lm_logits=lm_logits,
            logits_per_text=clip_output.logits_per_text,
            logits_per_image=clip_output.logits_per_image,
            text_embeds=clip_output.text_embeds,
            image_embeds=clip_output.image_embeds,
        )
