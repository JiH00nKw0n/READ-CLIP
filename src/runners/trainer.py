import copy
import logging
import os
from itertools import zip_longest
from typing import Optional, Dict, Union, Callable, List, Tuple, Any

import safetensors
import torch
from peft import PeftModel
from torch import nn
from torch.utils.data import Dataset, RandomSampler, IterableDataset
from transformers import (
    is_torch_xla_available, PreTrainedModel, TrainingArguments,
    DataCollator, PreTrainedTokenizerBase, BaseImageProcessor,
    FeatureExtractionMixin, ProcessorMixin, TrainerCallback, is_torch_xpu_available, is_torch_mlu_available,
    is_torch_musa_available, is_torch_npu_available, is_apex_available
)
from transformers.trainer import TRAINING_ARGS_NAME, TRAINER_STATE_NAME
from transformers.trainer_callback import ExportableState
from transformers.trainer_utils import has_length, EvalPrediction
from transformers.utils import WEIGHTS_NAME, SAFE_WEIGHTS_NAME, is_peft_available, is_torch_mps_available, \
    is_accelerate_available

from src.common.registry import registry
from src.runners.base import BaseTrainer
from src.utils import contrastive_loss

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
if is_accelerate_available():
    from accelerate.utils import (
        DistributedType,
    )

if is_apex_available():
    from apex import amp

__all__ = [
    "RandomSamplerTrainer",
    "NegCLIPRandomSamplerTrainer",
    "ReadCLIPTrainer"
]

logger = logging.getLogger(__name__)


def neg_clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative CLIP loss by combining the caption and image loss.
    """
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t()[:len(similarity)])
    return (caption_loss + image_loss) / 2.0


class MultiLossMixin:
    """
    Mixin to handle multiple losses (LM loss + Contrastive loss) and logging their averages.
    Also provides a method to compute token-level accuracy.
    """

    def update_multi_loss_state(self, lm_loss: torch.Tensor, cont_loss: torch.Tensor, text_cont_loss: torch.Tensor):
        self.state.lm_loss = getattr(self.state, "lm_loss", torch.tensor(0.0, device=lm_loss.device))
        self.state.contrastive_loss = getattr(
            self.state, "contrastive_loss", torch.tensor(0.0, device=cont_loss.device)
        )
        self.state.text_contrastive_loss = getattr(
            self.state, "text_contrastive_loss", torch.tensor(0.0, device=text_cont_loss.device)
        )

        self.state.lm_loss += lm_loss.detach()
        self.state.contrastive_loss += cont_loss.detach()
        self.state.text_contrastive_loss += text_cont_loss.detach()

    def log_multi_loss(self, logs: Dict[str, float], steps: int):
        if hasattr(self.state, "lm_loss") and hasattr(self.state, "contrastive_loss") and hasattr(
                self.state, "text_contrastive_loss"
        ):
            tr_lm_loss_scalar = self._nested_gather(self.state.lm_loss).mean().item()
            tr_contrastive_loss_scalar = self._nested_gather(self.state.contrastive_loss).mean().item()
            tr_text_cont_loss_scalar = self._nested_gather(self.state.text_contrastive_loss).mean().item()

            logs["lm_loss"] = round(tr_lm_loss_scalar / steps, 4)
            logs["contrastive_loss"] = round(tr_contrastive_loss_scalar / steps, 4)
            logs["text_contrastive_loss"] = round(tr_text_cont_loss_scalar / steps, 4)

            # reset
            self.state.lm_loss -= self.state.lm_loss
            self.state.contrastive_loss -= self.state.contrastive_loss
            self.state.text_contrastive_loss -= self.state.text_contrastive_loss

    def compute_token_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute accuracy treating -100 as padding, converting them to 0 and then treating 0 as padding.
        """
        prediction = logits.argmax(dim=-1)
        labels = labels.clone()
        labels[labels == -100] = 0

        non_pad_mask = (labels != 0)

        correct = (prediction == labels) & non_pad_mask
        return correct.float().masked_select(non_pad_mask).mean().detach()

    def compute_log_probs_and_update_metrics(
            self, logits: torch.Tensor, labels: torch.Tensor, prefix: str
    ):
        """
        Compute log probs (and token-level probabilities) for given logits and labels.
        Update trainer state with these metrics using the given prefix ("positive" or "negative").
        """
        if logits is None or labels is None:
            return
        # labels_clone: labels 복사본
        labels = labels.clone()
        # -100을 vocab 내 유효한 토큰(예: 패딩 토큰 인덱스)으로 변경
        labels[labels == -100] = 0

        valid_mask = (labels != 0)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        gathered_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)  # (batch, seq_len)
        gathered_log_probs = gathered_log_probs * valid_mask

        avg_log_prob = gathered_log_probs.sum(dim=-1).mean()  # sum over seq, mean over batch
        avg_token_prob = gathered_log_probs.exp()[valid_mask].mean()

        acc = self.compute_token_accuracy(logits, labels)

        def update_state(name, value):
            if value is not None:
                setattr(self.state, name, getattr(self.state, name, torch.tensor(0.0, device=labels.device)) + value)

        update_state(f"train_{prefix}_log_prob", avg_log_prob)
        update_state(f"train_{prefix}_token_prob", avg_token_prob)
        update_state(f"train_{prefix}_token_accuracy", acc)

    def log_token_metrics(self, logs: Dict[str, float], steps: int):
        for prefix in ["positive", "negative"]:
            for metric_name in [
                f"train_{prefix}_log_prob", f"train_{prefix}_token_prob", f"train_{prefix}_token_accuracy"
            ]:
                if hasattr(self.state, metric_name):
                    val = self._nested_gather(getattr(self.state, metric_name)).mean().item()
                    logs[metric_name] = round(val / steps, 4)
                    setattr(self.state, metric_name, torch.tensor(0.0, device=self.args.device))


@registry.register_trainer('RandomSamplerTrainer')
class RandomSamplerTrainer(BaseTrainer):
    """
    Trainer that uses a RandomSampler for the training dataset.
    """

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if self.args.group_by_length:
            raise ValueError("Argument `group_by_length` must be `False`.")
        else:
            generator = torch.Generator()
            generator.manual_seed(2024)

            return RandomSampler(self.train_dataset, generator=generator)


@registry.register_trainer('NegCLIPRandomSamplerTrainer')
class NegCLIPRandomSamplerTrainer(RandomSamplerTrainer):
    """
    Trainer that integrates negative CLIP loss.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(
            inputs, **{
                'return_dict': True,
                'return_loss': False,
            }
        )
        outputs = model(**inputs)
        loss = neg_clip_loss(outputs.logits_per_image)
        return (loss, outputs) if return_outputs else loss


@registry.register_trainer('ReadCLIPTrainer')
class ReadCLIPTrainer(MultiLossMixin, RandomSamplerTrainer):
    """
    Trainer that combines negative CLIP loss, multiple losses (LM + Contrastive),
    and logs token-level probabilities for positive/negative contexts.
    Negative는 optional. Reference는 없음.
    """

    def __init__(
            self, model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            processing_class: Optional[
                Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            compute_loss_func: Optional[Callable] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
                    None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            training_weight: float = 1.0,
            text_weight: float = 0.0,
            save_decoder: bool = True,
            use_negative_in_text_contrastive: bool = False,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
        self.training_weight = training_weight
        self.text_weight = text_weight
        self.negative_weight = 0.0
        self.save_decoder = save_decoder
        self.use_negative_in_text_contrastive = use_negative_in_text_contrastive

    def set_training_weight(self, training_weight: float):
        self.training_weight = training_weight
        logger.info(f"Current training weight: {self.training_weight}")

    def set_text_weight(self, text_weight: float):
        self.text_weight = text_weight
        logger.info(f"Current text weight: {self.text_weight}")

    def set_save_decoder(self, save_decoder: bool = True):
        self.save_decoder = save_decoder
        logger.info(f"Saving Decoder: {self.save_decoder}")

    def set_use_negative_in_text_contrastive(self, use_negative_in_text_contrastive: bool = True):
        self.use_negative_in_text_contrastive = use_negative_in_text_contrastive
        logger.info(f"Use Negative in Text Contrastive Loss: {self.use_negative_in_text_contrastive}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(
            inputs, **{
                'return_dict': True,
            }
        )

        decoder_input_ids = inputs.pop("decoder_input_ids", [None])
        labels = inputs.pop("labels", [None])
        negative_decoder_input_ids = inputs.pop("negative_decoder_input_ids", [None])
        negative_labels = inputs.pop("negative_labels", [None])
        paraphrased_input_ids = inputs.pop("paraphrased_input_ids", None)
        paraphrased_attention_mask = inputs.pop("paraphrased_attention_mask", None)

        loss = torch.tensor(0.0, device=self.model.device)
        cont_loss = torch.tensor(0.0, device=self.model.device)
        lm_loss = torch.tensor(0.0, device=self.model.device)
        text_cont_loss = torch.tensor(0.0, device=self.model.device)

        logger.info(f"text weight: {self.text_weight}")
        logger.info(f"training weight: {self.training_weight}")

        num_labels = float(len(labels))
        for i, (sub_decoder_input_ids, sub_labels, sub_negative_decoder_input_ids, sub_negative_labels) in enumerate(
                zip_longest(
                    decoder_input_ids, labels, negative_decoder_input_ids, negative_labels
                )
        ):
            _inputs = {
                **inputs,
                "decoder_input_ids": [sub_decoder_input_ids] if sub_decoder_input_ids is not None else None,
                "labels": [sub_labels] if sub_labels is not None else None,
                "negative_decoder_input_ids": [
                    sub_negative_decoder_input_ids] if sub_negative_decoder_input_ids is not None else None,
                "negative_labels": [sub_negative_labels] if sub_negative_labels is not None else None,
            }

            output = model(**_inputs)
            if i == 0:
                logits = output.lm_logits

            _cont_loss = output.cont_loss
            _positive_lm_loss = output.positive_lm_loss
            _negative_lm_loss = output.negative_lm_loss

            _lm_loss = _positive_lm_loss - self.negative_weight * _negative_lm_loss
            _loss = _cont_loss + self.training_weight * _lm_loss

            logger.info(f"Positive LM Loss: {_positive_lm_loss}")
            logger.info(f"Negative LM Loss: {_negative_lm_loss}")
            logger.info(f"LM Loss: {_lm_loss}")

            self._custom_backward(_loss / num_labels)

            cont_loss += _cont_loss.detach() / num_labels
            lm_loss += _lm_loss.detach() / num_labels
            loss += _loss.detach() / num_labels

        if paraphrased_input_ids is not None and paraphrased_attention_mask is not None:
            _inputs = {
                "input_ids": inputs.get("input_ids", None),
                "attention_mask": inputs.get("attention_mask", None),
                "paraphrased_input_ids": paraphrased_input_ids,
                "paraphrased_attention_mask": paraphrased_attention_mask,
                "use_negative_in_text_contrastive": self.use_negative_in_text_contrastive,
            }

            _text_cont_loss = model.module.get_text_contrastive_loss(**_inputs)

            self._custom_backward(self.text_weight * _text_cont_loss)

            text_cont_loss += _text_cont_loss.detach()
            loss += self.text_weight * _text_cont_loss.detach()

        self.update_multi_loss_state(lm_loss, cont_loss, text_cont_loss)

        # Compute log probs and metrics if exists
        if labels[0] is not None:
            self.compute_log_probs_and_update_metrics(logits[0], labels[0], prefix="positive")

        return loss

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_loss -= tr_loss
            steps = self.state.global_step - self._globalstep_last_logged

            logs["loss"] = round(tr_loss_scalar / steps, 4)
            self.log_multi_loss(logs, steps)
            self.log_token_metrics(logs, steps)  # token-level metrics

            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs, start_time)

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if self.save_decoder:
            _output_dir = output_dir + "_"
            os.makedirs(_output_dir, exist_ok=True)
            logger.info(f"Saving full model checkpoint to {_output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving clip model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            state_dict = self.model.clip.state_dict()
            if self.save_decoder:
                _state_dict = self.model.state_dict()

            if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                self.accelerator.unwrap_model(self.model.clip).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
                if self.save_decoder:
                    self.accelerator.unwrap_model(copy.deepcopy(self.model).to("cpu")).save_pretrained(
                        _output_dir, state_dict=_state_dict, safe_serialization=self.args.save_safetensors
                    )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            state_dict = self.model.clip.state_dict()
            self.model.clip.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )
            if self.save_decoder:
                _state_dict = self.model.state_dict()
                torch.save(_state_dict, os.path.join(_output_dir, WEIGHTS_NAME))

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        for cb in [
            cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
        ]:
            cb_name = cb.__class__.__name__
            cb_state = cb.state()
            if isinstance(self.state.stateful_callbacks[cb_name], list):
                self.state.stateful_callbacks[cb_name].append(cb_state)
            else:
                self.state.stateful_callbacks[cb_name] = cb_state
        # Good practice: save your training arguments together with the trained model

        self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        if self.save_decoder:
            self.state.save_to_json(os.path.join(_output_dir, TRAINER_STATE_NAME))
            torch.save(self.args, os.path.join(_output_dir, TRAINING_ARGS_NAME))

    def training_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
        if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        return loss.detach()

    def _custom_backward(self, loss):

        kwargs = {}

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # Finally we need to normalize the loss for reporting
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)
