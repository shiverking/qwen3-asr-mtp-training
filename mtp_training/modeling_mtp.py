from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers.masking_utils import create_causal_mask

from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
    Qwen3ASRThinkerTextRMSNorm,
)

from .objectives import (
    BranchResult,
    align_branch_with_backbone,
    branch_cross_entropy,
    normalized_branch_weights,
    shifted_targets,
)


@dataclass
class MTPOutput:
    loss: torch.Tensor
    main_loss: torch.Tensor
    main_correct: torch.Tensor
    main_valid: torch.Tensor
    main_predictions: torch.Tensor
    main_token_losses: torch.Tensor
    branch_losses: list[torch.Tensor]
    branch_token_losses: list[torch.Tensor]
    branch_correct: list[torch.Tensor]
    branch_valid: list[torch.Tensor]
    branch_predictions: list[torch.Tensor]
    branch_backbone_correct: list[torch.Tensor]
    branch_backbone_valid: list[torch.Tensor]


class MTPBranch(nn.Module):
    def __init__(self, text_config, decoder_layer: nn.Module, layer_index: int):
        super().__init__()
        hidden_size = text_config.hidden_size
        self.hidden_norm = Qwen3ASRThinkerTextRMSNorm(hidden_size, eps=text_config.rms_norm_eps)
        self.embedding_norm = Qwen3ASRThinkerTextRMSNorm(hidden_size, eps=text_config.rms_norm_eps)
        self.projection = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        nn.init.normal_(
            self.projection.weight,
            mean=0.0,
            std=text_config.initializer_range,
        )
        self.decoder_layer = copy.deepcopy(decoder_layer)
        self.decoder_layer.self_attn.layer_idx = layer_index

    def forward(
        self,
        previous_hidden: torch.Tensor,
        shifted_embedding: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        text_model: nn.Module,
        position_offset: int = 0,
    ) -> torch.Tensor:
        fused = self.projection(
            torch.cat(
                [self.hidden_norm(previous_hidden), self.embedding_norm(shifted_embedding)],
                dim=-1,
            )
        )
        sequence_length = fused.shape[1]
        cache_position = torch.arange(sequence_length, device=fused.device)
        branch_position_ids = position_ids[
            ..., position_offset : position_offset + sequence_length
        ]
        text_position_ids = branch_position_ids[0]
        causal_mask = create_causal_mask(
            config=text_model.config,
            input_embeds=fused,
            attention_mask=attention_mask[:, :sequence_length],
            cache_position=cache_position,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        position_embeddings = text_model.rotary_emb(fused, branch_position_ids)
        return self.decoder_layer(
            fused,
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )


class Qwen3ASRMTPModel(nn.Module):
    """Attach ParaASR-style serial MTP branches without patching Qwen3-ASR."""

    def __init__(
        self,
        asr_model: nn.Module,
        depth: int = 3,
        alpha: float = 0.9,
        branch_position_mode: str = "base",
        loss_reduction: str = "token_mean",
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if branch_position_mode not in ("base", "shifted"):
            raise ValueError("branch_position_mode must be base or shifted")
        if loss_reduction not in ("token_mean", "sample_mean"):
            raise ValueError("loss_reduction must be token_mean or sample_mean")
        self.asr_model = asr_model
        self.depth = depth
        self.alpha = alpha
        self.branch_position_mode = branch_position_mode
        self.loss_reduction = loss_reduction
        thinker = self.thinker
        text_model = thinker.model
        self.branches = nn.ModuleList(
            MTPBranch(
                text_model.config,
                text_model.layers[-1],
                len(text_model.layers) + branch_index,
            )
            for branch_index in range(depth)
        )

    @property
    def thinker(self):
        return self.asr_model.thinker

    def configure_trainable(self, stage: int) -> dict[str, int]:
        if stage not in (1, 2):
            raise ValueError("stage must be 1 or 2")
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.branches.parameters():
            parameter.requires_grad = True
        if stage == 2:
            for parameter in self.thinker.model.parameters():
                parameter.requires_grad = True
            for module in (self.thinker.audio_tower.proj1, self.thinker.audio_tower.proj2):
                for parameter in module.parameters():
                    parameter.requires_grad = True
            for parameter in self.thinker.lm_head.parameters():
                parameter.requires_grad = True
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.parameters())
        return {"trainable": trainable, "total": total}

    def _backbone_hidden(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        thinker = self.thinker
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        inputs_embeds = thinker.get_input_embeddings()(input_ids)
        if batch.get("input_features") is not None:
            audio_features = thinker.get_audio_features(
                batch["input_features"],
                feature_attention_mask=batch.get("feature_attention_mask"),
                audio_feature_lengths=batch.get("audio_feature_lengths"),
            )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            audio_mask = thinker.get_placeholder_mask(input_ids, inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)
        position_ids, _ = thinker.get_rope_index(attention_mask)
        outputs = thinker.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
        )
        return outputs.last_hidden_state, position_ids

    def forward(self, stage: int, **batch) -> MTPOutput:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        loss_mask = batch["loss_mask"]
        main_hidden, position_ids = self._backbone_hidden(batch)
        main_targets, main_valid = shifted_targets(input_ids, loss_mask, 1)
        main_result = branch_cross_entropy(
            main_hidden[:, :-1],
            self.thinker.lm_head,
            main_targets,
            main_valid,
            self.loss_reduction,
        )

        branch_results: list[BranchResult] = []
        previous_hidden = main_hidden
        token_embedding = self.thinker.get_input_embeddings()
        for branch_index, branch in enumerate(self.branches, start=1):
            previous_hidden = branch(
                previous_hidden=previous_hidden[:, :-1],
                shifted_embedding=token_embedding(input_ids[:, branch_index:]),
                attention_mask=attention_mask,
                position_ids=position_ids,
                text_model=self.thinker.model,
                position_offset=(
                    branch_index if self.branch_position_mode == "shifted" else 0
                ),
            )
            targets, valid = shifted_targets(input_ids, loss_mask, branch_index + 1)
            branch_results.append(
                branch_cross_entropy(
                    self.thinker.model.norm(previous_hidden[:, :-1]),
                    self.thinker.lm_head,
                    targets,
                    valid,
                    self.loss_reduction,
                )
            )

        weights = normalized_branch_weights(self.depth, self.alpha, main_hidden.device)
        mtp_loss = sum(weight * result.loss for weight, result in zip(weights, branch_results))
        loss = mtp_loss if stage == 1 else main_result.loss + mtp_loss
        backbone_alignment = [
            align_branch_with_backbone(result, main_result, branch_index)
            for branch_index, result in enumerate(branch_results, start=1)
        ]
        return MTPOutput(
            loss=loss,
            main_loss=main_result.loss,
            main_correct=main_result.correct,
            main_valid=main_result.valid,
            main_predictions=main_result.predicted,
            main_token_losses=main_result.token_losses,
            branch_losses=[result.loss for result in branch_results],
            branch_token_losses=[result.token_losses for result in branch_results],
            branch_correct=[result.correct for result in branch_results],
            branch_valid=[result.valid for result in branch_results],
            branch_predictions=[result.predicted for result in branch_results],
            branch_backbone_correct=[item[0] for item in backbone_alignment],
            branch_backbone_valid=[item[1] for item in backbone_alignment],
        )
