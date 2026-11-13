"""Portions of this file are derived from the La Proteina project.

La Proteina
Copyright (c) 2025 NVIDIA
Licensed under the Apache License, Version 2.0.
Source: https://github.com/NVIDIA-Digital-Bio/la-proteina

Modifications to the La Proteina-derived code were made by Dyno Therapeutics, Inc. (2026).
"""

import logging
from typing import Any, cast

import torch
import torch.nn as nn

from dynopsi.model.nn.layers import AdaptiveAttentionTransitionBlock, PairUpdate

logger = logging.getLogger(__name__)


class MainTrunk(nn.Module):
    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("MainTrunk is not implemented")


class DynoPsiTransformerTrunk(MainTrunk):
    """Primary neural network processing stack.

    Follows Figure 5d of Geffner et al. (2025).
    The trunk is comprised of a series of pair-biased multihead attention,
    transition, and (optional) triangle update layers to update single and pair representations.

    Args:
        config_shared: dict containing shared hyperparameters for the trunk that includes channel dimension for each of the single, conditioning, and pair representations, total number of layers
        config_attention_transition_block: dict containing hyperparameters for the adaptive biased multi-headattention and transition block (Figure 5e)
        config_pair_update: dict containing hyperparameters for the pair update block (Figure 24a)

    The pair representation is injected into attention via attn_bias_mode (config_shared["attn_bias_mode"]):
    - "pair": use the pair representation to bias self-attention
    """

    def __init__(
        self,
        config_shared: dict[str, Any],
        config_attention_transition_block: dict[str, Any],
        config_pair_update: dict[str, Any] | None = None,
    ):
        super().__init__()

        self.n_layers = config_shared["n_layers"]
        self.adaptive_attention_transition_blocks = nn.ModuleList(
            [
                AdaptiveAttentionTransitionBlock(
                    dim_single=config_shared["dim_single"],
                    dim_cond=config_shared["dim_cond"],
                    n_heads=config_attention_transition_block["n_heads"],
                    dim_head=config_attention_transition_block["dim_head"],
                    attn_bias_mode=config_shared.get("attn_bias_mode", "none"),
                    dim_pair=config_shared.get("dim_pair", None),
                    expansion_factor=config_attention_transition_block["expansion_factor"],
                    use_qkv_bias=config_attention_transition_block.get("use_qkv_bias", True),
                    use_qk_layernorm=config_attention_transition_block["use_qk_layernorm"],
                    use_flash_attn=config_attention_transition_block.get("use_flash_attn", False),
                    flash_attn_dtype=config_attention_transition_block.get("flash_attn_dtype", "bf16"),
                )
                for _ in range(self.n_layers)
            ]
        )

        if config_pair_update is not None:
            self.update_pair_rep = True
            self._init_pair_update(config_shared, config_pair_update)
        else:
            self.update_pair_rep = False

    def _init_pair_update(self, config_shared: dict[str, Any], config_pair_update: dict[str, Any]) -> None:
        self.n_pair_updates = config_pair_update["n_updates"]
        assert self.n_pair_updates <= self.n_layers, (
            f"n_pair_updates ({self.n_pair_updates}) must be less than n_layers ({self.n_layers})"
        )
        if self.n_layers % self.n_pair_updates != 0:
            logger.warning(
                "n_layers (%s) is not divisible by n_pair_updates (%s). "
                "This will result in unevenly spaced pair updates.",
                self.n_layers,
                self.n_pair_updates,
            )
        self.pair_update_layers = nn.ModuleList(
            [self._init_right_module(config_shared, config_pair_update) for _ in range(self.n_pair_updates)]
        )
        self.pair_update_every_n = self.n_layers // self.n_pair_updates

    def _init_right_module(self, config_shared: dict[str, Any], config_pair_update: dict[str, Any]) -> torch.nn.Module:
        attn_bias_mode = config_shared["attn_bias_mode"]
        if attn_bias_mode != "pair":
            raise ValueError(f"Unrecognized attn_bias_mode '{attn_bias_mode}'")
        dim_pair = config_shared.get("dim_pair")
        if dim_pair is None:
            raise ValueError("dim_pair is required when attn_bias_mode is 'pair'")
        return PairUpdate(
            dim_single=config_shared["dim_single"],
            dim_pair=cast(int, dim_pair),
            use_triangle_mult=config_pair_update["use_triangle_mult"],
            dim_triangle=config_pair_update["dim_triangle"],
            transition_n=config_pair_update["transition_n"],
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Run the model trunk.

        Args:
            batch: dictionary containing input tensors
                - "single_rep" real-values tensor of shape (batch, seq_len, dim_single)
                - "conditional_feats" real-valued tensor of shape (batch, seq_len, dim_cond)
                - "pair_rep" real-valued tensor of shape (batch, seq_len, seq_len, dim_pair) | None
                - "mask" binary tensor of shape (batch, seq_len)

        Returns
        -------
            batch dictionary containing outputs from the trunk
                - "single_rep" real-valued tensor of final single representation, shape (batch, seq_len, dim_single)
                - "pair_rep" real-valued tensor of final pair representation
                - "mask" copy of the binary tensor mask (batch, seq_len)
        """
        mask = batch["mask"]
        single_rep = batch["single_rep"]
        conditional_feats = batch["conditional_feats"]
        pair_rep = batch.get("pair_rep", None)

        for i in range(self.n_layers):
            single_rep = self.adaptive_attention_transition_blocks[i](
                single_rep=single_rep,
                seq_cond=conditional_feats,
                pair_rep=pair_rep,
                mask=mask,
            )

            if self.update_pair_rep and pair_rep is not None:
                n_pair_updates_by_i = 1 + i // self.pair_update_every_n
                if i % self.pair_update_every_n == 0 and n_pair_updates_by_i <= self.n_pair_updates:
                    cur_pair_update_idx = n_pair_updates_by_i - 1
                    pair_rep = self.pair_update_layers[cur_pair_update_idx](
                        single_rep,
                        pair_rep,
                        mask=mask,
                    )
            elif self.update_pair_rep and pair_rep is None:
                raise ValueError("Trunk was initialized to expect non-None pair_rep, but pair_rep = None was passed.")

        return {"single_rep": single_rep, "pair_rep": pair_rep, "mask": mask}
