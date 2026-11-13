import pytest
import torch

from dynopsi.model.nn.layers import (
    AdaptiveAttentionTransitionBlock,
    AdaptiveBiasedMultiHeadAttention,
    AdaptiveLayerNorm,
    AdaptiveScale,
    AdaptiveTransition,
    AttentionPairBias,
    PairUpdate,
)


def get_random_inputs_pair_rep() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = 2
    seq_len = 10
    dim_seq = 4
    dim_cond = 4
    dim_pair = 4

    single_rep = torch.randn(batch_size, seq_len, dim_seq)
    pair_rep = torch.randn(batch_size, seq_len, seq_len, dim_pair)
    seq_cond = torch.randn(batch_size, seq_len, dim_cond)
    mask = torch.ones(batch_size, seq_len)

    return single_rep, pair_rep, seq_cond, mask


@pytest.fixture
def random_inputs_pair_rep() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return get_random_inputs_pair_rep()


def alter_masked_positions_pair_rep(
    single_rep: torch.Tensor, pair_rep: torch.Tensor, seq_cond: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    single_mask = mask.unsqueeze(-1)
    single_rep_altered = single_rep * single_mask + (1 - single_mask) * 100 * torch.randn_like(single_rep)
    seq_cond_altered = seq_cond * single_mask + (1 - single_mask) * 100 * torch.randn_like(seq_cond)
    pair_mask = single_mask.unsqueeze(-1) * single_mask.unsqueeze(-2)
    pair_rep_altered = pair_rep * pair_mask + (1 - pair_mask) * 100 * torch.randn_like(pair_rep)
    return single_rep_altered, pair_rep_altered, seq_cond_altered


@pytest.fixture
def altered_inputs_pair_rep(
    random_inputs_pair_rep: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return alter_masked_positions_pair_rep(*random_inputs_pair_rep)


def test_adaptive_layer_norm_masking(
    random_inputs_pair_rep: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    altered_inputs_pair_rep: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    single_rep, pair_rep, seq_cond, mask = random_inputs_pair_rep
    single_rep_altered, pair_rep_altered, seq_cond_altered = altered_inputs_pair_rep
    layer = AdaptiveLayerNorm(dim=single_rep.shape[-1], dim_cond=seq_cond.shape[-1])
    output = layer(single_rep, seq_cond, mask)
    altered_output = layer(single_rep_altered, seq_cond_altered, mask)
    assert torch.allclose(output, altered_output)


def test_adaptive_scale_masking(
    random_inputs_pair_rep: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    altered_inputs_pair_rep: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    single_rep, pair_rep, seq_cond, mask = random_inputs_pair_rep
    single_rep_altered, pair_rep_altered, seq_cond_altered = altered_inputs_pair_rep
    layer = AdaptiveScale(dim=single_rep.shape[-1], dim_cond=seq_cond.shape[-1])
    output = layer(single_rep, seq_cond, mask)
    altered_output = layer(single_rep_altered, seq_cond_altered, mask)
    assert torch.allclose(output, altered_output)


def test_attention_pair_bias_masking__pair_rep(
    random_inputs_pair_rep: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    altered_inputs_pair_rep: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    single_rep, pair_rep, seq_cond, mask = random_inputs_pair_rep
    single_rep_altered, pair_rep_altered, seq_cond_altered = altered_inputs_pair_rep
    layer = AttentionPairBias(
        node_dim=single_rep.shape[-1],
        dim_head=3,
        heads=2,
        bias=True,
        dim_pair=pair_rep.shape[-1],
        attn_bias_mode="pair",
    )
    output = layer(single_rep, pair_rep, mask=mask)
    # Only pair_rep masking is tested here; single_rep masking is tested in
    # test_adaptive_biased_multihead_attention_masking.
    altered_output = layer(single_rep, pair_rep_altered, mask=mask)
    assert torch.allclose(output, altered_output, atol=1e-7)


def test_adaptive_biased_multihead_attention_masking__pair_rep(
    random_inputs_pair_rep: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    altered_inputs_pair_rep: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    single_rep, pair_rep, seq_cond, mask = random_inputs_pair_rep
    single_rep_altered, pair_rep_altered, seq_cond_altered = altered_inputs_pair_rep
    layer = AdaptiveBiasedMultiHeadAttention(
        dim_seq=single_rep.shape[-1],
        dim_cond=seq_cond.shape[-1],
        n_heads=2,
        dim_head=2,
        dim_pair=pair_rep.shape[-1],
        attn_bias_mode="pair",
    )
    output = layer(single_rep, seq_cond, pair_rep=pair_rep, mask=mask)
    altered_output = layer(single_rep_altered, seq_cond_altered, pair_rep=pair_rep_altered, mask=mask)
    assert torch.allclose(output, altered_output, atol=1e-7)


def test_adaptive_transition_masking(
    random_inputs_pair_rep: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    altered_inputs_pair_rep: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    single_rep, pair_rep, seq_cond, mask = random_inputs_pair_rep
    single_rep_altered, pair_rep_altered, seq_cond_altered = altered_inputs_pair_rep
    layer = AdaptiveTransition(dim=single_rep.shape[-1], dim_cond=seq_cond.shape[-1])
    output = layer(single_rep, seq_cond, mask)
    altered_output = layer(single_rep_altered, seq_cond_altered, mask)
    assert torch.allclose(output, altered_output, atol=1e-7)


def test_adaptive_attention_transition_block_masking__pair_rep(
    random_inputs_pair_rep: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    altered_inputs_pair_rep: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    single_rep, pair_rep, seq_cond, mask = random_inputs_pair_rep
    single_rep_altered, pair_rep_altered, seq_cond_altered = altered_inputs_pair_rep
    layer = AdaptiveAttentionTransitionBlock(
        dim_single=single_rep.shape[-1],
        dim_cond=seq_cond.shape[-1],
        n_heads=2,
        dim_head=2,
        dim_pair=pair_rep.shape[-1],
        attn_bias_mode="pair",
    )
    output = layer(single_rep, seq_cond, pair_rep=pair_rep, mask=mask)
    altered_output = layer(single_rep_altered, seq_cond_altered, pair_rep=pair_rep_altered, mask=mask)
    assert torch.allclose(output, altered_output, atol=1e-7)


def test_pair_update_masking(
    random_inputs_pair_rep: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    altered_inputs_pair_rep: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    single_rep, pair_rep, seq_cond, mask = random_inputs_pair_rep
    single_rep_altered, pair_rep_altered, seq_cond_altered = altered_inputs_pair_rep
    layer = PairUpdate(
        dim_single=single_rep.shape[-1],
        dim_pair=pair_rep.shape[-1],
        use_triangle_mult=True,
    )
    output = layer(single_rep, pair_rep, mask)
    altered_output = layer(single_rep_altered, pair_rep_altered, mask)
    assert torch.allclose(output, altered_output, atol=1e-7)
