from typing import Literal

import pytest
import torch

from dynopsi.model.modules.embedders import DynoPsiSingleConditionalPairEmbedder
from dynopsi.model.modules.heads import DynoPsiOutputHead
from dynopsi.model.modules.protein_module import ProteinModule
from dynopsi.model.modules.trunks import DynoPsiTransformerTrunk


@pytest.fixture
def device() -> Literal["cuda", "cpu"]:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def test_configs() -> dict:
    """Common test parameters used across test cases."""
    config_shared = dict(
        dim_single=48,
        attn_bias_mode="pair",
        dim_pair=24,
        dim_cond=16,
        n_layers=2,
    )
    config_embedder = dict(
        embed_dim=config_shared["dim_single"],
        pair_embed_dim=config_shared["dim_pair"],
        cond_embed_dim=config_shared["dim_cond"],
        time_dim=64,
        use_self_cond=False,
        scalar_feats=None,
        categorical_feats=None,
        per_residue_scalar_feats=None,
        per_residue_categorical_feats={"chain_breaks": 2},
        index_embed_dim=None,
        num_time_embeddings=1,
        num_distance_bins=64,
        num_distance_bins_sc=128,
        num_seq_sep_bins=127,
    )
    config_attention_transition_block = dict(
        n_heads=2,
        dim_head=12,
        expansion_factor=2,
        use_qk_bias=True,
        use_qk_layernorm=True,
    )
    config_pair_update = dict(
        n_updates=1,
        use_triangle_mult=True,
        dim_triangle=8,
        transition_n=2,
    )
    config_distogram_head = dict(
        no_bins=64,
        use_layer_norm=True,
    )
    config_coords_decoder_head = dict(
        use_layer_norm=True,
    )
    return {
        "batch_size": 3,
        "seq_len": 10,
        "num_atoms": 3,
        "config_shared": config_shared,
        "config_embedder": config_embedder,
        "config_attention_transition_block": config_attention_transition_block,
        "config_pair_update": config_pair_update,
        "config_distogram_head": config_distogram_head,
        "config_coords_decoder_head": config_coords_decoder_head,
    }


class TestDynoPsiModel:
    def test_outputs(self, test_configs: dict, device: Literal["cuda", "cpu"]) -> None:
        model = ProteinModule(
            embedder=DynoPsiSingleConditionalPairEmbedder(
                **test_configs["config_embedder"], num_atoms=test_configs["num_atoms"]
            ),
            main_trunk=DynoPsiTransformerTrunk(
                config_shared=test_configs["config_shared"],
                config_attention_transition_block=test_configs["config_attention_transition_block"],
                config_pair_update=test_configs["config_pair_update"],
            ),
            output_head=DynoPsiOutputHead(
                num_atoms=test_configs["num_atoms"],
                config_coords_decoder_head={
                    **test_configs["config_shared"],
                    **test_configs["config_coords_decoder_head"],
                },
                config_distogram_head={**test_configs["config_shared"], **test_configs["config_distogram_head"]},
            ),
        )
        model.to(device)

        # Construct test batch
        t = torch.rand(test_configs["batch_size"], device=device)
        coords = torch.randn(
            test_configs["batch_size"],
            test_configs["seq_len"],
            test_configs["num_atoms"],
            3,
            device=device,
        )
        indices = torch.arange(test_configs["seq_len"], device=device).repeat(test_configs["batch_size"], 1)
        asym_id = torch.randint(0, 3, (test_configs["batch_size"], test_configs["seq_len"]), device=device)
        residue_index = torch.randint(0, 20, (test_configs["batch_size"], test_configs["seq_len"]), device=device)
        batch = {
            "coords": coords,
            "token_index": indices,
            "t": t,
            "asym_id": asym_id,
            "residue_index": residue_index,
        }

        output = model(batch)
        assert output["out_coords"].shape == (
            test_configs["batch_size"],
            test_configs["seq_len"],
            test_configs["num_atoms"],
            3,
        )
        assert output["distogram_prediction"].shape == (
            test_configs["batch_size"],
            test_configs["seq_len"],
            test_configs["seq_len"],
            test_configs["num_atoms"],
            test_configs["num_atoms"],
            test_configs["config_distogram_head"]["no_bins"],
        )

    # Make sure the index shuffling affects the model outputs as expected
    def test_outputs_shuffled_chains(self, test_configs: dict, device: Literal["cuda", "cpu"]) -> None:
        model = ProteinModule(
            embedder=DynoPsiSingleConditionalPairEmbedder(
                **test_configs["config_embedder"], num_atoms=test_configs["num_atoms"]
            ),
            main_trunk=DynoPsiTransformerTrunk(
                config_shared=test_configs["config_shared"],
                config_attention_transition_block=test_configs["config_attention_transition_block"],
                config_pair_update=test_configs["config_pair_update"],
            ),
            output_head=DynoPsiOutputHead(
                num_atoms=test_configs["num_atoms"],
                config_coords_decoder_head={
                    **test_configs["config_shared"],
                    **test_configs["config_coords_decoder_head"],
                },
                config_distogram_head={**test_configs["config_shared"], **test_configs["config_distogram_head"]},
            ),
        )
        model.to(device)

        # Construct test inputs
        batch_size = 2
        seq_len = 10
        num_atoms = test_configs["num_atoms"]
        t = torch.rand(batch_size, device=device)
        coords = torch.randn(batch_size, seq_len, num_atoms, 3, device=device)
        # Test asym_id cases: single chain with 2 padding positions, three differently sizes chains with one padding position
        asym_id = torch.tensor([[0] * seq_len, [0] * 2 + [1] * 5 + [2] * 3], device=device)
        residue_index = torch.tensor([[0] * seq_len, [0] * 2 + [1] * 5 + [2] * 3], device=device)

        shuffled_indices = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
            ],
            device=device,
        )

        shuffled_indices_batch = {
            "coords": coords,
            "token_index": shuffled_indices,
            "t": t,
            "asym_id": asym_id,
            "residue_index": residue_index,
        }
        shuffled_indices_output = model(shuffled_indices_batch)

        shuffled_order = torch.argsort(shuffled_indices, dim=-1)
        for i in range(batch_size):
            shuffled_tokens_batch = {
                "coords": coords[i][shuffled_order[i]].unsqueeze(0),
                "token_index": shuffled_indices[i][shuffled_order[i]].unsqueeze(0),
                "t": t[i].reshape([1]),
                "asym_id": asym_id[i][shuffled_order[i]].unsqueeze(0),
                "residue_index": residue_index[i][shuffled_order[i]].unsqueeze(0),
            }
            shuffled_tokens_output = model(shuffled_tokens_batch)

            # check that the output is the same
            assert torch.allclose(
                shuffled_indices_output["out_coords"][i][shuffled_order[i]],
                shuffled_tokens_output["out_coords"].squeeze(),
                atol=1e-6,
            )
            assert torch.allclose(
                shuffled_indices_output["distogram_prediction"][i][shuffled_order][:, shuffled_order],
                shuffled_tokens_output["distogram_prediction"].squeeze(),
                atol=1e-6,
            )
