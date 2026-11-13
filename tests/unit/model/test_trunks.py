from typing import Literal

import pytest
import torch

from dynopsi.model.modules.trunks import DynoPsiTransformerTrunk


@pytest.fixture
def device() -> Literal["cuda", "cpu"]:
    return "cuda" if torch.cuda.is_available() else "cpu"


config_shared_pair = dict(
    dim_single=32,
    attn_bias_mode="pair",
    dim_pair=24,
    dim_cond=16,
    n_layers=2,
)
config_attention_transition_block = dict(
    n_heads=1,
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
test_configs_dict = {
    "batch_size": 2,
    "seq_len": 30,
    "config_shared_pair": config_shared_pair,
    "config_attention_transition_block": config_attention_transition_block,
    "config_pair_update": config_pair_update,
}


@pytest.fixture
def test_configs() -> dict:
    return test_configs_dict


class TestDynoPsiTransformerTrunk:
    def test_outputs__pair_rep(self, test_configs: dict, device: Literal["cuda", "cpu"]) -> None:
        torch.manual_seed(42)
        trunk = DynoPsiTransformerTrunk(
            config_shared=test_configs["config_shared_pair"],
            config_attention_transition_block=test_configs["config_attention_transition_block"],
            config_pair_update=test_configs["config_pair_update"],
        )
        trunk.to(device)

        single_rep = torch.randn(
            test_configs["batch_size"],
            test_configs["seq_len"],
            test_configs["config_shared_pair"]["dim_single"],
            device=device,
        )
        pair_rep = torch.randn(
            test_configs["batch_size"],
            test_configs["seq_len"],
            test_configs["seq_len"],
            test_configs["config_shared_pair"]["dim_pair"],
            device=device,
        )
        conditional_feats = torch.randn(
            test_configs["batch_size"],
            test_configs["seq_len"],
            test_configs["config_shared_pair"]["dim_cond"],
            device=device,
        )
        mask = torch.randint(0, 2, (test_configs["batch_size"], test_configs["seq_len"]), device=device)

        output = trunk(
            {"single_rep": single_rep, "pair_rep": pair_rep, "conditional_feats": conditional_feats, "mask": mask}
        )

        assert output["single_rep"].shape == (
            test_configs["batch_size"],
            test_configs["seq_len"],
            test_configs["config_shared_pair"]["dim_single"],
        )
        assert output["pair_rep"].shape == (
            test_configs["batch_size"],
            test_configs["seq_len"],
            test_configs["seq_len"],
            test_configs["config_shared_pair"]["dim_pair"],
        )
        assert (mask == output["mask"]).all()

        output_independent = [
            trunk(
                {
                    "single_rep": single_rep[i : i + 1],
                    "pair_rep": pair_rep[i : i + 1],
                    "conditional_feats": conditional_feats[i : i + 1],
                    "mask": mask[i : i + 1],
                }
            )
            for i in range(test_configs["batch_size"])
        ]
        output_independent_collated = {k: torch.cat([v[k] for v in output_independent], dim=0) for k in output.keys()}
        for k in output.keys():
            assert torch.allclose(output[k], output_independent_collated[k], atol=1e-5), (
                f"Trunk should work independently for each batch item for {k}"
            )

    def test_masking__pair_rep(self, test_configs: dict, device: Literal["cuda", "cpu"]) -> None:
        torch.manual_seed(42)
        trunk = DynoPsiTransformerTrunk(
            config_shared=test_configs["config_shared_pair"],
            config_attention_transition_block=test_configs["config_attention_transition_block"],
            config_pair_update=test_configs["config_pair_update"],
        )
        trunk.to(device)

        single_rep = torch.randn(
            test_configs["batch_size"],
            test_configs["seq_len"],
            test_configs["config_shared_pair"]["dim_single"],
            device=device,
        )
        pair_rep = torch.randn(
            test_configs["batch_size"],
            test_configs["seq_len"],
            test_configs["seq_len"],
            test_configs["config_shared_pair"]["dim_pair"],
            device=device,
        )
        conditional_feats = torch.randn(
            test_configs["batch_size"],
            test_configs["seq_len"],
            test_configs["config_shared_pair"]["dim_cond"],
            device=device,
        )
        mask = torch.randint(0, 2, (test_configs["batch_size"], test_configs["seq_len"]), device=device)

        output_base = trunk(
            {"single_rep": single_rep, "pair_rep": pair_rep, "conditional_feats": conditional_feats, "mask": mask}
        )

        # Change input values at masked positions.
        single_rep = single_rep * mask.unsqueeze(-1) + (1 - mask.unsqueeze(-1)) * 100 * torch.randn_like(
            single_rep, device=device
        )
        conditional_feats = conditional_feats * mask.unsqueeze(-1) + (1 - mask.unsqueeze(-1)) * 100 * torch.randn_like(
            conditional_feats, device=device
        )
        pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        pair_rep = pair_rep * pair_mask.unsqueeze(-1) + (1 - pair_mask.unsqueeze(-1)) * 100 * torch.randn_like(
            pair_rep, device=device
        )

        output_altered = trunk(
            {"single_rep": single_rep, "pair_rep": pair_rep, "conditional_feats": conditional_feats, "mask": mask}
        )

        keys_to_check = list(output_base.keys())
        for key in keys_to_check:
            assert torch.allclose(output_base[key], output_altered[key]), (
                f"Trunk output differs based on masked input positions for {key}"
            )
