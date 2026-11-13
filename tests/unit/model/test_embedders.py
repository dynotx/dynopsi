from typing import Any

import pytest
import torch

from dynopsi.data import constants
from dynopsi.model.modules.embedders import (
    ConditionalFeatureGenerator,
    DynoPsiPairRepEncoder,
    DynoPsiSingleConditionalPairEmbedder,
    DynoPsiSingleRepEncoder,
    _per_chain_centroid_shift,
)


@pytest.fixture
def test_params() -> dict:
    """Common test parameters used across test cases."""
    return {
        "batch_size": 2,
        "seq_len": 10,
        "num_atoms": 2,
        "num_time_embeddings": 2,
        "k": 10,
        "num_distance_bins": 12,
    }


class TestDynoPsiSingleRepEncoder:
    def test_self_cond_embedder(self, test_params: dict) -> None:
        embedder = DynoPsiSingleRepEncoder(
            embed_dim=32,
            use_self_cond=True,
            num_atoms=test_params["num_atoms"],
            index_embed_dim=None,
        )

        coords = torch.randn(test_params["batch_size"], test_params["seq_len"], test_params["num_atoms"], 3)
        indices = torch.arange(test_params["seq_len"]).repeat(test_params["batch_size"], 1)
        sc_coords = torch.randn(test_params["batch_size"], test_params["seq_len"], test_params["num_atoms"], 3)

        embedder(coords, indices, sc_coords)

    def test_self_cond_null_case(self, test_params: dict) -> None:
        embedder = DynoPsiSingleRepEncoder(
            embed_dim=32,
            use_self_cond=True,
            num_atoms=test_params["num_atoms"],
            index_embed_dim=None,
        )

        coords = torch.randn(test_params["batch_size"], test_params["seq_len"], test_params["num_atoms"], 3)
        indices = torch.arange(test_params["seq_len"]).repeat(test_params["batch_size"], 1)

        # Zero out the bias to simplify testing
        embedder.x_sc_embedder.bias.data.zero_()
        feats_null = embedder(coords, indices, None)
        feats_zero = embedder(coords, indices, torch.zeros_like(coords))
        assert torch.allclose(feats_null, feats_zero), "Self-cond feats should be zero when sc_coords is None"

    def test_inpaint_embedder(self, test_params: dict) -> None:
        embedder = DynoPsiSingleRepEncoder(
            embed_dim=32,
            use_self_cond=False,
            num_atoms=test_params["num_atoms"],
            index_embed_dim=None,
            use_inpaint_dgram=True,
        )
        coords = torch.randn(test_params["batch_size"], test_params["seq_len"], test_params["num_atoms"], 3)
        indices = torch.arange(test_params["seq_len"]).repeat(test_params["batch_size"], 1)
        inpaint_features = torch.randint(0, 4, (test_params["batch_size"], test_params["seq_len"]))
        embedder(coords, indices, inpaint_features=inpaint_features)

    def test_embedder_masking(self, test_params: dict) -> None:
        embedder = DynoPsiSingleRepEncoder(
            embed_dim=32,
            use_self_cond=False,
            num_atoms=1,
            index_embed_dim=None,
        )

        coords = torch.randn(test_params["batch_size"], test_params["seq_len"], 1, 3)
        indices = torch.arange(test_params["seq_len"]).repeat(test_params["batch_size"], 1)
        mask = torch.randint(0, 2, (test_params["batch_size"], test_params["seq_len"]))
        embedding_base = embedder(coords, indices, mask=mask)

        # Change input values at masked positions
        atom_mask = mask.unsqueeze(-1).unsqueeze(-1)
        coords = coords * atom_mask + (1 - atom_mask) * 100 * torch.randn_like(coords)
        indices = indices * mask + (1 - mask) * torch.randint(
            1, test_params["seq_len"] + 1, (test_params["batch_size"], test_params["seq_len"])
        )
        embedding_altered = embedder(coords, indices, mask=mask)

        assert torch.allclose(embedding_base, embedding_altered), (
            "Single rep encoding computation differs based on masked input positions"
        )


class TestCenterSingleRepPerChain:
    def _encoder(self, **kwargs: Any) -> DynoPsiSingleRepEncoder:
        return DynoPsiSingleRepEncoder(
            embed_dim=32,
            use_self_cond=True,
            num_atoms=2,
            index_embed_dim=None,
            **kwargs,
        )

    def test_flag_off_is_not_translation_invariant(self) -> None:
        embedder = self._encoder(center_single_rep_per_chain=False)
        coords = torch.randn(1, 6, 2, 3)
        chain_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])
        indices = torch.arange(6).unsqueeze(0)
        sc_coords = torch.randn_like(coords)
        baseline = embedder(coords, indices, sc_coords, chain_ids=chain_ids)
        translated = coords.clone()
        translated[:, 3:] = translated[:, 3:] + 10.0
        sc_translated = sc_coords.clone()
        sc_translated[:, 3:] = sc_translated[:, 3:] + 10.0
        moved = embedder(translated, indices, sc_translated, chain_ids=chain_ids)
        assert not torch.allclose(baseline, moved)
        assert torch.allclose(baseline, embedder(coords, indices, sc_coords))

    def test_flag_on_is_invariant_to_per_chain_translation(self) -> None:
        embedder = self._encoder(center_single_rep_per_chain=True)
        coords = torch.randn(2, 6, 2, 3)
        chain_ids = torch.tensor([[0, 0, 0, 1, 1, 1], [2, 2, 3, 3, 3, 3]])
        indices = torch.arange(6).unsqueeze(0).expand(2, -1)
        sc_coords = torch.randn_like(coords)
        baseline = embedder(coords, indices, sc_coords, chain_ids=chain_ids)
        translated = coords.clone()
        translated[0, 3:] = translated[0, 3:] + torch.tensor([4.0, -2.0, 9.0])
        translated[1, :2] = translated[1, :2] - 7.0
        sc_translated = sc_coords.clone()
        sc_translated[0, 3:] = sc_translated[0, 3:] + torch.tensor([4.0, -2.0, 9.0])
        sc_translated[1, :2] = sc_translated[1, :2] - 7.0
        moved = embedder(translated, indices, sc_translated, chain_ids=chain_ids)
        assert torch.allclose(baseline, moved, atol=1e-5)

    def test_flag_on_without_chain_ids_matches_flag_off(self) -> None:
        coords = torch.randn(1, 4, 2, 3)
        indices = torch.arange(4).unsqueeze(0)
        off = self._encoder(center_single_rep_per_chain=False)
        on = self._encoder(center_single_rep_per_chain=True)
        on.load_state_dict(off.state_dict())
        assert torch.allclose(off(coords, indices), on(coords, indices))

    def test_pad_residues_do_not_move_centroid(self) -> None:
        embedder = self._encoder(center_single_rep_per_chain=True)
        coords = torch.randn(1, 5, 2, 3)
        chain_ids = torch.tensor([[0, 0, 0, 1, 1]])
        mask = torch.tensor([[1, 1, 0, 1, 1]])
        indices = torch.arange(5).unsqueeze(0)
        clean = embedder(coords, indices, mask=mask, chain_ids=chain_ids)
        dirty = coords.clone()
        dirty[:, 2] = dirty[:, 2] + 100.0
        dirty_out = embedder(dirty, indices, mask=mask, chain_ids=chain_ids)
        valid = mask.bool()[0]
        assert torch.allclose(clean[0, valid], dirty_out[0, valid], atol=1e-5)

    def test_all_pad_chain_shift_is_zero(self) -> None:
        coords = torch.randn(1, 6, 2, 3)
        chain_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])
        mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
        shift = _per_chain_centroid_shift(coords, chain_ids, mask)
        assert torch.equal(shift[:, 3:], torch.zeros(1, 3, 3, dtype=coords.dtype))

        embedder = self._encoder(center_single_rep_per_chain=True)
        indices = torch.arange(6).unsqueeze(0)
        clean = embedder(coords, indices, mask=mask, chain_ids=chain_ids)
        dirty = coords.clone()
        dirty[:, 3:] = dirty[:, 3:] + 100.0
        dirty_out = embedder(dirty, indices, mask=mask, chain_ids=chain_ids)
        valid = mask.bool()[0]
        assert torch.allclose(clean[0, valid], dirty_out[0, valid], atol=1e-5)

    def _pair_embedder(self, **kwargs: Any) -> DynoPsiSingleConditionalPairEmbedder:
        return DynoPsiSingleConditionalPairEmbedder(
            embed_dim=32,
            pair_embed_dim=None,
            cond_embed_dim=None,
            index_embed_dim=None,
            num_time_embeddings=1,
            num_seq_sep_bins=127,
            num_distance_bins=64,
            num_distance_bins_sc=128,
            time_dim=64,
            num_atoms=2,
            use_self_cond=True,
            scalar_feats=None,
            per_residue_scalar_feats=None,
            categorical_feats=None,
            per_residue_categorical_feats=None,
            center_single_rep_per_chain=True,
            **kwargs,
        )

    def _two_chain_batch(self) -> dict[str, torch.Tensor]:
        coords = torch.randn(1, 6, 2, 3)
        return {
            "coords": coords,
            "sc_coords": torch.randn_like(coords),
            "token_index": torch.arange(6).unsqueeze(0),
            "residue_index": torch.arange(6).unsqueeze(0),
            "t": torch.rand(1),
            "asym_id": torch.tensor([[0, 0, 0, 1, 1, 1]]),
            "mask": torch.ones(1, 6, dtype=torch.long),
        }

    def test_pair_embedder_does_not_center_pair_rep(self) -> None:
        embedder = self._pair_embedder()
        batch = self._two_chain_batch()
        coords_before = batch["coords"].clone()
        sc_before = batch["sc_coords"].clone()
        baseline = embedder(batch)
        assert torch.equal(batch["coords"], coords_before)
        assert torch.equal(batch["sc_coords"], sc_before)

        moved = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
        shift = torch.tensor([4.0, -2.0, 9.0])
        moved["coords"][:, 3:] = moved["coords"][:, 3:] + shift
        moved["sc_coords"][:, 3:] = moved["sc_coords"][:, 3:] + shift
        out = embedder(moved)
        assert torch.allclose(baseline["single_rep"], out["single_rep"], atol=1e-5)
        assert not torch.allclose(baseline["pair_rep"], out["pair_rep"], atol=1e-4)


class TestConditionalFeatureGenerator:
    def test_conditional_feat_equalities(self, test_params: dict) -> None:
        conditional_feature_embedder = ConditionalFeatureGenerator(
            embed_dim=32,
            scalar_feats=["charge"],
            per_residue_scalar_feats=["flexibility"],
            categorical_feats={
                "is_backbone": 2,
            },
            per_residue_categorical_feats={
                "residue_type": 20,
            },
            num_time_embeddings=test_params["num_time_embeddings"],
        )

        assert conditional_feature_embedder.get_feature_names() == [
            "charge",
            "flexibility",
            "is_backbone",
            "residue_type",
        ]

        batch = {
            "residue_type": torch.randint(0, 20, (test_params["batch_size"], test_params["seq_len"])),
            "is_backbone": torch.randint(0, 2, (test_params["batch_size"],)),
            "charge": torch.rand(test_params["batch_size"], 1),
            "flexibility": torch.rand(test_params["batch_size"], test_params["seq_len"], 1),
        }
        embedding = conditional_feature_embedder(
            batch,
            t=torch.rand(test_params["batch_size"]),
            seq_len=test_params["seq_len"],
            t_ss=torch.rand(test_params["batch_size"]),
        )
        assert embedding.shape == (
            test_params["batch_size"],
            test_params["seq_len"],
            32 * (len(conditional_feature_embedder.embedders) + test_params["num_time_embeddings"]),
        )

        per_feat_tensors = torch.chunk(
            embedding,
            (len(conditional_feature_embedder.embedders) + test_params["num_time_embeddings"]),
            dim=-1,
        )
        ordering_dict = {
            "scalar": True,
            "per_residue_scalar": False,
            "categorical": True,
            "per_residue_categorical": False,
            "time": True,
        }
        for i, (name, ordering) in enumerate(ordering_dict.items()):
            all_same = (torch.var(per_feat_tensors[i], dim=(-2,)) == 0).all()
            assert all_same == ordering, (
                f"All sequence positions should {'' if ordering else 'not'} have the same values for {name} feature"
            )
        assert conditional_feature_embedder.get_output_dim() == embedding.shape[-1]

    def test_time_embedding(self, test_params: dict) -> None:
        conditional_feature_embedder = ConditionalFeatureGenerator(
            embed_dim=32,
            scalar_feats=None,
            per_residue_scalar_feats=None,
            categorical_feats=None,
            per_residue_categorical_feats=None,
            num_time_embeddings=1,
        )

        batch: dict[str, Any] = {}
        embedding = conditional_feature_embedder(
            batch, t=torch.rand(test_params["batch_size"]), seq_len=test_params["seq_len"]
        )
        assert embedding.shape == (test_params["batch_size"], test_params["seq_len"], 32)
        assert (torch.var(embedding, dim=(-2,)) == 0).all()

    def test_noised_mask(self, test_params: dict) -> None:
        conditional_feature_embedder = ConditionalFeatureGenerator(
            embed_dim=32,
            per_residue_categorical_feats={"noised_mask": 2},
            scalar_feats=None,
            per_residue_scalar_feats=None,
            categorical_feats=None,
            num_time_embeddings=1,
        )
        batch = {
            "noised_mask": torch.randint(0, 2, (test_params["batch_size"], test_params["seq_len"])),
        }
        embedding = conditional_feature_embedder(
            batch, t=torch.rand(test_params["batch_size"]), seq_len=test_params["seq_len"]
        )
        assert embedding.shape == (test_params["batch_size"], test_params["seq_len"], 64)
        assert "noised_mask" in conditional_feature_embedder.get_feature_names()

    @pytest.mark.parametrize(
        ("feature_name", "feature_enum"),
        [
            ("chain_is_loopy", constants.DSSPLoopFeature),
            ("has_homooligomer_partner", constants.HomooligomerPartnerFeature),
            ("secondary_structure_feature", constants.SimplifiedDSSPFeature),
        ],
    )
    def test_dssp_categorical_feature_vocab_matches_enum(
        self,
        test_params: dict,
        feature_name: str,
        feature_enum: (
            type[constants.DSSPLoopFeature]
            | type[constants.HomooligomerPartnerFeature]
            | type[constants.SimplifiedDSSPFeature]
        ),
    ) -> None:
        enum_values = [int(member) for member in feature_enum]
        # Embeddings require dense zero-based IDs, not merely len(enum) entries.
        assert sorted(enum_values) == list(range(len(feature_enum)))
        conditional_feature_embedder = ConditionalFeatureGenerator(
            embed_dim=32,
            scalar_feats=None,
            per_residue_scalar_feats=None,
            categorical_feats=None,
            per_residue_categorical_feats={feature_name: len(feature_enum)},
            num_time_embeddings=1,
        )
        embedding = conditional_feature_embedder(
            {
                feature_name: torch.tensor(
                    [enum_values] * test_params["batch_size"],
                    dtype=torch.long,
                )
            },
            t=torch.rand(test_params["batch_size"]),
            seq_len=len(enum_values),
        )
        assert conditional_feature_embedder.embedders[feature_name].num_embeddings == len(feature_enum)
        assert embedding.shape == (
            test_params["batch_size"],
            len(enum_values),
            64,  # categorical embedding + time embedding
        )
        with pytest.raises(IndexError):
            conditional_feature_embedder(
                {
                    feature_name: torch.full(
                        (1, 1),
                        len(feature_enum),  # out of range index
                        dtype=torch.long,
                    )
                },
                t=torch.rand(1),
                seq_len=1,
            )

    def test_embedder_masking(self, test_params: dict) -> None:
        conditional_feature_embedder = ConditionalFeatureGenerator(
            embed_dim=12,
            scalar_feats=["charge"],
            per_residue_scalar_feats=["flexibility"],
            categorical_feats={
                "is_backbone": 2,
            },
            per_residue_categorical_feats={
                "residue_type": 20,
            },
            num_time_embeddings=1,
        )

        batch = {
            "residue_type": torch.randint(0, 20, (test_params["batch_size"], test_params["seq_len"])),
            "is_backbone": torch.randint(0, 2, (test_params["batch_size"],)),
            "charge": torch.rand(test_params["batch_size"], 1),
            "flexibility": torch.rand(test_params["batch_size"], test_params["seq_len"], 1),
        }
        mask = torch.randint(0, 2, (test_params["batch_size"], test_params["seq_len"]))
        t = torch.rand(test_params["batch_size"])

        embedding_base = conditional_feature_embedder(batch, t=t, seq_len=test_params["seq_len"], mask=mask)

        # Change per-residue input values at masked positions
        residue_type = batch["residue_type"]
        residue_type = residue_type * mask + (1 - mask) * torch.randint(
            0, 20, (test_params["batch_size"], test_params["seq_len"])
        )
        flexibility = batch["flexibility"]
        flexibility = flexibility * mask.unsqueeze(-1) + (1 - mask.unsqueeze(-1)) * torch.randn_like(flexibility)
        batch.update({"residue_type": residue_type, "flexibility": flexibility})
        embedding_altered = conditional_feature_embedder(batch, t=t, seq_len=test_params["seq_len"], mask=mask)

        assert torch.allclose(embedding_base, embedding_altered), (
            "Conditional feature embeddings differ based on masked input positions"
        )


class TestDynoPsiPairRepEncoder:
    def test_self_cond_embedder(self, test_params: dict) -> None:
        embedder = DynoPsiPairRepEncoder(
            time_dim=12,
            num_atoms=test_params["num_atoms"],
            use_self_cond=True,
            num_seq_sep_bins=127,
            num_distance_bins=64,
            num_distance_bins_sc=128,
        )
        coords = torch.randn(test_params["batch_size"], test_params["seq_len"], test_params["num_atoms"], 3)
        indices = torch.arange(test_params["seq_len"]).repeat(test_params["batch_size"], 1)
        t = torch.rand(test_params["batch_size"])
        sc_coords = torch.randn(test_params["batch_size"], test_params["seq_len"], test_params["num_atoms"], 3)
        embedder(coords, indices, t, sc_coords)

    def test_self_cond_null_case(self, test_params: dict) -> None:
        embedder = DynoPsiPairRepEncoder(
            time_dim=12,
            num_atoms=test_params["num_atoms"],
            use_self_cond=True,
            num_seq_sep_bins=127,
            num_distance_bins=64,
            num_distance_bins_sc=128,
        )
        coords = torch.randn(test_params["batch_size"], test_params["seq_len"], test_params["num_atoms"], 3)
        indices = torch.arange(test_params["seq_len"]).repeat(test_params["batch_size"], 1)
        t = torch.rand(test_params["batch_size"])
        embedder(coords, indices, t, None)

    def test_inpaint_embedder(self, test_params: dict) -> None:
        embedder = DynoPsiPairRepEncoder(
            time_dim=12,
            num_atoms=test_params["num_atoms"],
            use_self_cond=False,
            num_seq_sep_bins=127,
            num_distance_bins=64,
            num_distance_bins_sc=128,
            use_inpaint_dgram=True,
        )
        true_coords = torch.randn(test_params["batch_size"], test_params["seq_len"], test_params["num_atoms"], 3)
        coords = torch.randn(test_params["batch_size"], test_params["seq_len"], test_params["num_atoms"], 3)
        indices = torch.arange(test_params["seq_len"]).repeat(test_params["batch_size"], 1)
        t = torch.rand(test_params["batch_size"])
        inpaint_features = torch.randint(0, 4, (test_params["batch_size"], test_params["seq_len"]))
        embedder(coords, indices, t, true_coords=true_coords, inpaint_features=inpaint_features)

    def test_pair_categorical_feature_embedder(self, test_params: dict) -> None:
        embedder = DynoPsiPairRepEncoder(
            time_dim=12,
            num_atoms=test_params["num_atoms"],
            use_self_cond=False,
            num_seq_sep_bins=127,
            num_distance_bins=64,
            num_distance_bins_sc=128,
            pair_noised_embed_dim=4,
            pair_inpaint_embed_dim=16,
        )
        true_coords = torch.randn(test_params["batch_size"], test_params["seq_len"], test_params["num_atoms"], 3)
        coords = torch.randn(test_params["batch_size"], test_params["seq_len"], test_params["num_atoms"], 3)
        indices = torch.arange(test_params["seq_len"]).repeat(test_params["batch_size"], 1)
        t = torch.rand(test_params["batch_size"])
        inpaint_features = torch.randint(0, 4, (test_params["batch_size"], test_params["seq_len"]))
        noised_mask = torch.randint(0, 2, (test_params["batch_size"], test_params["seq_len"]))
        embedder(
            coords, indices, t, true_coords=true_coords, inpaint_features=inpaint_features, noised_mask=noised_mask
        )

    def test_inpaint_masking(self, test_params: dict) -> None:
        embedder = DynoPsiPairRepEncoder(
            time_dim=12,
            num_atoms=test_params["num_atoms"],
            use_self_cond=False,
            num_seq_sep_bins=127,
            num_distance_bins=test_params["num_distance_bins"],
            num_distance_bins_sc=128,
            use_inpaint_dgram=True,
        )

        true_coords = torch.randn(test_params["batch_size"], test_params["seq_len"], test_params["num_atoms"], 3)
        inpaint_features = torch.randint(0, 4, (test_params["batch_size"], test_params["seq_len"]))

        result = embedder.add_inpaint(true_coords, inpaint_features)
        full_distogram = embedder.dgram_inpaint(true_coords)

        assert result.shape == (
            test_params["batch_size"],
            test_params["seq_len"],
            test_params["seq_len"],
            embedder.dgram_inpaint.get_output_dim(),
        )

        binder_context_mask = inpaint_features == constants.InpaintFeature.BINDER_CONTEXT
        target_context_mask = inpaint_features == constants.InpaintFeature.TARGET_CONTEXT

        binder_context_mask_2d = binder_context_mask.unsqueeze(-1) * binder_context_mask.unsqueeze(-2)
        target_context_mask_2d = target_context_mask.unsqueeze(-1) * target_context_mask.unsqueeze(-2)

        pair_context_mask = binder_context_mask_2d | target_context_mask_2d

        assert torch.all(result.sum(dim=-1)[~pair_context_mask] == 0), "Non-context positions have no distogram values"
        assert torch.all(result.sum(dim=-1)[pair_context_mask] > 0), "Context positions have distogram values"
        assert torch.allclose(result, full_distogram * pair_context_mask.unsqueeze(-1)), (
            "Result should be the same as the full distogram masked by the pair context mask"
        )

        binder_only_features = inpaint_features.clone()
        binder_only_features[~(binder_only_features == constants.InpaintFeature.BINDER_CONTEXT)] = (
            constants.InpaintFeature.BINDER_INPAINT
        )
        binder_only_features_mask = binder_only_features == constants.InpaintFeature.BINDER_CONTEXT
        binder_only_features_mask_2d = binder_only_features_mask.unsqueeze(-1) * binder_only_features_mask.unsqueeze(-2)
        binder_result = embedder.add_inpaint(true_coords, binder_only_features)
        assert torch.allclose(
            binder_result,
            full_distogram * binder_only_features_mask_2d.unsqueeze(-1),
        ), "All-binder-context case should match expected masking"

        no_context_features = torch.full(
            (test_params["batch_size"], test_params["seq_len"]),
            constants.InpaintFeature.BINDER_INPAINT,
        )
        no_context_result = embedder.add_inpaint(true_coords, no_context_features)
        assert torch.all(no_context_result == 0), "No-context features should result in all-zero distogram"

    def test_inpaint_masking_compiles(self, test_params: dict) -> None:
        embedder = DynoPsiPairRepEncoder(
            time_dim=12,
            num_atoms=test_params["num_atoms"],
            use_self_cond=False,
            num_seq_sep_bins=127,
            num_distance_bins=test_params["num_distance_bins"],
            num_distance_bins_sc=128,
            use_inpaint_dgram=True,
        )
        true_coords = torch.randn(test_params["batch_size"], test_params["seq_len"], test_params["num_atoms"], 3)
        inpaint_features = torch.randint(0, 4, (test_params["batch_size"], test_params["seq_len"]))

        expected = embedder.add_inpaint(true_coords, inpaint_features)
        compiled_add_inpaint = torch.compile(embedder.add_inpaint, backend="eager")

        torch.testing.assert_close(compiled_add_inpaint(true_coords, inpaint_features), expected)

    def test_embedder_masking(self, test_params: dict) -> None:
        embedder = DynoPsiPairRepEncoder(
            time_dim=12,
            num_atoms=test_params["num_atoms"],
            use_self_cond=True,
            num_seq_sep_bins=127,
            num_distance_bins=64,
            num_distance_bins_sc=128,
        )

        coords = torch.randn(test_params["batch_size"], test_params["seq_len"], test_params["num_atoms"], 3)
        indices = torch.arange(test_params["seq_len"]).repeat(test_params["batch_size"], 1)
        t = torch.rand(test_params["batch_size"])
        sc_coords = torch.randn(test_params["batch_size"], test_params["seq_len"], test_params["num_atoms"], 3)
        mask = torch.randint(0, 2, (test_params["batch_size"], test_params["seq_len"]))

        embedding_base = embedder(coords, indices, t, sc_coords, mask)

        # Change input values at masked positions
        atom_mask = mask.unsqueeze(-1).unsqueeze(-1)
        coords = coords * atom_mask + (1 - atom_mask) * 100 * torch.randn_like(coords)
        indices = indices * mask + (1 - mask) * torch.randint(
            1, test_params["seq_len"] + 1, (test_params["batch_size"], test_params["seq_len"])
        )
        sc_coords = sc_coords * atom_mask + (1 - atom_mask) * 100 * torch.randn_like(sc_coords)
        embedding_altered = embedder(coords, indices, t, sc_coords, mask)

        assert torch.allclose(embedding_base, embedding_altered), (
            "Pair rep encoding computation differs based on masked input positions"
        )

    def test_embedder_asym_id(self, test_params: dict) -> None:
        embedder = DynoPsiPairRepEncoder(
            time_dim=12,
            num_atoms=test_params["num_atoms"],
            use_self_cond=False,
            num_seq_sep_bins=127,
            num_distance_bins=64,
            num_distance_bins_sc=128,
        )
        coords = torch.randn(test_params["batch_size"], test_params["seq_len"], test_params["num_atoms"], 3)
        indices = torch.arange(test_params["seq_len"]).repeat(test_params["batch_size"], 1)
        chain_len = test_params["seq_len"] // 2
        chain_ids = torch.tensor([0] * chain_len + [1] * chain_len).repeat(test_params["batch_size"], 1)
        t = torch.rand(test_params["batch_size"])
        embedding = embedder(coords, indices, t)
        embedding_chain_ids = embedder(coords, indices, t, chain_ids=chain_ids)

        # The only part of the pair embedding that should change is the relative position encoding bit
        start_idx = embedder.dgram.get_output_dim()
        end_idx = start_idx + embedder.seq_idx_rel_pos_encoder.get_output_dim()
        rel_pos = embedding[:, :, :, start_idx:end_idx]
        rel_pos_chain_ids = embedding_chain_ids[:, :, :, start_idx:end_idx]

        assert not torch.allclose(rel_pos, rel_pos_chain_ids), (
            "Single and multi-chain rel pos embeddings should differ for inter-chain pairs"
        )

        assert torch.allclose(rel_pos[:, :chain_len, :chain_len], rel_pos_chain_ids[:, :chain_len, :chain_len]), (
            "intra-chain relative positions embeddings should be consistent"
        )
        assert torch.allclose(rel_pos[:, chain_len:, chain_len:], rel_pos_chain_ids[:, chain_len:, chain_len:]), (
            "Intra-chain relative positions embeddings should be consistent"
        )


class TestDynoPsiSingleConditionalPairEmbedder:
    def test_batch_independence(self, test_params: dict) -> None:
        """Test that batch computation matches individual forward passes."""
        embedder = DynoPsiSingleConditionalPairEmbedder(
            embed_dim=32,
            pair_embed_dim=None,
            cond_embed_dim=None,
            index_embed_dim=None,
            num_time_embeddings=1,
            num_seq_sep_bins=127,
            num_distance_bins=64,
            num_distance_bins_sc=128,
            time_dim=64,
            num_atoms=test_params["num_atoms"],
            use_self_cond=False,
            scalar_feats=["charge"],
            per_residue_scalar_feats=["flexibility"],
            categorical_feats={
                "is_backbone": 2,
            },
            per_residue_categorical_feats={
                "residue_type": 20,
            },
        )

        assert embedder.conditional_feature_generator.get_feature_names() == [
            "charge",
            "flexibility",
            "is_backbone",
            "residue_type",
        ]

        batch = {
            "coords": torch.randn(2, test_params["seq_len"], test_params["num_atoms"], 3),
            "token_index": torch.arange(test_params["seq_len"]).repeat(2, 1),
            "t": torch.rand(2),
            "charge": torch.rand(2, 1),
            "flexibility": torch.rand(2, test_params["seq_len"], 1),
            "residue_type": torch.randint(0, 20, (2, test_params["seq_len"])),
            "is_backbone": torch.randint(0, 2, (2,)),
            "residue_index": torch.randint(0, 20, (2, test_params["seq_len"])),
        }

        batch_embeddings = embedder(batch)
        batch1 = {k: v[0:1] for k, v in batch.items()}
        batch2 = {k: v[1:2] for k, v in batch.items()}

        emb1 = embedder(batch1)
        emb2 = embedder(batch2)

        for key in ["single_rep", "pair_rep", "conditional_feats"]:
            individual_concat = torch.cat([emb1[key], emb2[key]], dim=0)
            assert torch.allclose(batch_embeddings[key], individual_concat, atol=1e-5), (
                f"Batch computation differs from individual computations for {key}"
            )

    def test_forward_pass(self, test_params: dict) -> None:
        embedder_sc = DynoPsiSingleConditionalPairEmbedder(
            embed_dim=32,
            pair_embed_dim=None,
            cond_embed_dim=None,
            index_embed_dim=None,
            num_time_embeddings=1,
            num_seq_sep_bins=127,
            num_distance_bins=64,
            num_distance_bins_sc=128,
            time_dim=64,
            use_inpaint_dgram=True,
            pair_noised_embed_dim=4,
            pair_inpaint_embed_dim=16,
            num_atoms=test_params["num_atoms"],
            use_self_cond=True,
            scalar_feats=["charge"],
            per_residue_scalar_feats=["flexibility"],
            categorical_feats={
                "is_backbone": 2,
            },
            per_residue_categorical_feats={
                "residue_type": 20,
                "chain_breaks": 2,
            },
        )

        t = torch.rand(test_params["batch_size"])
        true_coords = coords = torch.randn(
            test_params["batch_size"], test_params["seq_len"], test_params["num_atoms"], 3
        )
        coords = torch.randn(test_params["batch_size"], test_params["seq_len"], test_params["num_atoms"], 3)
        indices = torch.arange(test_params["seq_len"]).repeat(test_params["batch_size"], 1)
        sc_coords = torch.randn(test_params["batch_size"], test_params["seq_len"], test_params["num_atoms"], 3)

        scalar_feats = torch.rand(test_params["batch_size"], 1)
        per_residue_scalar_feats = torch.rand(test_params["batch_size"], test_params["seq_len"], 1)
        categorical_feats = torch.randint(0, 2, (test_params["batch_size"],))
        per_residue_categorical_feats = torch.randint(0, 20, (test_params["batch_size"], test_params["seq_len"]))
        asym_id = torch.randint(0, 2, (test_params["batch_size"], test_params["seq_len"]))
        residue_index = torch.randint(0, 20, (test_params["batch_size"], test_params["seq_len"]))
        inpaint_features = torch.randint(0, 4, (test_params["batch_size"], test_params["seq_len"]))
        noised_mask = torch.randint(0, 2, (test_params["batch_size"], test_params["seq_len"]))

        batch = {
            "true_coords": true_coords,
            "coords": coords,
            "token_index": indices,
            "sc_coords": sc_coords,
            "t": t,
            "charge": scalar_feats,
            "flexibility": per_residue_scalar_feats,
            "residue_type": per_residue_categorical_feats,
            "is_backbone": categorical_feats,
            "asym_id": asym_id,
            "residue_index": residue_index,
            "inpaint_feature": inpaint_features,
            "noised_mask": noised_mask,
        }
        embedding_sc = embedder_sc.forward(batch)
        embedder_no_sc = DynoPsiSingleConditionalPairEmbedder(
            embed_dim=32,
            time_dim=64,
            num_atoms=test_params["num_atoms"],
            use_self_cond=False,
            scalar_feats=["charge"],
            per_residue_scalar_feats=["flexibility"],
            categorical_feats={
                "is_backbone": 2,
            },
            per_residue_categorical_feats={
                "residue_type": 20,
                "chain_breaks": 2,
            },
            num_seq_sep_bins=127,
            num_distance_bins=64,
            num_distance_bins_sc=128,
            use_inpaint_dgram=True,
            pair_noised_embed_dim=4,
            pair_inpaint_embed_dim=16,
            index_embed_dim=None,
            num_time_embeddings=1,
            pair_embed_dim=None,
            cond_embed_dim=None,
        )
        batch.pop("sc_coords")

        embedding_no_sc = embedder_no_sc.forward(batch)
        for key in ["single_rep", "pair_rep", "conditional_feats"]:
            shape_sc = embedding_sc[key].shape
            shape_no_sc = embedding_no_sc[key].shape
            assert embedding_sc[key].shape == embedding_no_sc[key].shape, (
                f"Batch computation differs from individual computations for {key}, sc: {shape_sc}, no sc: {shape_no_sc}"
            )

    def test_masking(self, test_params: dict) -> None:
        embedder = DynoPsiSingleConditionalPairEmbedder(
            embed_dim=32,
            time_dim=64,
            num_atoms=test_params["num_atoms"],
            use_self_cond=True,
            scalar_feats=["charge"],
            per_residue_scalar_feats=["flexibility"],
            categorical_feats={
                "is_backbone": 2,
            },
            per_residue_categorical_feats={
                "residue_type": 20,
            },
            num_seq_sep_bins=127,
            num_distance_bins=64,
            num_distance_bins_sc=128,
            use_inpaint_dgram=True,
            index_embed_dim=None,
            num_time_embeddings=1,
            pair_embed_dim=None,
            cond_embed_dim=None,
        )

        t = torch.rand(test_params["batch_size"])
        true_coords = torch.randn(test_params["batch_size"], test_params["seq_len"], test_params["num_atoms"], 3)
        coords = torch.randn(test_params["batch_size"], test_params["seq_len"], test_params["num_atoms"], 3)
        indices = torch.arange(test_params["seq_len"]).repeat(test_params["batch_size"], 1)
        sc_coords = torch.randn(test_params["batch_size"], test_params["seq_len"], test_params["num_atoms"], 3)
        mask = torch.randint(0, 2, (test_params["batch_size"], test_params["seq_len"]))

        scalar_feats = torch.rand(test_params["batch_size"], 1)
        per_residue_scalar_feats = torch.rand(test_params["batch_size"], test_params["seq_len"], 1)
        categorical_feats = torch.randint(0, 2, (test_params["batch_size"],))
        per_residue_categorical_feats = torch.randint(0, 20, (test_params["batch_size"], test_params["seq_len"]))
        residue_index = torch.randint(0, 20, (test_params["batch_size"], test_params["seq_len"]))
        inpaint_features = torch.randint(0, 4, (test_params["batch_size"], test_params["seq_len"]))
        noised_mask = torch.randint(0, 2, (test_params["batch_size"], test_params["seq_len"]))

        batch = {
            "true_coords": true_coords,
            "coords": coords,
            "token_index": indices,
            "sc_coords": sc_coords,
            "t": t,
            "mask": mask,
            "charge": scalar_feats,
            "flexibility": per_residue_scalar_feats,
            "residue_type": per_residue_categorical_feats,
            "is_backbone": categorical_feats,
            "residue_index": residue_index,
            "inpaint_feature": inpaint_features,
            "noised_mask": noised_mask,
        }

        embedding_base = embedder.forward(batch)

        # Change input values at masked positions.
        atom_mask = mask.unsqueeze(-1).unsqueeze(-1)
        residue_mask = mask.unsqueeze(-1)
        true_coords = true_coords * atom_mask + (1 - atom_mask) * 100 * torch.randn_like(true_coords)
        coords = coords * atom_mask + (1 - atom_mask) * 100 * torch.randn_like(coords)
        indices = indices * mask + (1 - mask) * torch.randint(
            1, 10, (test_params["batch_size"], test_params["seq_len"])
        )
        sc_coords = sc_coords * atom_mask + (1 - atom_mask) * 100 * torch.randn_like(sc_coords)
        inpaint_features = inpaint_features * mask + (1 - mask) * torch.randint(
            0, 4, (test_params["batch_size"], test_params["seq_len"])
        )
        noised_mask = noised_mask * mask + (1 - mask) * torch.randint(
            0, 2, (test_params["batch_size"], test_params["seq_len"])
        )
        per_residue_scalar_feats = per_residue_scalar_feats * residue_mask + (
            1 - residue_mask
        ) * 100 * torch.randn_like(per_residue_scalar_feats)
        per_residue_categorical_feats = per_residue_categorical_feats * mask + (1 - mask) * torch.randint(
            0, 20, (test_params["batch_size"], test_params["seq_len"])
        )
        batch.update(
            {
                "true_coords": true_coords,
                "coords": coords,
                "token_index": indices,
                "sc_coords": sc_coords,
                "flexibility": per_residue_scalar_feats,
                "residue_type": per_residue_categorical_feats,
                "inpaint_feature": inpaint_features,
                "noised_mask": noised_mask,
            }
        )

        embedding_altered = embedder.forward(batch)

        for key in ["single_rep", "pair_rep", "conditional_feats"]:
            assert torch.allclose(embedding_base[key], embedding_altered[key]), (
                f"Batch computation differs based on masked input positions for {key}"
            )

    def test_multiple_embed_dims(self, test_params: dict) -> None:
        embedder = DynoPsiSingleConditionalPairEmbedder(
            embed_dim=32,
            pair_embed_dim=16,
            cond_embed_dim=24,
            time_dim=64,
            num_atoms=test_params["num_atoms"],
            use_self_cond=True,
            scalar_feats=["charge"],
            per_residue_scalar_feats=["flexibility"],
            categorical_feats={
                "is_backbone": 2,
            },
            per_residue_categorical_feats={
                "residue_type": 20,
                "chain_breaks": 2,
            },
            num_seq_sep_bins=127,
            num_distance_bins=64,
            num_distance_bins_sc=128,
            use_inpaint_dgram=True,
            index_embed_dim=None,
            num_time_embeddings=1,
        )

        t = torch.rand(test_params["batch_size"])
        true_coords = coords = torch.randn(
            test_params["batch_size"], test_params["seq_len"], test_params["num_atoms"], 3
        )
        indices = torch.arange(test_params["seq_len"]).repeat(test_params["batch_size"], 1)
        sc_coords = torch.randn(test_params["batch_size"], test_params["seq_len"], test_params["num_atoms"], 3)
        scalar_feats = torch.rand(test_params["batch_size"], 1)
        per_residue_scalar_feats = torch.rand(test_params["batch_size"], test_params["seq_len"], 1)
        categorical_feats = torch.randint(0, 2, (test_params["batch_size"],))
        per_residue_categorical_feats = torch.randint(0, 20, (test_params["batch_size"], test_params["seq_len"]))
        asym_id = torch.randint(0, 3, (test_params["batch_size"], test_params["seq_len"]))
        residue_index = torch.randint(0, 20, (test_params["batch_size"], test_params["seq_len"]))
        inpaint_features = torch.randint(0, 4, (test_params["batch_size"], test_params["seq_len"]))
        noised_mask = torch.randint(0, 2, (test_params["batch_size"], test_params["seq_len"]))

        batch = {
            "true_coords": true_coords,
            "coords": coords,
            "token_index": indices,
            "sc_coords": sc_coords,
            "t": t,
            "charge": scalar_feats,
            "flexibility": per_residue_scalar_feats,
            "residue_type": per_residue_categorical_feats,
            "is_backbone": categorical_feats,
            "asym_id": asym_id,
            "residue_index": residue_index,
            "inpaint_feature": inpaint_features,
            "noised_mask": noised_mask,
        }
        embedding = embedder.forward(batch)

        assert embedding["single_rep"].shape == (test_params["batch_size"], test_params["seq_len"], 32)
        assert embedding["pair_rep"].shape == (
            test_params["batch_size"],
            test_params["seq_len"],
            test_params["seq_len"],
            16,
        )
        assert embedding["conditional_feats"].shape == (test_params["batch_size"], test_params["seq_len"], 24)
