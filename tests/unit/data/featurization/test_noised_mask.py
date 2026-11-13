import pytest
import torch

from dynopsi.data.constants import AtomFilter, DesignTask, InpaintFeature
from dynopsi.data.featurization import Features, primitives
from dynopsi.data.structure import Structure
from dynopsi.data.utils import ChainResidueRepr
from tests.unit.data.conftest import assert_metadata_equality, assert_structure_equality


class TestNoiseMaskFeaturizer:
    inpaint_feature = {
        DesignTask.MONOMER_NOISE_BINDER_INPAINT: torch.tensor(
            [InpaintFeature.BINDER_INPAINT] * 2 + [InpaintFeature.BINDER_CONTEXT] * 4
        ),
        DesignTask.MONOMER_NOISE_TARGET_CHAIN_NO_IPT: torch.tensor([InpaintFeature.TARGET_CONTEXT] * 6),
        DesignTask.MONOMER_NOISE_BINDER_CHAIN_NO_IPT: torch.tensor([InpaintFeature.BINDER_CONTEXT] * 6),
        DesignTask.MULTIMER_BINDER_DESIGN: torch.tensor(
            [InpaintFeature.BINDER_INPAINT] * 2 + [InpaintFeature.TARGET_CONTEXT] * 4
        ),
        DesignTask.MULTIMER_TARGET_DOCKING: torch.tensor(
            [InpaintFeature.BINDER_CONTEXT] * 2 + [InpaintFeature.TARGET_CONTEXT] * 4
        ),
        DesignTask.MULTIMER_DOCKING_NOISE_ALL: torch.tensor(
            [InpaintFeature.BINDER_CONTEXT] * 2 + [InpaintFeature.TARGET_CONTEXT] * 4
        ),
        DesignTask.MULTIMER_BINDER_DESIGN_NOISE_ALL: torch.tensor(
            [InpaintFeature.BINDER_INPAINT] * 2 + [InpaintFeature.TARGET_CONTEXT] * 4
        ),
        DesignTask.MULTIMER_TARGET_DOCKING_INPAINT_BINDER: torch.tensor(
            [InpaintFeature.BINDER_INPAINT] * 2
            + [InpaintFeature.BINDER_CONTEXT] * 2
            + [InpaintFeature.TARGET_CONTEXT] * 2
        ),
    }

    expected_noise_mask = {
        DesignTask.MONOMER_NOISE_BINDER_INPAINT: torch.tensor(
            [True, True, False, False, False, False]
        ),  # 2xnoise binder, 4xbinder context
        DesignTask.MONOMER_NOISE_TARGET_CHAIN_NO_IPT: torch.tensor(
            [True, True, True, True, True, True]
        ),  # 6xtarget context (noised)
        DesignTask.MONOMER_NOISE_BINDER_CHAIN_NO_IPT: torch.tensor(
            [True, True, True, True, True, True]
        ),  # 6xbinder context (noised)
        DesignTask.MULTIMER_BINDER_DESIGN: torch.tensor(
            [True, True, False, False, False, False]
        ),  # 2xnoise binder, 4xtarget context
        DesignTask.MULTIMER_TARGET_DOCKING: torch.tensor(
            [False, False, True, True, True, True]
        ),  # 2xbinder context (unnoised), 4xtarget context (noised)
        DesignTask.MULTIMER_DOCKING_NOISE_ALL: torch.tensor(
            [True, True, True, True, True, True]
        ),  # 2xbinder context (noised), 4xtarget context (noised)
        DesignTask.MULTIMER_BINDER_DESIGN_NOISE_ALL: torch.tensor(
            [True, True, True, True, True, True]
        ),  # 2xbinder inpaint (noised), 4xtarget context (noised)
        DesignTask.MULTIMER_TARGET_DOCKING_INPAINT_BINDER: torch.tensor(
            [True, True, False, False, True, True]
        ),  # 2xnoise binder, 2xbinder context, 2xnoise target
    }

    @pytest.mark.parametrize(
        ["design_task", "verbose_metadata"], [(d, False) for d in DesignTask] + [(d, True) for d in DesignTask]
    )
    def test_noise_mask_featurizer(
        self,
        verbose_metadata: bool,
        design_task: DesignTask,
        feature_dict_standard: dict[str, torch.Tensor],
        structure_standard_no_interfaces: Structure,
        structure_chain_B_C_only: Structure,
    ) -> None:
        input_feature_dict = {
            k: v.clone() for k, v in feature_dict_standard.items() if k not in ["noised_mask", "inpaint_feature"]
        }
        input_feature_dict["inpaint_feature"] = self.inpaint_feature[design_task]

        input_features = Features(
            feature_dict=input_feature_dict,
            atoms_to_include=AtomFilter.BB,
            input_reference_structure=structure_standard_no_interfaces,
        )

        featurizer = primitives.NoiseMaskFeaturizer(
            name="test_noise_mask_featurizer",
            design_task=design_task,
            verbose_metadata=verbose_metadata,
        )
        out_feats = next(featurizer([input_features]))
        assert torch.allclose(out_feats.feature_dict["noised_mask"], self.expected_noise_mask[design_task])

        assert out_feats.featurization_steps == [("noise_mask", "test_noise_mask_featurizer")]

        expected_metadata = {
            "test_noise_mask_featurizer": {
                "name": "test_noise_mask_featurizer",
                "type": "noise_mask",
                "design_task": design_task,
                "verbose_metadata": verbose_metadata,
            }
        }
        if verbose_metadata:
            expected_metadata["test_noise_mask_featurizer"]["structure"] = out_feats.input_reference_structure

        assert_metadata_equality(observed=out_feats.featurization_metadata, expected=expected_metadata)

        context_positions = torch.isin(
            input_feature_dict["inpaint_feature"],
            torch.tensor([InpaintFeature.BINDER_CONTEXT, InpaintFeature.TARGET_CONTEXT]),
        )

        non_noised_repr = ChainResidueRepr.from_asym_id_mask(
            context_positions & input_feature_dict["token_pad_mask"].bool(),
            asym_id=input_feature_dict["asym_id"],
            residue_index=input_feature_dict["residue_index"],
            index_type="residue_index",
        )
        expected_structure = structure_standard_no_interfaces.crop(
            residue_index_chain_spec=non_noised_repr, index_type="residue_index", reset_residue_index=False
        )
        assert_structure_equality(observed=out_feats.input_reference_structure, expected=expected_structure)

    def test_noise_mask_featurizer_no_inpaint_features(self, feature_dict_standard: dict[str, torch.Tensor]) -> None:
        """Try to add noise mask with no inpaint features present in the input"""
        with pytest.raises(KeyError):
            featurizer = primitives.NoiseMaskFeaturizer(
                name="test_noise_mask_featurizer",
                design_task=DesignTask.MULTIMER_BINDER_DESIGN,
            )
            features_no_inpaint = {k: v.clone() for k, v in feature_dict_standard.items() if k != "inpaint_feature"}
            next(featurizer([Features(feature_dict=features_no_inpaint, atoms_to_include=AtomFilter.BB)]))
