"""Verify featurizations."""

import os
import pickle

import numpy as np

from dynopsi.data.constants import AtomFilter
from dynopsi.data.featurization import Features, primitives
from dynopsi.data.featurization.inference_featurization_pipeline import DynoPsiBinderDesignFeaturizationPipeline
from dynopsi.data.structure import Structure
from dynopsi.data.structure_io import save_structures_to_file

# Single length for testing purposes
LENGTHS = [60]

# Define featurization pipelines
featurization_pipelines = {
    # Verify some correct specifications for multiple targets
    "il7ra_3di3": {
        idx_type: DynoPsiBinderDesignFeaturizationPipeline(
            name="il7ra_3di3_binder_design",
            index_type=idx_type,
            rcsb_record_id="3di3",
            crops=crop_dict,
            hotspots=hotspot_dict,
            new_chains_lengths=LENGTHS,
            new_chains_centers_of_mass={
                "il7center": primitives.PredefinedCenterOfMassEstimator(centers_of_mass=[(28.1, 41.4, 47.9)]),
            },
        )
        for idx_type, crop_dict, hotspot_dict in [
            ("auth_residue_index", {"crop_B17_209": "B17-209"}, {"hotspot_VLY": "B58,B80,B139"}),
            ("original_residue_index", {"crop_B21_213": "B21-213"}, {"hotspot_VLY": "B62,B84,B143"}),
        ]
    },
    "pdl1_5o45": {
        idx_type: DynoPsiBinderDesignFeaturizationPipeline(
            name="pdl1_5o45_binder_design",
            index_type=idx_type,
            rcsb_record_id="5o45",
            crops=crop_dict,
            hotspots=hotspot_dict,
            new_chains_lengths=LENGTHS,
            new_chains_centers_of_mass={
                "hitcenter": primitives.PredefinedCenterOfMassEstimator(
                    centers_of_mass=[(22.265675, 11.73316, -24.249834)]
                ),
            },
        )
        for idx_type, crop_dict, hotspot_dict in [
            ("auth_residue_index", {"crop_A19_132": "A19-132"}, {"hotspot_IYQM": "A54,A56,A66,A115"}),
            ("original_residue_index", {"crop_A3_116": "A3-116"}, {"hotspot_IYQM": "A38,A40,A50,A99"}),
        ]
    },
    "insulinr_4zxb": {
        idx_type: DynoPsiBinderDesignFeaturizationPipeline(
            name="insulinr_4zxb_binder_design",
            index_type=idx_type,
            rcsb_record_id="4zxb",
            crops=crop_dict,
            hotspots=hotspot_dict,
            new_chains_lengths=LENGTHS,
            new_chains_centers_of_mass={
                "hitcenter": primitives.PredefinedCenterOfMassEstimator(centers_of_mass=[(24.241, 105.14, 48.663)]),
            },
        )
        for idx_type, crop_dict, hotspot_dict in [
            ("auth_residue_index", {"crop_E6_155": "E6-155"}, {"hotspot_FFF": "E64,E88,E96"}),
            ("original_residue_index", {"crop_E6_155": "E6-155"}, {"hotspot_FFF": "E64,E88,E96"}),
        ]
    },
    "sc2rbd_6m0j": {
        idx_type: DynoPsiBinderDesignFeaturizationPipeline(
            name="sc2rbd_6m0j_binder_design",
            index_type=idx_type,
            rcsb_record_id="6m0j",
            crops=crop_dict,
            hotspots=hotspot_dict,
            new_chains_lengths=LENGTHS,
            new_chains_centers_of_mass={
                "vibecenter": primitives.PredefinedCenterOfMassEstimator(centers_of_mass=[(-37.993, 28.582, -5.792)]),
            },
        )
        for idx_type, crop_dict, hotspot_dict in [
            ("auth_residue_index", {"crop_E333_526": "E333-526"}, {"hotspot_GYSTY": "E485,E489,E494,E500,E505"}),
            ("original_residue_index", {"crop_B15_208": "B15-208"}, {"hotspot_GYSTY": "B167,B171,B176,B182,B187"}),
        ]
    },
    "trka_1www": {
        idx_type: DynoPsiBinderDesignFeaturizationPipeline(
            name="trka_1www_binder_design",
            index_type=idx_type,
            rcsb_record_id="1www",
            crops=crop_dict,
            hotspots=hotspot_dict,
            new_chains_lengths=LENGTHS,
            new_chains_centers_of_mass={
                "vibecenter": primitives.PredefinedCenterOfMassEstimator(centers_of_mass=[[29.398, -9.294, -13.985]]),
            },
        )
        for idx_type, crop_dict, hotspot_dict in [
            ("auth_residue_index", {"crop_X282_382": "X282-382"}, {"hotspot_VML": "X294,X296,X333"}),
            ("original_residue_index", {"crop_C1_101": "C1-101"}, {"hotspot_VML": "C13,C15,C52"}),
        ]
    },
    "vegfa_1bj1": {
        idx_type: DynoPsiBinderDesignFeaturizationPipeline(
            name="vegfa_1bj1_binder_design",
            index_type=idx_type,
            rcsb_record_id="1bj1",
            crops=crop_dict,
            hotspots=hotspot_dict,
            new_chains_lengths=LENGTHS,
            new_chains_centers_of_mass={
                "testvibecenter": primitives.PredefinedCenterOfMassEstimator(
                    centers_of_mass=[[32.0603, -6.7852173, 29.20792]]
                ),
            },
        )
        for idx_type, crop_dict, hotspot_dict in [
            ("auth_residue_index", {"crop_V14_107_W14_107": "V14-107,W14-107"}, {"hotspot_MII": "W81,W83,W91"}),
            ("original_residue_index", {"crop_C7_100_D7_100": "C7-100,D7-100"}, {"hotspot_MII": "D74,D76,D84"}),
        ]
    },
    # Also test AFDB option
    "pdl1_afdb": {
        "auth_residue_index": DynoPsiBinderDesignFeaturizationPipeline(
            name="pdl1_Q9NZQ7_binder_design",
            index_type="auth_residue_index",
            afdb_uniprot_id="Q9NZQ7",
            crops={"crop_A19_132": "A19-132"},
            hotspots={"hotspot_IYQM": "A54,A56,A66,A115"},
            new_chains_lengths=LENGTHS,
            new_chains_centers_of_mass={
                "hitcenter": primitives.PredefinedCenterOfMassEstimator(
                    centers_of_mass=[(-40.110016, -2.3027444, 16.633)]
                ),
            },
        )
    },
    # Also test local structure path option
    "trka_local": {
        "auth_residue_index": DynoPsiBinderDesignFeaturizationPipeline(
            name="trka_local_cif",
            index_type="auth_residue_index",
            structure_path=f"{os.path.dirname(os.path.abspath(__file__))}/../test_files/1www_pdb_X_282_382.cif",
            crops={"crop_X282_382": "X282-382"},
            hotspots={"hotspot_VML": "X294,X296,X333"},
            new_chains_lengths=LENGTHS,
            new_chains_centers_of_mass={
                "vibecenter": primitives.PredefinedCenterOfMassEstimator(centers_of_mass=[[29.398, -9.294, -13.985]]),
            },
        )
    },
    # Also purposefully verify what happens when making a mistake in specification.
    # This mistake should be visible in the output structures produced for verification.
    "il7ra_mistaken_index_type": {
        "mistaken_index_type": DynoPsiBinderDesignFeaturizationPipeline(
            name="il7ra_mistaken_index_type",
            index_type="auth_residue_index",
            rcsb_record_id="3di3",
            # Use different crop and hotspot than the correct ones to avoid clobbering
            crops={"crop_B22_212": "B22-212"},
            hotspots={"hotspot_VY": "B62,B143"},
            new_chains_lengths=LENGTHS,
            new_chains_centers_of_mass={
                "il7center": primitives.PredefinedCenterOfMassEstimator(centers_of_mass=[(28.1, 41.4, 47.9)]),
            },
        )
    },
}


def test_invalid_featurization_pipelines():
    # Also purposefully verify mistakes that should raise errors.
    shared_kwargs = dict(
        name="vegfa_invalid",
        index_type="auth_residue_index",
        new_chains_lengths=LENGTHS,
        new_chains_centers_of_mass={
            "testvibecenter": primitives.PredefinedCenterOfMassEstimator(
                centers_of_mass=[[32.0603, -6.7852173, 29.20792]]
            ),
        },
    )

    n_catches = 0
    try:
        # Empty hotspot
        DynoPsiBinderDesignFeaturizationPipeline(
            **shared_kwargs, rcsb_record_id="1bj1", crops={"crop_V14_107_W14_107": "V14-107,W14-107"}, hotspots={}
        )
    except ValueError:
        n_catches += 1

    try:
        # Hotspot outside of cropped structure
        featurization_pipeline = DynoPsiBinderDesignFeaturizationPipeline(
            **shared_kwargs,
            rcsb_record_id="1bj1",
            crops={"crop_V14_107_W14_85": "V14-107,W14-85"},
            hotspots={"hotspot_MII": "W81,W83,W91"},
        )
        # Should error when trying to run featurization
        next(featurization_pipeline([Features(atoms_to_include=AtomFilter.BB)]))
    except ValueError:
        n_catches += 1

    try:
        # More than one structure input argument
        DynoPsiBinderDesignFeaturizationPipeline(
            **shared_kwargs,
            rcsb_record_id="1bj1",
            crops={"crop_V14_107_W14_107": "V14-107,W14-107"},
            hotspots={"hotspot_MII": "W81,W83,W91"},
            afdb_uniprot_id="P15692",
        )
    except ValueError:
        n_catches += 1

    try:
        # Missing all of rcsb_record_id, afdb_uniprot_id, structure_path
        DynoPsiBinderDesignFeaturizationPipeline(
            **shared_kwargs, crops={"crop_V14_107_W14_107": "V14-107,W14-107"}, hotspots={"hotspot_MII": "W81,W83,W91"}
        )
    except ValueError:
        n_catches += 1

    assert n_catches == 4


def test_structure_equality(struct1: Structure, struct2: Structure):
    attribs = ["chains", "features", "residues", "atoms", "interfaces", "residue_interfaces"]
    for attr in attribs:
        attr1 = getattr(struct1, attr)
        attr2 = getattr(struct2, attr)
        # check size
        assert attr1.shape == attr2.shape, f"Attribute {attr} has different shape: {attr1.shape} != {attr2.shape}"
        # check values
        for field_name in attr1.dtype.names:
            if field_name == "plddt":
                continue
            assert np.array_equal(attr1[field_name], attr2[field_name]), (
                f"Attribute {attr} field {field_name} has different values: {attr1[field_name]} != {attr2[field_name]}"
            )


def test_feature_dict_equality(feature_dict1: dict, feature_dict2: dict):
    for key in feature_dict1.keys():
        assert np.array_equal(feature_dict1[key].numpy(), feature_dict2[key].numpy()), (
            f"Feature dict key {key} has different values: {feature_dict1[key]} != {feature_dict2[key]}"
        )


if __name__ == "__main__":
    OUT_DIR = f"{os.getenv('HOME')}/temp/test_featurization_pipelines"
    REF_FEAT_DIR = os.path.join(OUT_DIR, "ref_feats")
    os.makedirs(REF_FEAT_DIR, exist_ok=True)
    BASE_STRUCTURE_DIR = os.path.join(OUT_DIR, "input_structures")
    os.makedirs(BASE_STRUCTURE_DIR, exist_ok=True)

    output_features = {}

    n_saved = 0
    for target, feat_pipelines in featurization_pipelines.items():
        print(f"Featurizing {target}")
        structs = []
        feature_dicts = []
        for idx_name, featurization_pipeline in feat_pipelines.items():
            print(f"Featurizing {idx_name}")
            for features in featurization_pipeline([Features(atoms_to_include=AtomFilter.BB)]):
                structs.append(features.input_reference_structure)
                feature_dicts.append(features.feature_dict)

        for i in range(1, len(structs)):
            test_structure_equality(structs[0], structs[i])
            test_feature_dict_equality(feature_dicts[0], feature_dicts[i])

        # just save the last of each idx type for visualization
        # this is whatever is currently "features"
        feat_name = featurization_pipeline.get_featurization_name(features)
        features.save_to_mmcif(f"{REF_FEAT_DIR}/{feat_name}.cif")
        features.save_to_pdb(f"{REF_FEAT_DIR}/{feat_name}.pdb")
        output_features[feat_name] = features.feature_dict
        if features.input_reference_structure is not None:
            outpath = os.path.join(BASE_STRUCTURE_DIR, f"{feat_name}_input_structure.npz")
            features.input_reference_structure.dump_to_npz(
                os.path.join(BASE_STRUCTURE_DIR, f"{feat_name}_input_structure.npz")
            )
            print("saving structure to pdb")
            save_structures_to_file(
                [features.input_reference_structure],
                os.path.join(BASE_STRUCTURE_DIR, f"{feat_name}_input_structure.pdb"),
            )
            save_structures_to_file(
                [features.input_reference_structure],
                os.path.join(BASE_STRUCTURE_DIR, f"{feat_name}_input_structure.cif"),
            )

        n_saved += 1

    assert len(output_features) == n_saved

    with open(os.path.join(OUT_DIR, "output_features.pkl"), "wb") as f:
        pickle.dump(output_features, f)

    test_invalid_featurization_pipelines()
