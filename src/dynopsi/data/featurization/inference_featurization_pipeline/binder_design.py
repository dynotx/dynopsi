import logging
import os
from collections.abc import Iterable, Iterator, Mapping
from typing import Literal, Self

from pydantic import Field, PrivateAttr, model_validator

from dynopsi.data.conditioning import DesignConditioning
from dynopsi.data.constants import AtomFilter, DesignDesignation, DesignTask, SimplifiedDSSPFeature
from dynopsi.data.featurization.base import Features
from dynopsi.data.featurization.inference_featurization_pipeline.inference_featurization_pipeline_base import (
    InferenceFeaturizationPipeline,
)
from dynopsi.data.featurization.inference_featurization_pipeline.registry import register_featurization_pipeline
from dynopsi.data.featurization.primitives import (
    CenterOfMassFeaturizer,
    CropFeaturizer,
    DataAugmentationConfig,
    HotspotFeaturizer,
    InpaintFeaturizer,
    NewChainsCentersOfMassFunctionType,
    NewChainsFeaturizer,
    PredefinedCenterOfMassEstimator,
    PriorMeansConfig,
    ReferenceStructureFeaturizer,
)
from dynopsi.data.featurization.registry import get_featurizer_type
from dynopsi.data.featurization.sample_centers import (
    PER_SAMPLE_FEATURES_METADATA_KEY,
    RANDOM_GLOBAL_TRANSLATION_STD_METADATA_KEY,
    SAMPLE_CENTERS_OF_MASS_METADATA_KEY,
    centers_match,
    draw_prior_com_offsets,
)
from dynopsi.data.utils import ChainResidueRepr
from dynopsi.utils import ConstantFunction

logger = logging.getLogger(__name__)


@register_featurization_pipeline()
class DynoPsiBinderDesignFeaturizationPipeline(InferenceFeaturizationPipeline):
    """Flexible featurization pipeline for binder design.

    Defines an inference pipeline configuration for multimer binder design tasks.

    Parameters
    ----------
    type: Literal["dynopsi_binder_design_featurization_pipeline"]
        Discriminative type of the featurization pipeline, unique to each subclass of InferenceFeaturizationPipeline. Do not modify.
    name : str
        Name of the featurization pipeline, used for naming output files. See `get_featurization_name`.
    rcsb_record_id: str | None, default None
        The RCSB PDB record ID for the reference structure, which will be downloaded from the PDB. Exactly one of rcsb_record_id, afdb_uniprot_id, or structure_path must be provided.
    afdb_uniprot_id: str | None, default None
        The UniProt ID for the reference structure, which will be downloaded from AFDB, v6. Exactly one of rcsb_record_id, afdb_uniprot_id, or structure_path must be provided.
    structure_path: str | None, default None
        The file path to the reference structure. The format of the file can be .pdb or .mmcif/.cif and will be automatically determined by its extension.
        Exactly one of rcsb_record_id, afdb_uniprot_id, or structure_path must be provided.
    index_type : Literal["auth_residue_index", "original_residue_index"]
        Type of chain and residue indexing used.
        "auth_residue_index" refers to the author-assigned indexing, which may start at any number,
        or contain gaps. Structures with insertion-coded residues are rejected.
        "original_residue_index" refers to standardized indexing, where chains are labeled A, B, C, ...
        and each chain's residues start from 1.
        It is strongly recommended that you check outputs using Features.visualize_configuration if you use auth_residue_index.
    crops: dict[str, ChainResidueRepr]
        One or more crops to apply to the input structure in a fanout manner. Format is {"arbitrary_crop_name": ChainResidueRepr}. For example: {"crop_A1_10": "A1-10"}.
        The crop must be compatible with index_type.
    hotspots: dict[str, ChainResidueRepr]
        One or more hotspots to apply to the input structure in a fanout manner. Format is {"arbitrary_hotspot_name": ChainResidueRepr}. For example: {"hotspot_A5_A8": "A5,A8"}.
        The hotspot must be compatible with index_type.
    new_chains_lengths: list[int]
        One or more lengths for the new chains to be generated, added in a fanout manner.
    new_chains_centers_of_mass: dict[str, NewChainsCentersOfMassFunctionType]
        One or more functions to compute the centers of mass for the new chains, added in a fanout manner. Format is {"arbitrary_chain_center_name": <NewChainsCentersOfMassFunctionType>}.
        For example: {"hitcenter": PredefinedCenterOfMassEstimator(centers_of_mass=[[25.0, 40.0, 30.0]])}
        See ``dynopsi.data.featurization.primitives.new_chains`` for possible center-of-mass estimators.
    sample_centers_of_mass: list[tuple[float, float, float]] | None
        Optional per-sample binder centers in the input structure frame. When set, each sample is
        featurized independently at its own center.
        When ``new_chains_centers_of_mass`` is empty, a single predefined estimator is derived from
        row 0. When both are set, ``sample_centers_of_mass[0]`` must match that predefined center.
        Length must be at least ``num_samples``. Requires a single crop/hotspot/length product.
    prior_com_translation_std_angstrom: float, default 0.0
        Inference-only prior-COM jitter in angstrom (inherited from ``InferenceFeaturizationPipeline``).
        When > 0, ``DesignPipeline.sample`` draws an independent ``N(0, std**2 I)`` offset for the
        predefined binder center ``mu`` and featurizes sample ``i`` independently at ``mu + delta_i``,
        changing the binder-vs-target sitting while keeping the design-chain COM at the origin after
        centering. Requires a single predefined estimator and a single crop/hotspot/length product (no Rg
        estimator). Mutually exclusive with ``sample_centers_of_mass`` and with
        ``random_global_translation_std_angstrom`` > 0.
    prior_com_translation_seed: int | None, default None
        Seed for ``prior_com_translation_std_angstrom`` (inherited). Required when that std is > 0 and
        must be unset otherwise.
    binder_atoms_to_include: AtomFilter
        Atom filter for designed chains.
        Must be either AtomFilter.BB or AtomFilter.CA.
    design_task : Literal["MULTIMER_BINDER_DESIGN", "MULTIMER_BINDER_DESIGN_NOISE_ALL"]
        Which inpaint-feature tokens are noised. ``MULTIMER_BINDER_DESIGN`` noises
        binder inpaint only. ``MULTIMER_BINDER_DESIGN_NOISE_ALL`` also noises
        target context.
    prior_centering_strategy : Literal["origin", "chain_and_noise_separated"]
        How prior means are centered. ``origin`` centers them at zero;
        ``chain_and_noise_separated`` centers each chain on its noised residues.
    design_conditioning: DesignConditioning
        Constant-write categorical conditioning; see :data:`DesignConditioning`.
    design_secondary_structure: dict[str, dict[SimplifiedDSSPFeature, ChainResidueRepr]] | None
        Each name is one design.
        Each value maps a SimplifiedDSSPFeature to residue spans on designed tokens.
        Spans in one design must not overlap.
        Omit this field when no design sets secondary structure.
    """

    type: Literal["dynopsi_binder_design_featurization_pipeline"] = "dynopsi_binder_design_featurization_pipeline"

    name: str
    design_task: Literal["MULTIMER_BINDER_DESIGN", "MULTIMER_BINDER_DESIGN_NOISE_ALL"] = "MULTIMER_BINDER_DESIGN"
    prior_centering_strategy: Literal["origin", "chain_and_noise_separated"] = "chain_and_noise_separated"
    index_type: Literal["auth_residue_index", "original_residue_index"]
    rcsb_record_id: str | None = None
    afdb_uniprot_id: str | None = None
    structure_path: str | None = None
    crops: dict[str, ChainResidueRepr] = Field(default_factory=dict)
    hotspots: dict[str, ChainResidueRepr]
    new_chains_lengths: list[int] = Field(default_factory=list)
    new_chains_centers_of_mass: dict[str, NewChainsCentersOfMassFunctionType] = Field(default_factory=dict)
    sample_centers_of_mass: list[tuple[float, float, float]] | None = None
    random_global_translation_std_angstrom: float = 0.0
    binder_atoms_to_include: AtomFilter
    design_conditioning: DesignConditioning = None
    design_secondary_structure: dict[str, dict[SimplifiedDSSPFeature, ChainResidueRepr]] | None = None

    ### Private fields ###
    _atoms_to_include: AtomFilter = PrivateAttr()
    # reference structure required
    _reference_structure_featurizer: list[ReferenceStructureFeaturizer] = PrivateAttr(default_factory=list)
    # crop optional
    _crop_featurizer: list[CropFeaturizer] = PrivateAttr(default_factory=list)
    # hotspot required
    _hotspot_featurizer: list[HotspotFeaturizer] = PrivateAttr(default_factory=list)
    # new chains required
    _new_chains_featurizer: list[NewChainsFeaturizer] = PrivateAttr(default_factory=list)
    # Resolved from design_task.
    _design_task: DesignTask = PrivateAttr()
    # inpaint should not be provided, it is automatically added to the pipeline
    _inpaint_featurizer: list[InpaintFeaturizer] = PrivateAttr(default_factory=list)
    # center of mass should not be provided, it is automatically added to the pipeline
    _center_of_mass_featurizer: list[CenterOfMassFeaturizer] = PrivateAttr(default_factory=list)

    def supplied_design_conditioning(self) -> Mapping[str, Mapping[str, object]] | None:
        return self.design_conditioning

    @model_validator(mode="after")
    def parse_to_inference_featurization_pipeline(self) -> Self:
        if self.random_global_translation_std_angstrom < 0:
            raise ValueError("random_global_translation_std_angstrom must be non-negative")
        # Non-negativity and the std<->seed pairing for prior_com_translation are validated on the
        # base class; only the binder-specific exclusivity constraints live here.
        if self.prior_com_translation_std_angstrom > 0:
            if self.sample_centers_of_mass is not None:
                raise ValueError(
                    "prior_com_translation_std_angstrom > 0 and sample_centers_of_mass are mutually exclusive"
                )
            if self.random_global_translation_std_angstrom > 0:
                raise ValueError(
                    "prior_com_translation_std_angstrom > 0 and "
                    "random_global_translation_std_angstrom > 0 are mutually exclusive"
                )
        if self.sample_centers_of_mass is not None and self.random_global_translation_std_angstrom > 0:
            raise ValueError(
                "sample_centers_of_mass and random_global_translation_std_angstrom > 0 are mutually exclusive"
            )
        if self.sample_centers_of_mass is not None:
            if len(self.sample_centers_of_mass) < 1:
                raise ValueError("sample_centers_of_mass must be non-empty")
            if not self.new_chains_centers_of_mass:
                first = (
                    float(self.sample_centers_of_mass[0][0]),
                    float(self.sample_centers_of_mass[0][1]),
                    float(self.sample_centers_of_mass[0][2]),
                )
                self.new_chains_centers_of_mass = {
                    "center": PredefinedCenterOfMassEstimator(centers_of_mass=(first,)),
                }

        self._atoms_to_include = self._validate_supported_atoms_to_include(
            atoms_to_include=self.binder_atoms_to_include,
            supported_atoms_to_include=(AtomFilter.BB, AtomFilter.CA),
            pipeline_name=type(self).__name__,
        )

        if self.rcsb_record_id is not None:
            ref_struct_name = self.rcsb_record_id
        elif self.afdb_uniprot_id is not None:
            ref_struct_name = self.afdb_uniprot_id
        elif self.structure_path is not None:
            ref_struct_name = os.path.basename(self.structure_path).split(".")[0]
        else:
            raise ValueError("Either rcsb_record_id, afdb_uniprot_id, or structure_path must be provided")

        self._reference_structure_featurizer = [
            ReferenceStructureFeaturizer(
                name=ref_struct_name,
                rcsb_record_id=self.rcsb_record_id,
                afdb_uniprot_id=self.afdb_uniprot_id,
                structure_path=self.structure_path,
                design_designation=DesignDesignation.FIXED,
                atoms_to_include=self._atoms_to_include,
                verbose_metadata=self.verbose_metadata,
            )
        ]

        self._crop_featurizer = [
            CropFeaturizer(
                name=crop_name,
                residue_index_chain_spec=crop_spec,
                index_type=self.index_type,
                verbose_metadata=self.verbose_metadata,
            )
            for crop_name, crop_spec in self.crops.items()
        ]

        self._hotspot_featurizer = [
            HotspotFeaturizer(
                name=hotspot_name,
                hotspot_positions=hotspot_spec,
                index_type=self.index_type,
                verbose_metadata=self.verbose_metadata,
            )
            for hotspot_name, hotspot_spec in self.hotspots.items()
        ]

        if len(self._hotspot_featurizer) == 0:
            raise ValueError("Hotspots are required for the binder design pipeline")

        self._new_chains_featurizer = [
            NewChainsFeaturizer(
                name=f"length_{length}_{center_of_mass_name}",
                lengths=(length,),
                estimate_centers_of_mass_function=center_of_mass,
                atoms_to_include=self._atoms_to_include,
                verbose_metadata=self.verbose_metadata,
            )
            for length in self.new_chains_lengths
            for center_of_mass_name, center_of_mass in self.new_chains_centers_of_mass.items()
        ]
        if len(self._new_chains_featurizer) == 0:
            raise ValueError("New chains are required for the binder design pipeline")

        self._design_task = DesignTask[self.design_task]

        self._inpaint_featurizer = [
            InpaintFeaturizer(
                inpainting_positions=(),
                inpainting_lengths=(),
                inpaint_region_centers_of_mass=(),
                atoms_to_include=self._atoms_to_include,
                verbose_metadata=self.verbose_metadata,
            )
        ]

        self._center_of_mass_featurizer = [
            CenterOfMassFeaturizer(
                data_augmentation_config=DataAugmentationConfig(
                    centering_strategy="design_chains", random_global_translation_std=0.0, random_global_rotation=False
                ),
                prior_means_config=PriorMeansConfig(
                    prior_centering_strategy=self.prior_centering_strategy,
                    # at inference time, keep the center of mass w/out perturbation
                    prior_mean_offset_function=ConstantFunction(constant=0.0),
                ),
                verbose_metadata=self.verbose_metadata,
            )
        ]

        self._append_custom_conditioning_featurizers(
            design_conditioning=self.design_conditioning,
            design_secondary_structure=self.design_secondary_structure,
            index_type=self.index_type,
        )

        if self.sample_centers_of_mass is not None or self.prior_com_translation_std_angstrom > 0:
            predefined_centers = self._require_predefined_centers()
            if self.sample_centers_of_mass is not None:
                # The explicit list carries one binder center per sample, so it targets a single
                # predefined center (K = 1) and row 0 must match it.
                if len(predefined_centers) != 1:
                    raise ValueError("sample_centers_of_mass requires a single-chain predefined center")
                sample_row_0 = tuple(float(x) for x in self.sample_centers_of_mass[0])
                if not centers_match(sample_row_0, predefined_centers[0]):
                    raise ValueError("sample_centers_of_mass[0] must match the predefined featurization center")

        self._build_featurization_pipeline()

        return self

    def _require_predefined_centers(self) -> tuple[tuple[float, float, float], ...]:
        """Return the predefined binder center ``mu``, rejecting Rg, fanout, or missing estimators.

        Shared by the ``sample_centers_of_mass`` list path and the ``prior_com_translation_std_angstrom``
        draw path. Binder design builds a single-length ``NewChainsFeaturizer`` (``lengths=(length,)``),
        and ``PredefinedCenterOfMassEstimator.check_num_chains_compatible`` requires one center per
        length, so a validated binder estimator carries exactly one center: this returns a single-element
        tuple ``(mu,)``. Named-estimator, crop, hotspot, and length fanout (multiple jobs) stay rejected.
        """
        estimators = list(self.new_chains_centers_of_mass.values())
        if len(estimators) != 1 or not isinstance(estimators[0], PredefinedCenterOfMassEstimator):
            raise ValueError(
                "sample_centers_of_mass and prior_com_translation_std_angstrom require exactly one "
                "predefined_center_of_mass estimator"
            )
        estimator = estimators[0]
        product_count = self._featurization_product_count()
        if product_count != 1:
            raise ValueError(
                "sample_centers_of_mass and prior_com_translation_std_angstrom require a single "
                f"featurization product (one crop/hotspot/length); got {product_count}"
            )
        return tuple((float(center[0]), float(center[1]), float(center[2])) for center in estimator.centers_of_mass)

    def get_featurization_name(self, features_dict: dict[str, Features]) -> str:
        """Generate a unique name for each feature set in the pipeline based on key featurization steps.

        Constructs a name from reference structure, crop, hotspot, and new chain
        steps, omitting steps that are constant across binder design featurizations.

        Parameters
        ----------
        features_dict : dict[str, Features]
            This pipeline only allows a dictionary with a single key 'features'.

        Returns
        -------
        str
            The name is in the format: `{step1_name}___{step2_name}...`

        Raises
        ------
        ValueError if a dictionary is provided that does not contain exactly one
        key called 'features'
        """
        if features_dict.keys() != {"features"}:
            raise ValueError(
                """"
                For DynoPsiBinderDesignFeaturizationPipeline, you must supply a dictionary that only contains the following key: {'features'}
                """
            )

        steps_to_include = [
            "reference_structure",
            "crop",
            "hotspot",
            "new_chains",
        ]
        if len(self._custom_conditioning_featurizers) > 0:
            steps_to_include.extend(
                [get_featurizer_type(featurizer_list[0]) for featurizer_list in self._custom_conditioning_featurizers]
            )
        return (
            "___".join(name for step, name in features_dict["features"].featurization_steps if step in steps_to_include)
            + f"___atoms_{self._atoms_to_include.value}"
        )

    def _featurize_scalar(self, features: Features) -> Iterator[dict[str, Features]]:
        if self._featurization_pipeline is None:
            raise ValueError("featurization pipeline is not built")
        for featurized in self._featurization_pipeline([features]):
            if self.random_global_translation_std_angstrom > 0:
                featurized.featurization_metadata[RANDOM_GLOBAL_TRANSLATION_STD_METADATA_KEY] = (
                    self.random_global_translation_std_angstrom
                )
            yield {"features": featurized}

    def _place_center_sets(
        self, features: Features, center_sets: list[list[tuple[float, float, float]]]
    ) -> tuple[list[Features], list[list[tuple[float, float, float]]]]:
        """Featurize one copy per sample at the binder center(s) of its ``center_set``.

        ``center_sets`` has one entry per sample; each entry carries that sample's predefined binder
        centers. Binder design is single-center, so each entry is one center: the per-sample
        ``sample_centers_of_mass`` row or the jittered ``mu + delta_i`` from
        ``prior_com_translation_std_angstrom``. The loop stays length-agnostic so the explicit-list
        and prior-COM-draw writers share one placement. Returns ``(per_sample, normalized centers)``;
        the caller owns any carrier write.
        """
        normalized = [
            [(float(com[0]), float(com[1]), float(com[2])) for com in center_set] for center_set in center_sets
        ]
        per_sample: list[Features] = []
        for center_set in normalized:
            estimator = PredefinedCenterOfMassEstimator(centers_of_mass=tuple(center_set))
            new_chain_copies = [
                featurizer.model_copy(update={"estimate_centers_of_mass_function": estimator})
                for featurizer in self._new_chains_featurizer
            ]
            local_pipeline = self._compose_featurization_pipeline(new_chain_copies)
            products = list(local_pipeline([features.safe_copy()]))
            per_sample.append(products[0])
        return per_sample, normalized

    def _featurize_per_sample(
        self, features: Features, center_sets: list[list[tuple[float, float, float]]]
    ) -> Iterator[dict[str, Features]]:
        """Featurize the explicit ``sample_centers_of_mass`` list and write the per-sample carrier.

        Shares :meth:`_place_center_sets` with the ``prior_com_translation_std_angstrom`` draw path.
        Used by ``__call__`` when ``sample_centers_of_mass`` is set.
        """
        per_sample, normalized = self._place_center_sets(features, center_sets)
        carrier = per_sample[0].safe_copy()
        carrier.featurization_metadata[SAMPLE_CENTERS_OF_MASS_METADATA_KEY] = normalized
        carrier.featurization_metadata[PER_SAMPLE_FEATURES_METADATA_KEY] = per_sample
        yield {"features": carrier}

    def __call__(self, inputs: Iterable[Features]) -> Iterator[dict[str, Features]]:
        """Run the binder design featurization pipeline.

        Parameters
        ----------
        inputs : Iterable[Features]
            Input Features objects to process.

        Yields
        ------
        dict[str, Features]
            Featurized outputs for binder design using the configured atom filter, keyed with 'features'
        """
        if self._featurization_pipeline is None:
            raise ValueError("featurization pipeline is not built")
        for features in inputs:
            features.atoms_to_include = self._atoms_to_include
            if self.sample_centers_of_mass is None:
                yield from self._featurize_scalar(features)
            else:
                yield from self._featurize_per_sample(features, [[com] for com in self.sample_centers_of_mass])

    def _place_prior_com_samples(
        self, features: Features, *, num_samples: int
    ) -> tuple[list[Features], list[list[tuple[float, float, float]]]]:
        """Place the per-sample prior-COM cloud around the predefined binder pin.

        Draws an independent offset for the predefined binder center ``mu`` and every sample, then
        featurizes sample ``i`` at ``mu + delta_i``. Binder design is single-center, so the
        ``(num_samples, 1)`` cloud is drawn once and each sample sees its own binder COM. ``__call__``
        (``dynopsi check``) uses the scalar pin at ``mu``.
        """
        assert self.prior_com_translation_seed is not None
        mus = self._require_predefined_centers()
        offsets = draw_prior_com_offsets(
            num_samples,
            len(mus),
            self.prior_com_translation_std_angstrom,
            seed=self.prior_com_translation_seed,
        )
        center_sets = [
            [
                (
                    mus[k][0] + float(offsets[i, k, 0]),
                    mus[k][1] + float(offsets[i, k, 1]),
                    mus[k][2] + float(offsets[i, k, 2]),
                )
                for k in range(len(mus))
            ]
            for i in range(num_samples)
        ]
        return self._place_center_sets(features, center_sets)
