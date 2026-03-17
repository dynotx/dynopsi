import logging
import os
from collections.abc import Iterable, Iterator
from typing import Literal

from pydantic import Field, PrivateAttr, model_validator
from typing_extensions import Self

from dynopsi.data.constants import AtomFilter, DesignDesignation, DesignTask
from dynopsi.data.featurization.base import Features, Featurizer
from dynopsi.data.featurization.compose import ComposeFanoutFeaturizer, ComposeSequentialFeaturizer
from dynopsi.data.featurization.primitives import (
    CenterOfMassFeaturizer,
    CropFeaturizer,
    DataAugmentationConfig,
    HotspotFeaturizer,
    InpaintFeaturizer,
    NewChainsCentersOfMassFunctionType,
    NewChainsFeaturizer,
    NoiseMaskFeaturizer,
    PriorMeansConfig,
    ReferenceStructureFeaturizer,
)
from dynopsi.data.featurization.registry import register_featurizer
from dynopsi.data.utils import ChainResidueRepr
from dynopsi.utils import ConstantFunction

logger = logging.getLogger(__name__)


class InferenceFeaturizationPipeline(Featurizer):
    """Featurization pipeline configuration for inference.

    This class enforces a standardized order of multiple featurization steps (i.e. Featurizer configs)
    and prevents duplicates of the same Featurizer class in sequence. This ensures consistent processing across
    different inference runs.

    All attributes are private and set during validation, not at construction time. Subclasses must have a
    `type` discriminator field and should implement a Pydantic @model_validator(mode="after") to populate the
    private attributes from the subclass' own fields.
    See DynoPsiBinderDesignFeaturizationPipeline for an example.

    Each featurizer category is applied in a fixed sequence:
    1. Reference structure featurization
    2. Cropping
    3. Hotspot annotation
    4. Inpainting
    5. New chain generation
    6. Noise mask application (always included)
    7. Center of mass computation

    Parameters
    ----------
    verbose_metadata : bool, optional
        Whether to include verbose metadata, keyed by featurizer, in the output Features. See individual
        Featurizers for the specific definition of verbose_metadata for that Featurizer.
    _design_task : DesignTask
        The design task specification for this pipeline.
    _atoms_to_include : AtomFilter | None, optional
        Filter specifying which atoms to include in the output.
    _reference_structure_featurizer : list[ReferenceStructureFeaturizer], optional
        Featurizers for loading/processing reference structures.
    _crop_featurizer : list[CropFeaturizer], optional
        Featurizers for cropping structures.
    _hotspot_featurizer : list[HotspotFeaturizer], optional
        Featurizers for annotating hotspot residues.
    _inpaint_featurizer : list[InpaintFeaturizer], optional
        Featurizers for inpainting regions.
    _new_chains_featurizer : list[NewChainsFeaturizer], optional
        Featurizers for generating new chains.
    _center_of_mass_featurizer : list[CenterOfMassFeaturizer], optional
        Featurizers for computing center of mass.
    """

    verbose_metadata: bool = False

    _design_task: DesignTask = PrivateAttr()
    _atoms_to_include: AtomFilter = PrivateAttr()

    _reference_structure_featurizer: list[ReferenceStructureFeaturizer] = PrivateAttr(default_factory=list)
    _crop_featurizer: list[CropFeaturizer] = PrivateAttr(default_factory=list)
    _hotspot_featurizer: list[HotspotFeaturizer] = PrivateAttr(default_factory=list)
    _inpaint_featurizer: list[InpaintFeaturizer] = PrivateAttr(default_factory=list)
    _new_chains_featurizer: list[NewChainsFeaturizer] = PrivateAttr(default_factory=list)
    _center_of_mass_featurizer: list[CenterOfMassFeaturizer] = PrivateAttr(default_factory=list)

    _noise_mask_featurizer: NoiseMaskFeaturizer | None = PrivateAttr(default=None)
    _featurization_pipeline: Featurizer | None = PrivateAttr(default=None)

    @property
    def atoms_to_include(self) -> AtomFilter:
        return self._atoms_to_include

    def _build_featurization_pipeline(self) -> None:
        # create the noise mask featurizer config
        self._noise_mask_featurizer = NoiseMaskFeaturizer(design_task=self._design_task)

        # build the pipeline config
        pipeline_config_list: list[Featurizer] = []
        if len(self._reference_structure_featurizer) > 0:
            pipeline_config_list.append(ComposeFanoutFeaturizer(featurizers=self._reference_structure_featurizer))
        if len(self._crop_featurizer) > 0:
            pipeline_config_list.append(ComposeFanoutFeaturizer(featurizers=self._crop_featurizer))
        if len(self._hotspot_featurizer) > 0:
            pipeline_config_list.append(ComposeFanoutFeaturizer(featurizers=self._hotspot_featurizer))
        if len(self._inpaint_featurizer) > 0:
            pipeline_config_list.append(ComposeFanoutFeaturizer(featurizers=self._inpaint_featurizer))
        if len(self._new_chains_featurizer) > 0:
            pipeline_config_list.append(ComposeFanoutFeaturizer(featurizers=self._new_chains_featurizer))

        pipeline_config_list.append(self._noise_mask_featurizer)

        if len(self._center_of_mass_featurizer) > 0:
            pipeline_config_list.append(ComposeFanoutFeaturizer(featurizers=self._center_of_mass_featurizer))

        if len(pipeline_config_list) > 1:
            self._featurization_pipeline = ComposeSequentialFeaturizer(featurizers=pipeline_config_list)
        else:
            self._featurization_pipeline = pipeline_config_list[0]

    def get_featurization_name(self, features: Features) -> str:
        """Get a unique name for the featurization based on applied featurization steps.

        Subclasses must implement `get_featurization_name`.
        The name should not necessarily include all pipeline steps, as many steps will be
        shared for all featurizations, but it should reflect steps relevant for distinguishing
        different featurizations.

        Parameters
        ----------
        features : Features
            The featurized output to generate a name for.

        Returns
        -------
        str
            A unique identifier for this featurization configuration.
        """
        raise NotImplementedError("get_featurization_name should be implemented by the subclass")

    def visualize_configuration(
        self, output_dir: str, output_format: Literal["pdb", "mmcif", "npz"]
    ) -> list[dict[str, Features]]:
        """Run the featurization pipeline and save outputs for validation.

        Parameters
        ----------
        output_dir : str
            Directory to save output files. Created if it does not exist.
        output_format : Literal["pdb", "mmcif", "npz"]
            Output file format.

        Returns
        -------
        list[dict[str, Features]]
            The featurized outputs, keyed by the input features key.
            At present, there is only one input features key, "features".

        Raises
        ------
        ValueError
            If `output_format` is not a valid output format.
        """
        os.makedirs(output_dir, exist_ok=True)
        features_list: list[dict[str, Features]] = []
        input_features_key = "features"
        for features in self([Features()]):
            features_list.append({input_features_key: features})
            output_name = self.get_featurization_name(features)
            if output_format == "pdb":
                output_file = os.path.join(output_dir, f"{output_name}.pdb")
                logger.info(f"Saving features to {output_file}")
                features.save_to_pdb(output_file)
            elif output_format == "mmcif":
                output_file = os.path.join(output_dir, f"{output_name}.cif")
                logger.info(f"Saving features to {output_file}")
                features.save_to_mmcif(output_file)
            elif output_format == "npz":
                output_file = os.path.join(output_dir, f"{output_name}.npz")
                logger.info(f"Saving features to {output_file}")
                features.save_to_npz(output_file)
            else:
                raise ValueError(f"Invalid format: {output_format}")

        return features_list


@register_featurizer()
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
        contain gaps, or include insertion codes.
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
        For example: {"hitcenter": primitives.NewChainsCenterOfMassPredefined(centers_of_mass=((25, 40, 30)))}.
        See dynopsi.data.featurization.primitives.new_chains for possible center of mass functions.
    """

    type: Literal["dynopsi_binder_design_featurization_pipeline"] = "dynopsi_binder_design_featurization_pipeline"

    name: str
    index_type: Literal["auth_residue_index", "original_residue_index"]
    rcsb_record_id: str | None = None
    afdb_uniprot_id: str | None = None
    structure_path: str | None = None
    crops: dict[str, ChainResidueRepr] = Field(default_factory=dict)
    hotspots: dict[str, ChainResidueRepr]
    new_chains_lengths: list[int] = Field(default_factory=list)
    new_chains_centers_of_mass: dict[str, NewChainsCentersOfMassFunctionType] = Field(default_factory=dict)

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
    # default fields for the binder design pipeline
    # do not change these, they will be checked by the model validator
    _design_task: DesignTask = PrivateAttr(default=DesignTask.MULTIMER_BINDER_DESIGN)
    # inpaint should not be provided, it is automatically added to the pipeline
    _inpaint_featurizer: list[InpaintFeaturizer] = PrivateAttr(default_factory=list)
    # center of mass should not be provided, it is automatically added to the pipeline
    _center_of_mass_featurizer: list[CenterOfMassFeaturizer] = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def parse_to_inference_featurization_pipeline(self) -> Self:
        # update atoms to BB for dyno-psi inputs
        self._atoms_to_include = AtomFilter.BB

        # ref structure
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

        # crops
        self._crop_featurizer = [
            CropFeaturizer(
                name=crop_name,
                residue_index_chain_spec=crop_spec,
                index_type=self.index_type,
                verbose_metadata=self.verbose_metadata,
            )
            for crop_name, crop_spec in self.crops.items()
        ]

        # hotspots
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

        # new chains
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

        self._design_task = DesignTask.MULTIMER_BINDER_DESIGN

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
                    prior_centering_strategy="chain_and_noise_separated",
                    # at inference time, keep the center of mass w/out perturbation
                    prior_mean_offset_function=ConstantFunction(constant=0.0),
                ),
                verbose_metadata=self.verbose_metadata,
            )
        ]

        self._build_featurization_pipeline()

        return self

    def get_featurization_name(self, features: Features) -> str:
        """Generate a unique name for the pipeline based on key featurization steps.

        Constructs a name from reference structure, crop, hotspot, and new chain
        steps, omitting steps that are constant across binder design featurizations.

        Parameters
        ----------
        features : Features
            The featurized output to generate a name for.

        Returns
        -------
        str
            Name in format: `{step1_name}___{step2_name}...`
        """
        steps_to_include = [
            "reference_structure",
            "crop",
            "hotspot",
            "new_chains",
        ]
        parts = [name for step, name in features.featurization_steps if step in steps_to_include]
        return "___".join(parts)

    def __call__(self, inputs: Iterable[Features]) -> Iterator[Features]:
        """Run the binder design featurization pipeline.

        Parameters
        ----------
        inputs : Iterable[Features]
            Input Features objects to process.

        Yields
        ------
        Features
            Featurized backbone-only outputs for binder design.
        """
        if self._featurization_pipeline is None:
            raise ValueError("featurization pipeline is not built")
        else:
            # modify features in place with expected atoms to include
            for features in inputs:
                features.atoms_to_include = self._atoms_to_include
            yield from self._featurization_pipeline(inputs)
