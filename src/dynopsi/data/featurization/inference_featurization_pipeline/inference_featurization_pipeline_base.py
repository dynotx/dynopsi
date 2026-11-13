import logging
import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Literal, Self

from pydantic import PrivateAttr, model_validator

from dynopsi.data import conditioning
from dynopsi.data.conditioning import DesignConditioning
from dynopsi.data.constants import AtomFilter, DesignTask, SimplifiedDSSPFeature
from dynopsi.data.featurization.base import Features, Featurizer
from dynopsi.data.featurization.compose import ComposeFanoutFeaturizer, ComposeSequentialFeaturizer
from dynopsi.data.featurization.primitives import (
    CenterOfMassFeaturizer,
    CropFeaturizer,
    HotspotFeaturizer,
    InpaintFeaturizer,
    NewChainsFeaturizer,
    NoiseMaskFeaturizer,
    ReferenceStructureFeaturizer,
    SecondaryStructureConditioningFeaturizer,
    make_constant_write,
)
from dynopsi.data.featurization.registry import get_featurizer_type
from dynopsi.data.featurization.sample_centers import (
    PER_SAMPLE_FEATURES_METADATA_KEY,
    SAMPLE_CENTERS_OF_MASS_METADATA_KEY,
)
from dynopsi.data.utils.sequence_utils import ChainResidueRepr
from dynopsi.utils.strict_config import StrictConfig

logger = logging.getLogger(__name__)


class InferenceFeaturizationPipeline(StrictConfig):
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
    prior_com_translation_std_angstrom : float, default 0.0
        Inference-only prior-COM jitter in angstrom, shared by every pipeline that supports a
        draw-then-featurize sampling path. When > 0, :meth:`featurize_for_sampling` draws an
        independent ``N(0, std**2 I)`` offset for each prior COM ``k`` on the featurization product
        and featurizes sample ``i`` at ``mu_k + delta_{i,k}``, varying the design-vs-target sitting
        while the joint design COM stays at the origin after centering. ``K`` is the number of prior
        COMs (the predefined binder centers, or the noised chains under ``chain_and_noise_separated``
        for inpaint). Must be
        non-negative. This is not ``prior_mean_offset_function``, which offsets ``prior_mean`` only
        and leaves the sitting unchanged. Each concrete pipeline documents its own constraints.
    prior_com_translation_seed : int | None, default None
        Seed for ``prior_com_translation_std_angstrom``. Required when that std is > 0 and must be
        unset otherwise. The full ``num_samples`` x ``K`` cloud is drawn once at featurize time on a
        dedicated CPU generator, so the same seed, ``num_samples``, and ``K`` reproduce the same
        offsets. It is not derived from ``prior_seed`` or a batch slice.
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
    _custom_conditioning_featurizers: list[list[Featurizer]], optional
        Featurizers for applying custom conditioning features. The outer level
        lists will be composed in a fanout manner, and each sub-list contains
        instances of Featurizers of the same type, but with different parameters.
        For example, if you want to apply two different design_is_loopy features,
        you can do:
        [
            [
            ChainIsLoopyConditioningFeaturizer(design_is_loopy=design_is_loopy1),
            ChainIsLoopyConditioningFeaturizer(design_is_loopy=design_is_loopy2)
            ]
        ]
    """

    verbose_metadata: bool = False
    prior_com_translation_std_angstrom: float = 0.0
    prior_com_translation_seed: int | None = None

    _design_task: DesignTask = PrivateAttr()
    _atoms_to_include: AtomFilter = PrivateAttr()

    _reference_structure_featurizer: list[ReferenceStructureFeaturizer] = PrivateAttr(default_factory=list)
    _crop_featurizer: list[CropFeaturizer] = PrivateAttr(default_factory=list)
    _hotspot_featurizer: list[HotspotFeaturizer] = PrivateAttr(default_factory=list)
    _inpaint_featurizer: list[InpaintFeaturizer] = PrivateAttr(default_factory=list)
    _new_chains_featurizer: list[NewChainsFeaturizer] = PrivateAttr(default_factory=list)
    _center_of_mass_featurizer: list[CenterOfMassFeaturizer] = PrivateAttr(default_factory=list)
    _custom_conditioning_featurizers: list[list[Featurizer]] = PrivateAttr(default_factory=list)

    _noise_mask_featurizer: NoiseMaskFeaturizer | None = PrivateAttr(default=None)
    _featurization_pipeline: Featurizer | None = PrivateAttr(default=None)

    @property
    def atoms_to_include(self) -> AtomFilter:
        return self._atoms_to_include

    def supplied_design_conditioning(self) -> Mapping[str, Mapping[str, object]] | None:
        """Variant maps this pipeline supplies. The base returns None."""
        return None

    @model_validator(mode="after")
    def _validate_prior_com_translation_scalars(self) -> Self:
        """Validate the shared prior-COM jitter scalars (non-negative std, std<->seed pairing).

        Reads only the two public scalar fields, so it is order-independent with respect to the
        subclass validator that populates the private featurizer attributes.
        """
        if self.prior_com_translation_std_angstrom < 0:
            raise ValueError("prior_com_translation_std_angstrom must be non-negative")
        if self.prior_com_translation_std_angstrom > 0:
            if self.prior_com_translation_seed is None:
                raise ValueError("prior_com_translation_seed is required when prior_com_translation_std_angstrom > 0")
        elif self.prior_com_translation_seed is not None:
            raise ValueError("prior_com_translation_seed requires prior_com_translation_std_angstrom > 0")
        return self

    def featurize_for_sampling(self, inputs: Iterable[Features], *, num_samples: int) -> Iterator[dict[str, Features]]:
        """Featurize for a sampling run that will produce ``num_samples`` designs.

        When ``prior_com_translation_std_angstrom`` is <= 0, yield ``self(inputs)`` and ignore
        ``num_samples``. When it is > 0, this method writes the sampling carrier: it requires a built
        pipeline, ``num_samples`` >= 1, and ``prior_com_translation_seed``, calls
        :meth:`_place_prior_com_samples`, and stores ``sample_centers_of_mass`` and
        ``per_sample_center_features`` on a copy of the first sample.
        """
        if self.prior_com_translation_std_angstrom <= 0:
            yield from self(inputs)
            return
        if self._featurization_pipeline is None:
            raise ValueError("featurization pipeline is not built")
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1 to draw prior COM offsets")
        assert self.prior_com_translation_seed is not None
        for features in inputs:
            features.atoms_to_include = self._atoms_to_include
            per_sample, sample_centers = self._place_prior_com_samples(features, num_samples=num_samples)
            carrier = per_sample[0].safe_copy()
            carrier.featurization_metadata[SAMPLE_CENTERS_OF_MASS_METADATA_KEY] = sample_centers
            carrier.featurization_metadata[PER_SAMPLE_FEATURES_METADATA_KEY] = per_sample
            yield {"features": carrier}

    def _place_prior_com_samples(
        self, features: Features, *, num_samples: int
    ) -> tuple[list[Features], list[list[tuple[float, float, float]]]]:
        """Draw the per-sample prior-COM cloud and featurize one copy per sample at its placement.

        Returns ``(per_sample, sample_centers)``: the ``num_samples`` independently featurized
        ``Features`` and the per-sample list of prior COMs placed. A concrete pipeline implements how
        it re-places its prior COMs (the predefined binder pins, or the per-noised-chain inpaint
        segment COMs). The default raises because only pipelines with a prior-COM placement define one.
        """
        raise NotImplementedError("_place_prior_com_samples should be implemented by a prior-COM-capable subclass")

    def _featurization_product_count(self) -> int:
        """Return the size of the crop/hotspot/inpaint/new-chains/conditioning fanout product.

        A count > 1 means the pipeline fans out into multiple independent jobs; the prior-COM draw
        and the explicit ``sample_centers_of_mass`` list both require a single product.
        """
        product_count = 1
        for featurizer_list in (
            self._crop_featurizer,
            self._hotspot_featurizer,
            self._inpaint_featurizer,
            self._new_chains_featurizer,
        ):
            if len(featurizer_list) > 1:
                product_count *= len(featurizer_list)
        for custom_list in self._custom_conditioning_featurizers:
            if len(custom_list) > 1:
                product_count *= len(custom_list)
        return product_count

    @staticmethod
    def _validate_supported_atoms_to_include(
        atoms_to_include: AtomFilter, supported_atoms_to_include: tuple[AtomFilter, ...], pipeline_name: str
    ) -> AtomFilter:
        if atoms_to_include not in supported_atoms_to_include:
            supported_values = ", ".join(atom_filter.value for atom_filter in supported_atoms_to_include)
            raise ValueError(
                f"{pipeline_name} supports only atoms_to_include in ({supported_values}), got {atoms_to_include.value!r}."
            )
        return atoms_to_include

    def _compose_featurization_pipeline(
        self,
        new_chains_featurizer: list[NewChainsFeaturizer] | None = None,
        inpaint_featurizer: list[InpaintFeaturizer] | None = None,
    ) -> Featurizer:
        if new_chains_featurizer is None:
            new_chains_featurizer = self._new_chains_featurizer
        if inpaint_featurizer is None:
            inpaint_featurizer = self._inpaint_featurizer

        pipeline_config_list: list[Featurizer] = []
        if len(self._reference_structure_featurizer) > 0:
            pipeline_config_list.append(ComposeFanoutFeaturizer(featurizers=self._reference_structure_featurizer))
        if len(self._crop_featurizer) > 0:
            pipeline_config_list.append(ComposeFanoutFeaturizer(featurizers=self._crop_featurizer))
        if len(self._hotspot_featurizer) > 0:
            pipeline_config_list.append(ComposeFanoutFeaturizer(featurizers=self._hotspot_featurizer))
        if len(inpaint_featurizer) > 0:
            pipeline_config_list.append(ComposeFanoutFeaturizer(featurizers=inpaint_featurizer))
        if len(new_chains_featurizer) > 0:
            pipeline_config_list.append(ComposeFanoutFeaturizer(featurizers=new_chains_featurizer))

        assert self._noise_mask_featurizer is not None
        pipeline_config_list.append(self._noise_mask_featurizer)

        if len(self._center_of_mass_featurizer) > 0:
            pipeline_config_list.append(ComposeFanoutFeaturizer(featurizers=self._center_of_mass_featurizer))

        if len(self._custom_conditioning_featurizers) > 0:
            for featurizer_list in self._custom_conditioning_featurizers:
                featurizer_type = get_featurizer_type(featurizer_list[0])
                for featurizer_instance in featurizer_list[1:]:
                    if get_featurizer_type(featurizer_instance) != featurizer_type:
                        raise ValueError(
                            f"All featurizers in a custom conditioning featurizer list must have the same type, got {get_featurizer_type(featurizer_instance)} and {featurizer_type}."
                        )
                pipeline_config_list.append(ComposeFanoutFeaturizer(featurizers=featurizer_list))

        if len(pipeline_config_list) > 1:
            return ComposeSequentialFeaturizer(featurizers=pipeline_config_list)
        return pipeline_config_list[0]

    def _build_featurization_pipeline(self) -> None:
        self._noise_mask_featurizer = NoiseMaskFeaturizer(design_task=self._design_task)
        self._featurization_pipeline = self._compose_featurization_pipeline()

    def _append_custom_conditioning_featurizers(
        self,
        *,
        design_conditioning: DesignConditioning,
        design_secondary_structure: dict[str, dict[SimplifiedDSSPFeature, ChainResidueRepr]] | None,
        index_type: Literal["auth_residue_index", "original_residue_index"],
    ) -> None:
        design_conditioning = design_conditioning or {}
        for condition in conditioning.CATEGORICAL_CONDITIONS:
            variants = design_conditioning.get(condition.key)
            if not variants:
                continue
            self._custom_conditioning_featurizers.append(
                [make_constant_write(key=condition.key, name=name, value=value) for name, value in variants.items()]
            )

        if design_secondary_structure is not None and design_secondary_structure != {}:
            self._custom_conditioning_featurizers.append(
                [
                    SecondaryStructureConditioningFeaturizer(
                        name=design_secondary_structure_name,
                        design_secondary_structure=design_secondary_structure,
                        index_type=index_type,
                    )
                    for design_secondary_structure_name, design_secondary_structure in design_secondary_structure.items()
                ]
            )

    def get_featurization_name(self, features_dict: dict[str, Features]) -> str:
        """Generate a unique name for each feature set in the pipeline based on key featurization steps.

        Subclasses must implement `get_featurization_name`.

        Parameters
        ----------
        features_dict : dict[str, Features]
            A dictionary of feature sets, keyed by the input features key.

        Returns
        -------
        str
            The pipeline name
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

        Raises
        ------
        ValueError
            If `output_format` is not a valid output format.
        """
        os.makedirs(output_dir, exist_ok=True)
        features_list: list[dict[str, Features]] = []
        for features_dict in self([Features()]):
            features_list.append(features_dict)
            pipeline_name = self.get_featurization_name(features_dict)
            for feat_key, features in features_dict.items():
                output_file_prefix = os.path.join(output_dir, f"{pipeline_name}___{feat_key}")
                save_fct: dict[str, Callable[..., None]] = {
                    "pdb": features.save_to_pdb,
                    "mmcif": features.save_to_mmcif,
                    "npz": features.save_to_npz,
                }
                output_file = f"{output_file_prefix}.{output_format}"
                kwargs = {} if output_format == "npz" else {"residue_index_type": "residue_index"}
                save_fct[output_format](output_file, **kwargs)
                logger.info(f"Saving features to {output_file}")

        return features_list

    def __call__(self, inputs: Iterable[Features]) -> Iterator[dict[str, Features]]:
        raise NotImplementedError("__call__ should be implemented by the subclass")
