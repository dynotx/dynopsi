from collections.abc import Mapping
from enum import Enum, IntEnum
from typing import ClassVar

STANDARD_RESIDUES = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
]


TOKENS = [
    "<pad>",
    "-",
    *STANDARD_RESIDUES,
]

TOKEN_IDS = {token: i for i, token in enumerate(TOKENS)}

AA_THREE_TO_ONE_MAP = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "UNK": "X",
}

AA_ONE_TO_THREE_MAP = {v: k for k, v in AA_THREE_TO_ONE_MAP.items()}

NONCANONICAL_RESIDUE_TO_NATURAL: Mapping[str, str] = {
    "NLE": "LEU",
    "NVA": "VAL",
    "HSE": "SER",
    "HCS": "CYS",
    "HAR": "ARG",
    "ABU": "ALA",
    "TLE": "LEU",
    "AIL": "ILE",
    "ALO": "THR",
    "SEC": "CYS",
    "MSE": "MET",
    "CIR": "ARG",
    "ORN": "LYS",
    "AIB": "ALA",
    "MLE": "LEU",
    "MVA": "VAL",
    "PFF": "PHE",
    "PHI": "PHE",
    "TYS": "TYR",
}


class AminoAcidSequenceFeature(IntEnum):
    """Compact protein sequence vocabulary used for target conditioning."""

    PAD = 0
    ALA = 1
    ARG = 2
    ASN = 3
    ASP = 4
    CYS = 5
    GLN = 6
    GLU = 7
    GLY = 8
    HIS = 9
    ILE = 10
    LEU = 11
    LYS = 12
    MET = 13
    PHE = 14
    PRO = 15
    SER = 16
    THR = 17
    TRP = 18
    TYR = 19
    VAL = 20
    UNK = 21
    MASK = 22

    @classmethod
    def from_resname(cls, resname: str) -> "AminoAcidSequenceFeature":
        if resname in NONCANONICAL_RESIDUE_TO_NATURAL:
            resname = NONCANONICAL_RESIDUE_TO_NATURAL[resname]
        return cls.__members__.get(resname, cls.UNK)


N_AMINO_ACID_SEQUENCE_FEATURES = len(AminoAcidSequenceFeature)


ATOM37_ORDER: tuple[str, ...] = (
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
)

ELEMENT_TO_ATOM_NAME = {
    "N": ["N", "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ"],
    "C": [
        "CA",
        "C",
        "CB",
        "CD",
        "CD1",
        "CD2",
        "CE",
        "CE1",
        "CE2",
        "CE3",
        "CG",
        "CG1",
        "CG2",
        "CH2",
        "CZ",
        "CZ2",
        "CZ3",
    ],
    "O": ["O", "OG", "OG1", "OD1", "OD2", "OH", "OXT", "OE1", "OE2"],
    "S": ["SG", "SD"],
}

ATOM_NAME_TO_ELEMENT = {
    atom_name: elem for elem, atom_names in ELEMENT_TO_ATOM_NAME.items() for atom_name in atom_names
}

REF_ATOM_NAMES = {
    "PAD": [],
    "UNK": ["N", "CA", "C", "O", "CB"],
    "-": [],
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],  # noqa: E501
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
}

RES_TO_DISTO_ATOM = {
    "UNK": "CB",
    "ALA": "CB",
    "ARG": "CB",
    "ASN": "CB",
    "ASP": "CB",
    "CYS": "CB",
    "GLN": "CB",
    "GLU": "CB",
    "GLY": "CA",
    "HIS": "CB",
    "ILE": "CB",
    "LEU": "CB",
    "LYS": "CB",
    "MET": "CB",
    "PHE": "CB",
    "PRO": "CB",
    "SER": "CB",
    "THR": "CB",
    "TRP": "CB",
    "TYR": "CB",
    "VAL": "CB",
}


class AtomFilter(Enum):
    FULL = "full"
    BB = "bb"
    CA = "ca"

    def to_atom_names(self) -> tuple[str, ...]:
        if self == AtomFilter.FULL:
            # atom 37 order
            return ATOM37_ORDER
        elif self == AtomFilter.BB:
            return ("N", "CA", "C")
        elif self == AtomFilter.CA:
            return ("CA",)
        else:
            raise ValueError(f"Invalid atom filter: {self}")

    def num_atoms(self) -> int:
        if self == AtomFilter.FULL:
            return 37
        elif self == AtomFilter.BB:
            return 3
        elif self == AtomFilter.CA:
            return 1
        else:
            raise ValueError(f"Invalid atom filter: {self}")


### Interface Constants
INTERFACE_ATOMIC_DISTANCE_THRESHOLD_SIDECHAIN = 5.0
INTERFACE_ATOMIC_DISTANCE_THRESHOLD_BACKBONE = 10.0
INTERFACE_ATOMIC_DISTANCE_THRESHOLD_CA = 12.0


#################
# Featurization #
#################
FEATURE_KEYS = [
    "asym_id",
    "auth_asym_id",
    "entity_id",
    "sym_id",
    "residue_index",
    "original_residue_index",
    "auth_residue_index",
    "token_index",
    "token_pad_mask",
    "token_resolved_mask",
    "frame_resolved_mask",
    "res_type",
    "pocket_feature",
    "inpaint_feature",
    "coords",
    "atom_pad_mask",
    "atom_resolved_mask",
    "frames_idx",
    "atom_to_token",
    "binder_mask",
    "target_mask",
]


class DesignDesignation(Enum):
    FIXED = "fixed"
    DESIGN = "design"


class PocketFeature(IntEnum):
    UNSPECIFIED = 0
    UNSELECTED = 1
    POCKET = 2
    BINDER = 3


class InpaintFeature(IntEnum):
    BINDER_INPAINT = 0
    BINDER_CONTEXT = 1
    TARGET_INPAINT = 2
    TARGET_CONTEXT = 3


n_inpaint_features = len(InpaintFeature)


class DSSPEncoding[FeatureT: IntEnum]:
    name: ClassVar[str]
    feature_enum: type[FeatureT]
    code_to_feature: Mapping[str, FeatureT]

    @classmethod
    def feature_for_code(cls, code: str) -> FeatureT:
        return cls.code_to_feature[code]

    @classmethod
    def num_features(cls) -> int:
        return len(cls.feature_enum)

    @classmethod
    def get_mask_feature(cls) -> FeatureT:
        return cls.feature_enum["MASK_OR_MISSING"]


class FullDSSPFeature(IntEnum):
    MASK_OR_MISSING = 0
    COIL = 1
    HELIX = 2
    BETA_BRIDGE = 3
    STRAND = 4
    THREE_TURN_HELIX = 5
    FIVE_TURN_HELIX = 6
    TURN = 7
    BEND = 8


class FullDSSPEncoding(DSSPEncoding[FullDSSPFeature]):
    name = "full_dssp"
    feature_enum = FullDSSPFeature
    code_to_feature = {
        " ": FullDSSPFeature.COIL,
        "H": FullDSSPFeature.HELIX,
        "B": FullDSSPFeature.BETA_BRIDGE,
        "E": FullDSSPFeature.STRAND,
        "G": FullDSSPFeature.THREE_TURN_HELIX,
        "I": FullDSSPFeature.FIVE_TURN_HELIX,
        "T": FullDSSPFeature.TURN,
        "S": FullDSSPFeature.BEND,
    }


class SimplifiedDSSPFeature(IntEnum):
    MASK_OR_MISSING = 0
    LOOP_OR_OTHER = 1
    HELIX = 2
    STRAND = 3


DSSP_CODE_TO_SIMPLIFIED_FEATURE: dict[str, SimplifiedDSSPFeature] = {
    "H": SimplifiedDSSPFeature.HELIX,
    "G": SimplifiedDSSPFeature.HELIX,
    "I": SimplifiedDSSPFeature.HELIX,
    "E": SimplifiedDSSPFeature.STRAND,
    "B": SimplifiedDSSPFeature.STRAND,
    "T": SimplifiedDSSPFeature.LOOP_OR_OTHER,
    "S": SimplifiedDSSPFeature.LOOP_OR_OTHER,
    " ": SimplifiedDSSPFeature.LOOP_OR_OTHER,
}


class SimplifiedDSSPEncoding(DSSPEncoding[SimplifiedDSSPFeature]):
    name = "simplified"
    feature_enum = SimplifiedDSSPFeature
    code_to_feature = {
        "H": SimplifiedDSSPFeature.HELIX,
        "G": SimplifiedDSSPFeature.HELIX,
        "I": SimplifiedDSSPFeature.HELIX,
        "E": SimplifiedDSSPFeature.STRAND,
        "B": SimplifiedDSSPFeature.STRAND,
        "T": SimplifiedDSSPFeature.LOOP_OR_OTHER,
        "S": SimplifiedDSSPFeature.LOOP_OR_OTHER,
        " ": SimplifiedDSSPFeature.LOOP_OR_OTHER,
    }


class DSSPLoopFeature(IntEnum):
    MASK_OR_MISSING = 0
    NON_LOOP = 1
    LOOP_OR_OTHER = 2


class DSSPLoopEncoding(DSSPEncoding[DSSPLoopFeature]):
    name = "loop_aware_only"
    feature_enum = DSSPLoopFeature
    code_to_feature = {
        "H": DSSPLoopFeature.NON_LOOP,
        "G": DSSPLoopFeature.NON_LOOP,
        "I": DSSPLoopFeature.NON_LOOP,
        "E": DSSPLoopFeature.NON_LOOP,
        "B": DSSPLoopFeature.NON_LOOP,
        "T": DSSPLoopFeature.LOOP_OR_OTHER,
        "S": DSSPLoopFeature.LOOP_OR_OTHER,
        " ": DSSPLoopFeature.LOOP_OR_OTHER,
    }


class HomooligomerPartnerFeature(IntEnum):
    MASK_OR_MISSING = 0
    NO_PARTNER = 1
    HAS_PARTNER = 2


class DSSPBlockPairFeature(IntEnum):
    NON_ADJACENT_BLOCK = 0
    SAME_BLOCK = 1
    ADJACENT_BLOCK = 2


n_dssp_block_pair_features = len(DSSPBlockPairFeature)


class DesignTask(IntEnum):
    MONOMER_NOISE_BINDER_INPAINT = 0
    MONOMER_NOISE_TARGET_CHAIN_NO_IPT = 4
    MONOMER_NOISE_BINDER_CHAIN_NO_IPT = 5
    MULTIMER_TARGET_DOCKING = 7
    MULTIMER_BINDER_DESIGN = 10
    MULTIMER_TARGET_DOCKING_INPAINT_BINDER = 14
    MULTIMER_DOCKING_NOISE_ALL = 15
    MULTIMER_BINDER_DESIGN_NOISE_ALL = 16

    def get_noise_bool(self, inpaint_feature: InpaintFeature) -> bool | None:
        task_to_feature_noise: dict[DesignTask, dict[InpaintFeature, bool | None]] = {
            DesignTask.MONOMER_NOISE_BINDER_INPAINT: {
                InpaintFeature.BINDER_INPAINT: True,
                InpaintFeature.BINDER_CONTEXT: False,
                InpaintFeature.TARGET_INPAINT: None,
                InpaintFeature.TARGET_CONTEXT: None,
            },
            DesignTask.MONOMER_NOISE_TARGET_CHAIN_NO_IPT: {
                InpaintFeature.BINDER_INPAINT: None,
                InpaintFeature.BINDER_CONTEXT: None,
                InpaintFeature.TARGET_INPAINT: None,
                InpaintFeature.TARGET_CONTEXT: True,
            },
            DesignTask.MONOMER_NOISE_BINDER_CHAIN_NO_IPT: {
                InpaintFeature.BINDER_INPAINT: None,
                InpaintFeature.BINDER_CONTEXT: True,
                InpaintFeature.TARGET_INPAINT: None,
                InpaintFeature.TARGET_CONTEXT: None,
            },
            DesignTask.MULTIMER_TARGET_DOCKING: {
                InpaintFeature.BINDER_INPAINT: None,
                InpaintFeature.BINDER_CONTEXT: False,
                InpaintFeature.TARGET_INPAINT: None,
                InpaintFeature.TARGET_CONTEXT: True,
            },
            DesignTask.MULTIMER_BINDER_DESIGN: {
                InpaintFeature.BINDER_INPAINT: True,
                InpaintFeature.BINDER_CONTEXT: None,
                InpaintFeature.TARGET_INPAINT: None,
                InpaintFeature.TARGET_CONTEXT: False,
            },
            DesignTask.MULTIMER_TARGET_DOCKING_INPAINT_BINDER: {
                InpaintFeature.BINDER_INPAINT: True,
                InpaintFeature.BINDER_CONTEXT: False,
                InpaintFeature.TARGET_INPAINT: None,
                InpaintFeature.TARGET_CONTEXT: True,
            },
            DesignTask.MULTIMER_DOCKING_NOISE_ALL: {
                InpaintFeature.BINDER_INPAINT: None,
                InpaintFeature.BINDER_CONTEXT: True,
                InpaintFeature.TARGET_INPAINT: None,
                InpaintFeature.TARGET_CONTEXT: True,
            },
            DesignTask.MULTIMER_BINDER_DESIGN_NOISE_ALL: {
                InpaintFeature.BINDER_INPAINT: True,
                InpaintFeature.BINDER_CONTEXT: None,
                InpaintFeature.TARGET_INPAINT: None,
                InpaintFeature.TARGET_CONTEXT: True,
            },
        }
        return task_to_feature_noise[self][inpaint_feature]
