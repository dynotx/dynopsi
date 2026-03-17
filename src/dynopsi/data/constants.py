from enum import Enum, IntEnum

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


class DesignTask(IntEnum):
    MULTIMER_BINDER_DESIGN = 9

    def get_noise_bool(self, inpaint_feature: InpaintFeature) -> bool | None:
        task_to_feature_noise: dict[DesignTask, dict[InpaintFeature, bool | None]] = {
            DesignTask.MULTIMER_BINDER_DESIGN: {
                InpaintFeature.BINDER_INPAINT: True,
                InpaintFeature.BINDER_CONTEXT: None,
                InpaintFeature.TARGET_INPAINT: None,
                InpaintFeature.TARGET_CONTEXT: False,
            },
        }
        return task_to_feature_noise[self][inpaint_feature]
