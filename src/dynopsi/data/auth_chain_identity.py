"""``auth_asym_id`` is a table-key int, not a letter encoding.

Unique ``[A-Z]`` IDs keep letter-map ints; others use ``OPAQUE_AUTH_ASYM_ID_START+``.
Do not use ``get_asym_id_from_chain_id`` for auth identity.
"""

import string
from collections.abc import Iterable, Mapping

import numpy as np

CHAIN_ALPHABET = string.ascii_uppercase + string.ascii_lowercase
SINGLE_LETTER_CHAIN_TO_ASYM_ID_MAP = dict(zip(CHAIN_ALPHABET, range(len(CHAIN_ALPHABET))))
ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP = {v: k for k, v in SINGLE_LETTER_CHAIN_TO_ASYM_ID_MAP.items()}

UNIQUE_LETTER_AUTH_IDS = frozenset(string.ascii_uppercase)
AUTH_CHAIN_ID_MAX_LENGTH = 16
# Letter map is A-Z then a-z (0-51).
OPAQUE_AUTH_ASYM_ID_START = max(ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP) + 1


def is_unique_single_letter_auth_id(auth_id: str) -> bool:
    return len(auth_id) == 1 and auth_id in UNIQUE_LETTER_AUTH_IDS


def checked_auth_chain_id(auth_chain_id: str) -> str:
    """Raise when an author chain id is longer than ``AUTH_CHAIN_ID_MAX_LENGTH``."""
    if len(auth_chain_id) > AUTH_CHAIN_ID_MAX_LENGTH:
        raise ValueError(
            f"auth_chain_id {auth_chain_id!r} is longer than the supported maximum of {AUTH_CHAIN_ID_MAX_LENGTH}."
        )
    return auth_chain_id


def assign_auth_asym_id(auth_id: str, used: set[int]) -> int:
    if is_unique_single_letter_auth_id(auth_id):
        letter_int = SINGLE_LETTER_CHAIN_TO_ASYM_ID_MAP[auth_id]
        if letter_int not in used:
            return letter_int
    candidate = OPAQUE_AUTH_ASYM_ID_START
    while candidate in used:
        candidate += 1
    return candidate


def auth_chain_id_table_from_chains(chains: np.ndarray) -> dict[int, str]:
    return {int(chain["auth_asym_id"]): str(chain["auth_chain_id"]) for chain in chains}


def lookup_auth_chain_id(auth_asym_id: int, table: Mapping[int, str]) -> str:
    """Lookup requires a table entry; letter-map ints are not inferred."""
    auth_asym_id = int(auth_asym_id)
    if auth_asym_id in table:
        return table[auth_asym_id]
    raise KeyError(f"No persisted auth chain id for auth_asym_id={auth_asym_id}")


def lookup_auth_asym_id(chain_id: str, table: Mapping[int, str]) -> int | None:
    for auth_int, auth_id in table.items():
        if auth_id == chain_id:
            return int(auth_int)
    return None


def remap_auth_chain_id_table(table: Mapping[int, str], old_to_new: Mapping[int, int]) -> dict[int, str]:
    return {int(old_to_new.get(old, old)): auth_id for old, auth_id in table.items()}


def merge_auth_chain_id_tables(tables: Iterable[Mapping[int, str]]) -> dict[int, str]:
    """Merge tables in order; later entries overwrite the same key."""
    merged: dict[int, str] = {}
    for table in tables:
        merged.update({int(k): v for k, v in table.items()})
    return merged


def persisted_auth_ids(auth_asym_ids: Iterable[int], table: Mapping[int, str]) -> list[str]:
    return [lookup_auth_chain_id(auth_asym_id, table) for auth_asym_id in dict.fromkeys(int(i) for i in auth_asym_ids)]


def first_unused_auth_letters(num_letters: int, persisted_ids: Iterable[str]) -> tuple[str, ...]:
    """First unused ``[A-Z]`` letters among persisted auth ID strings, not ints."""
    used_letters = {auth_id for auth_id in persisted_ids if is_unique_single_letter_auth_id(auth_id)}
    letters: list[str] = []
    for letter in string.ascii_uppercase:
        if letter in used_letters:
            continue
        letters.append(letter)
        if len(letters) == num_letters:
            return tuple(letters)
    raise ValueError(f"Not enough unused [A-Z] auth IDs. Need {num_letters}, already used {sorted(used_letters)}.")


def letter_stable_auth_ints(letters: Iterable[str]) -> tuple[int, ...]:
    return tuple(SINGLE_LETTER_CHAIN_TO_ASYM_ID_MAP[letter] for letter in letters)


def resolve_unique_letter_auth_int(chain_id: str, table: Mapping[int, str]) -> int:
    """Resolve chain id to auth_asym_id: table first, then letter map for unique [A-Z]; other IDs must be persisted."""
    table_int = lookup_auth_asym_id(chain_id, table)
    if table_int is not None:
        return table_int
    if is_unique_single_letter_auth_id(chain_id):
        return SINGLE_LETTER_CHAIN_TO_ASYM_ID_MAP[chain_id]
    raise KeyError(f"No persisted auth_asym_id for chain id {chain_id!r}")
