import subprocess
import tarfile
import zipfile
from email import policy
from email.message import Message
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

EXPECTED_LICENSE_FILES = {
    "LICENSE/LICENSE",
    "LICENSE/LICENSES/BOLTZ",
    "LICENSE/LICENSES/FLASH_IPA",
    "LICENSE/LICENSES/FRAMEFLOW",
    "LICENSE/LICENSES/LAPROTEINA",
    "LICENSE/LICENSES/MULTIFLOW",
    "LICENSE/LICENSES/OPENFOLD",
    "LICENSE/LICENSES/PROTEINDOCKING",
    "LICENSE/NOTICE",
    "LICENSE/THIRD-PARTY-NOTICES",
}

FORBIDDEN_BASE_DEPENDENCIES = {"ipykernel", "ipywidgets", "mdtraj", "wandb"}
FORBIDDEN_WHEEL_PATHS = {
    PurePosixPath("dynopsi/_vendor"),
    PurePosixPath("dynopsi/data/annotation"),
    PurePosixPath("dynopsi/eval"),
    PurePosixPath("dynopsi/training"),
}


class DistributionArtifacts(NamedTuple):
    metadata: Message
    wheel_license_files: set[str]
    wheel_members: set[str]
    sdist_license_files: set[str]


@pytest.fixture(scope="module")
def distribution_artifacts(tmp_path_factory: pytest.TempPathFactory) -> DistributionArtifacts:
    build_dir = tmp_path_factory.mktemp("distribution-artifacts")
    repo_root = Path(__file__).resolve().parents[2]
    subprocess.run(
        ["uv", "build", "--out-dir", str(build_dir)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    wheel_path = next(build_dir.glob("*.whl"))
    with zipfile.ZipFile(wheel_path) as wheel:
        metadata_path = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = BytesParser(policy=policy.default).parsebytes(wheel.read(metadata_path))
        wheel_license_files = {
            name.split(".dist-info/licenses/", maxsplit=1)[1]
            for name in wheel.namelist()
            if ".dist-info/licenses/" in name and not name.endswith("/")
        }
        wheel_members = set(wheel.namelist())

    sdist_path = next(build_dir.glob("*.tar.gz"))
    with tarfile.open(sdist_path, "r:gz") as sdist:
        sdist_license_files = {
            PurePosixPath(*PurePosixPath(member.name).parts[1:]).as_posix()
            for member in sdist.getmembers()
            if member.isfile()
            and len(PurePosixPath(member.name).parts) > 2
            and PurePosixPath(member.name).parts[1] == "LICENSE"
        }

    return DistributionArtifacts(metadata, wheel_license_files, wheel_members, sdist_license_files)


def _is_base_requirement(requirement: Requirement) -> bool:
    return requirement.marker is None or requirement.marker.evaluate({"extra": ""})


def test_distribution_artifacts_include_legal_files(distribution_artifacts: DistributionArtifacts) -> None:
    assert distribution_artifacts.wheel_license_files == EXPECTED_LICENSE_FILES
    assert distribution_artifacts.metadata["License-Expression"] == "Apache-2.0"
    assert set(distribution_artifacts.metadata.get_all("License-File", [])) == EXPECTED_LICENSE_FILES
    assert distribution_artifacts.sdist_license_files == EXPECTED_LICENSE_FILES


def test_wheel_dependencies_match_release_contract(distribution_artifacts: DistributionArtifacts) -> None:
    requirements = [Requirement(value) for value in distribution_artifacts.metadata.get_all("Requires-Dist", [])]
    base_requirements = [requirement for requirement in requirements if _is_base_requirement(requirement)]
    base_dependency_names = {canonicalize_name(requirement.name) for requirement in base_requirements}
    assert {"numpy", "pyyaml"} <= base_dependency_names

    forbidden_base_requirements = {
        str(requirement)
        for requirement in base_requirements
        if canonicalize_name(requirement.name) in FORBIDDEN_BASE_DEPENDENCIES
    }
    assert not forbidden_base_requirements

    torch_requirements = [
        requirement for requirement in base_requirements if canonicalize_name(requirement.name) == "torch"
    ]
    assert len(torch_requirements) == 1
    assert torch_requirements[0].specifier == SpecifierSet(">=2.6.0,<2.8")


def test_wheel_excludes_internal_packages(distribution_artifacts: DistributionArtifacts) -> None:
    forbidden_wheel_members = {
        name
        for name in distribution_artifacts.wheel_members
        if any(
            PurePosixPath(name) == forbidden_path or forbidden_path in PurePosixPath(name).parents
            for forbidden_path in FORBIDDEN_WHEEL_PATHS
        )
    }
    assert not forbidden_wheel_members


@pytest.mark.parametrize(
    ("requirement_value", "is_forbidden_base_requirement"),
    [
        ("wandb", True),
        ('ipywidgets ; python_version >= "3.12"', True),
        ("wandb>=0.15", True),
        ("ipykernel==6.0", True),
        ("mdtraj<2,>=1.9", True),
        ('mdtraj; extra == "training"', False),
    ],
)
def test_forbidden_base_dependency_detection(requirement_value: str, is_forbidden_base_requirement: bool) -> None:
    requirement = Requirement(requirement_value)
    detected = canonicalize_name(requirement.name) in FORBIDDEN_BASE_DEPENDENCIES and _is_base_requirement(requirement)
    assert detected is is_forbidden_base_requirement
