"""Public entry points load only modules from this package."""

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

_SUBPROCESS_SCRIPT = """
import sys
from pathlib import Path

import dynopsi
from dynopsi import DesignPipeline  # noqa: F401
import dynopsi.cli  # noqa: F401

package_root = Path(dynopsi.__file__).resolve().parent
leaked = []
for name, module in sys.modules.items():
    if name != "dynopsi" and not name.startswith("dynopsi."):
        continue
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        continue
    if not Path(module_file).resolve().is_relative_to(package_root):
        leaked.append(name)
if leaked:
    raise AssertionError("imported modules outside this package: " + repr(sorted(leaked)))
print("INFERENCE_IMPORT_OK")
"""


def test_public_imports_stay_inside_the_package() -> None:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in [str(_REPO_ROOT), *sys.path] if p)}
    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env=env,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "INFERENCE_IMPORT_OK" in result.stdout
