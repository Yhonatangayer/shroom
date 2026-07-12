import os
import ast
import sys
import sysconfig
import importlib.util
import pytest
from pathlib import Path

# Base directory of the repo
REPO_ROOT = Path(__file__).parent.parent
SRC_DIR = REPO_ROOT / "src"
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"

_STDLIB_PATHS = tuple(
    p for p in {sysconfig.get_paths().get("stdlib"),
                sysconfig.get_paths().get("platstdlib")} if p
)
# site-packages lives *under* the stdlib dir in a venv, so third-party packages
# must be explicitly excluded from the stdlib-path check below.
_SITE_PATHS = tuple(
    p for p in {sysconfig.get_paths().get("purelib"),
                sysconfig.get_paths().get("platlib")} if p
)


def _is_under(path: str, roots) -> bool:
    path = os.path.realpath(path)
    for root in roots:
        root = os.path.realpath(root)
        if path == root or path.startswith(root + os.sep):
            return True
    return False


def is_stdlib_module(name: str) -> bool:
    """Return True if `name` is a standard-library or built-in module.

    On Python >= 3.10 this uses ``sys.stdlib_module_names``. On older versions
    ``sys.builtin_module_names`` only lists C built-ins and misses pure-Python
    stdlib modules (e.g. ``pickle``), so we locate the module and check whether
    it lives under the interpreter's stdlib path (but not site-packages).
    """
    if sys.version_info >= (3, 10):
        return name in sys.stdlib_module_names
    if name in sys.builtin_module_names:
        return True
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, ModuleNotFoundError, AttributeError):
        return False
    if spec is None:
        return False
    if spec.origin in ("built-in", "frozen"):
        return True
    origin = spec.origin
    if origin is None:
        locations = list(spec.submodule_search_locations or [])
        origin = locations[0] if locations else None
    if not origin:
        return False
    return _is_under(origin, _STDLIB_PATHS) and not _is_under(origin, _SITE_PATHS)

def get_install_requires():
    """Parse dependencies from pyproject.toml, including optional-dependencies."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    with open(PYPROJECT_TOML, "rb") as f:
        data = tomllib.load(f)

    project = data.get("project", {})
    deps = set(project.get("dependencies", []))
    for extra_deps in project.get("optional-dependencies", {}).values():
        deps.update(extra_deps)
    return deps

def get_imports_from_file(filepath):
    """Extract imported module names from a python file."""
    with open(filepath, "r") as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
    return imports

def test_imports_vs_requirements():
    """
    Check that all third-party imports in src/ are listed in pyproject.toml dependencies.
    """
    if not os.path.exists(SRC_DIR):
        pytest.skip("src directory not found")

    requirements = get_install_requires()
    # Normalize requirements (handle 'numpy>=1.2' etc)
    requirements_names = {req.split('>')[0].split('<')[0].split('=')[0].strip() for req in requirements}

    # Map import name to package name
    import_map = {
        "yaml": "pyyaml",
        "sklearn": "scikit-learn",
        "cv2": "opencv-python",
        "mpl_toolkits": "matplotlib",
        "pra": "pyroomacoustics",
    }

    # Walk src directory
    missing_deps = []

    for root, _, files in os.walk(SRC_DIR):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                imports = get_imports_from_file(filepath)

                for imp in imports:
                    # Skip internal imports (first-party packages under src/)
                    if imp in {"shroom", "shroom_dev"} or imp.startswith("."):
                        continue

                    # Skip stdlib / built-in modules
                    if is_stdlib_module(imp):
                        continue

                    # Skip known built-ins that might not be in stdlib list
                    if imp in {"typing", "abc", "pathlib", "os", "sys", "ast",
                               "warnings", "copy", "importlib", "collections",
                               "functools", "itertools", "math", "re"}:
                        continue

                    # Map import to package name
                    pkg_name = import_map.get(imp, imp)

                    # Check if in requirements
                    if pkg_name not in requirements_names:
                        missing_deps.append(
                            f"File: {os.path.relpath(filepath, REPO_ROOT)} "
                            f"imports '{imp}' (package '{pkg_name}')"
                        )

    if missing_deps:
        error_msg = (
            "Found imports in src/ that are missing from pyproject.toml dependencies:\n" +
            "\n".join(missing_deps) +
            "\n\nPlease add these packages to [project] dependencies in pyproject.toml."
        )
        pytest.fail(error_msg)
