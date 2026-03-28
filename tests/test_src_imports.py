import os
import ast
import sys
import pytest
from pathlib import Path

# Base directory of the repo
REPO_ROOT = Path(__file__).parent.parent
SRC_DIR = REPO_ROOT / "src"
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"

def get_stdlib_modules():
    """Get a set of standard library module names."""
    if sys.version_info >= (3, 10):
        return sys.stdlib_module_names
    else:
        return set(sys.builtin_module_names)

STDLIB_MODULES = get_stdlib_modules()

def get_install_requires():
    """Parse dependencies from pyproject.toml without executing it."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    with open(PYPROJECT_TOML, "rb") as f:
        data = tomllib.load(f)

    return set(data.get("project", {}).get("dependencies", []))

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
                    # Skip internal imports (shroom package itself)
                    if imp == "shroom" or imp.startswith("."):
                        continue

                    # Skip stdlib
                    if imp in STDLIB_MODULES:
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
