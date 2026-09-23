#!/usr/bin/env python

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Dependency:
    """
    One dependency entry.

    import_name:
        Name used by Python import checks.
        Example: sklearn

    pip_name:
        Name used by pip.
        Example: scikit-learn

    version_spec:
        Optional version constraint.
        Example: >=1.2,<1.8
    """

    import_name: str
    pip_name: str
    version_spec: str | None = None


BASE_DEPENDENCIES = [
    Dependency("numpy", "numpy", ">=1.23,<2.0"),
    Dependency("pandas", "pandas", ">=1.5,<3.0"),
    Dependency("matplotlib", "matplotlib", ">=3.7,<3.11"),
    Dependency("sklearn", "scikit-learn", ">=1.2,<1.8"),
    Dependency("scipy", "scipy", ">=1.10,<1.17"),
]

NOTEBOOK_DEPENDENCIES = [
    Dependency("jupyter", "jupyter"),
    Dependency("ipykernel", "ipykernel"),
    Dependency("notebook", "notebook"),
]

SCANPY_DEPENDENCIES = [
    Dependency("scanpy", "scanpy", ">=1.9,<1.12"),
    Dependency("anndata", "anndata", ">=0.9,<0.12"),
]

PROBERT_DEPENDENCIES = [
    Dependency("torch_optimizer", "torch-optimizer", ">=0.3,<0.4"),
    Dependency("tape", "tape-proteins"),
    Dependency("umap", "umap-learn", ">=0.5,<0.6"),
]

ESMC_DEPENDENCIES = [
    Dependency("esm", "esm"),
]

DEV_DEPENDENCIES = [
    Dependency("pytest", "pytest", ">=7"),
    Dependency("ruff", "ruff", ">=0.4"),
    Dependency("build", "build", ">=1.0"),
    Dependency("twine", "twine", ">=5.0"),
]


DEPENDENCY_GROUPS = {
    "base": BASE_DEPENDENCIES,
    "notebook": NOTEBOOK_DEPENDENCIES,
    "scanpy": SCANPY_DEPENDENCIES,
    "probert": PROBERT_DEPENDENCIES,
    "esmc": ESMC_DEPENDENCIES,
    "dev": DEV_DEPENDENCIES,
}


def is_installed(import_name: str) -> bool:
    """
    Return True if import_name can be imported in the current environment.
    """
    return importlib.util.find_spec(import_name) is not None


def pip_requirement(dep: Dependency) -> str:
    """
    Convert a Dependency object into a pip requirement string.
    """
    if dep.version_spec is None:
        return dep.pip_name

    return f"{dep.pip_name}{dep.version_spec}"


def unique_dependencies(dependencies: Iterable[Dependency]) -> list[Dependency]:
    """
    Remove duplicate dependencies while preserving order.

    Duplicate detection is based on import_name.
    """
    seen = set()
    unique = []

    for dep in dependencies:
        if dep.import_name in seen:
            continue

        seen.add(dep.import_name)
        unique.append(dep)

    return unique


def collect_dependencies(groups: list[str]) -> list[Dependency]:
    """
    Collect dependency objects from selected groups.
    """
    selected: list[Dependency] = []

    for group in groups:
        if group == "all":
            for deps in DEPENDENCY_GROUPS.values():
                selected.extend(deps)
            continue

        if group not in DEPENDENCY_GROUPS:
            valid = sorted(list(DEPENDENCY_GROUPS) + ["all"])
            raise ValueError(
                f"Unknown dependency group: {group!r}. "
                f"Valid groups are: {valid}"
            )

        selected.extend(DEPENDENCY_GROUPS[group])

    return unique_dependencies(selected)


def check_version_conflicts() -> None:
    """
    Print warnings for packages that are installed but likely too new
    for this project.

    This script only installs missing packages. It does not downgrade
    already-installed packages automatically.
    """
    try:
        from importlib.metadata import version
    except ImportError:
        return

    warnings = []

    version_rules = {
        "numpy": ("2.0", "numpy>=2.0 may break older scanpy / tape / bioinformatics stacks."),
        "pandas": ("3.0", "pandas>=3.0 may break older code expecting pandas 1.x/2.x behavior."),
        "scipy": ("1.17", "scipy>=1.17 may be too new for some pinned bioinformatics stacks."),
        "scikit-learn": ("1.8", "scikit-learn>=1.8 may be too new for older downstream code."),
    }

    for dist_name, (bad_min_version, message) in version_rules.items():
        try:
            installed_version = version(dist_name)
        except Exception:
            continue

        if version_gte(installed_version, bad_min_version):
            warnings.append(
                f"{dist_name} {installed_version}: {message}"
            )

    if warnings:
        print("\nVersion warnings:")
        for msg in warnings:
            print(f"  WARNING: {msg}")

        print(
            "\nThis installer only installs missing packages.\n"
            "If these versions cause problems, create a clean environment or manually downgrade, for example:\n"
            "  python -m pip install 'numpy>=1.23,<2.0' 'pandas>=1.5,<3.0'\n"
        )


def version_gte(current: str, minimum: str) -> bool:
    """
    Lightweight version comparison.

    Returns True when current >= minimum.

    This avoids requiring packaging as a dependency before installation.
    """
    def parse(v: str) -> tuple[int, ...]:
        parts = []
        for token in v.replace("-", ".").split("."):
            number = ""
            for char in token:
                if char.isdigit():
                    number += char
                else:
                    break

            if number == "":
                break

            parts.append(int(number))

        return tuple(parts)

    current_tuple = parse(current)
    minimum_tuple = parse(minimum)

    max_len = max(len(current_tuple), len(minimum_tuple))
    current_tuple = current_tuple + (0,) * (max_len - len(current_tuple))
    minimum_tuple = minimum_tuple + (0,) * (max_len - len(minimum_tuple))

    return current_tuple >= minimum_tuple


def warn_if_torch_needed(groups: list[str]) -> None:
    """
    Warn users that torch should be installed manually for ProBERT/ESMC workflows.

    We intentionally do not auto-install torch because pip may choose a huge
    or incompatible CUDA build.
    """
    needs_torch = any(group in {"probert", "esmc", "all"} for group in groups)

    if not needs_torch:
        return

    if is_installed("torch"):
        try:
            import torch

            print("\nTorch check:")
            print(f"  torch version: {torch.__version__}")
            print(f"  CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"  CUDA version used by torch: {torch.version.cuda}")
                print(f"  GPU count: {torch.cuda.device_count()}")
        except Exception as exc:
            print(f"\nWARNING: torch is installed but could not be imported cleanly: {exc}")
        return

    print(
        "\nWARNING: torch is not installed.\n"
        "ProBERT and ESMC workflows require PyTorch, but this script will not install torch automatically.\n"
        "Install PyTorch manually first using the command that matches your CPU/CUDA setup.\n"
        "CPU example:\n"
        "  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu\n"
        "\nAfter installing torch, rerun this script.\n"
    )


def install_packages(requirements: list[str], dry_run: bool = False) -> None:
    """
    Install missing packages using the current Python executable.
    """
    if not requirements:
        print("\nNo missing packages to install.")
        return

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        *requirements,
    ]

    print("\nInstalling missing packages:")
    for requirement in requirements:
        print(f"  - {requirement}")

    print("\nCommand:")
    print(" ".join(command))

    if dry_run:
        print("\nDry run enabled. Nothing was installed.")
        return

    subprocess.check_call(command)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check the current Python environment and install only missing "
            "dependencies for my_package."
        )
    )

    parser.add_argument(
        "--groups",
        nargs="+",
        default=["base"],
        choices=sorted(list(DEPENDENCY_GROUPS) + ["all"]),
        help=(
            "Dependency groups to check/install. "
            "Examples: base notebook scanpy probert esmc dev all"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what is missing. Do not install anything.",
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Install without asking for confirmation.",
    )

    parser.add_argument(
        "--skip-version-warnings",
        action="store_true",
        help="Do not print warnings about already-installed package versions.",
    )

    args = parser.parse_args()

    print("Checking environment:")
    print(f"  Python executable: {sys.executable}")
    print(f"  Dependency groups: {args.groups}")

    if not args.skip_version_warnings:
        check_version_conflicts()

    warn_if_torch_needed(args.groups)

    dependencies = collect_dependencies(args.groups)

    installed: list[Dependency] = []
    missing: list[Dependency] = []

    for dep in dependencies:
        if is_installed(dep.import_name):
            installed.append(dep)
        else:
            missing.append(dep)

    print("\nAlready installed:")
    if installed:
        for dep in installed:
            print(f"  ✓ {dep.import_name}")
    else:
        print("  None")

    print("\nMissing:")
    if missing:
        for dep in missing:
            print(f"  ✗ {dep.import_name}  ->  {pip_requirement(dep)}")
    else:
        print("  None")

    requirements = [pip_requirement(dep) for dep in missing]

    if not requirements:
        print("\nEnvironment is ready.")
        return

    if args.dry_run:
        install_packages(requirements, dry_run=True)
        return

    if not args.yes:
        answer = input("\nInstall missing packages now? [y/N]: ").strip().lower()

        if answer not in {"y", "yes"}:
            print("Installation cancelled.")
            return

    install_packages(requirements, dry_run=False)

    print("\nDone. Missing dependencies were installed.")


if __name__ == "__main__":
    main()