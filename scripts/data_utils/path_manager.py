import sys
import os
import os.path as path

HERE_PATH = path.normpath(path.dirname(__file__))
PROJECT_ROOT = path.normpath(path.join(HERE_PATH, '..'))
SUBMODULES_PATH = path.join(PROJECT_ROOT, 'submodules')


def add_submodule_to_path(submodule_name):
    """Add a specified submodule to the Python path."""
    submodule_path = path.join(SUBMODULES_PATH, submodule_name)
    if path.isdir(submodule_path):
        if submodule_path not in sys.path:
            sys.path.insert(0, submodule_path)
    else:
        raise ImportError(
            f"Submodule {submodule_name} not found: {submodule_path}"
        )


def init_all_submodules():
    """Initialize paths for all submodules."""
    if not path.isdir(SUBMODULES_PATH):
        raise ImportError(f"Submodules directory not found: {SUBMODULES_PATH}")

    submodules = [d for d in os.listdir(SUBMODULES_PATH)
                  if path.isdir(path.join(SUBMODULES_PATH, d))]

    for submodule in submodules:
        add_submodule_to_path(submodule)

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
